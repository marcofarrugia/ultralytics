# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import C2PSA, C2PSA_CAFM, CAFMAttention
from ultralytics.nn.modules.block import PSABlockCAFM


def _conv3d(x: torch.Tensor, layer: torch.nn.Conv3d) -> torch.Tensor:
    """Apply a Conv3d layer directly through the functional API."""
    return F.conv3d(x, layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)


def _reference_cafm(module: CAFMAttention, x: torch.Tensor) -> torch.Tensor:
    """Calculate the source CAFM local and global branches directly."""
    b, c, h, w = x.shape
    qkv = _conv3d(_conv3d(x.unsqueeze(2), module.qkv), module.qkv_dwconv).squeeze(2)

    local = qkv.reshape(b, h * w, 3 * module.num_heads, module.head_dim).permute(0, 2, 1, 3)
    local = _conv3d(local.unsqueeze(2), module.fc).squeeze(2)
    local = local.permute(0, 3, 1, 2).reshape(b, 9 * module.head_dim, h, w)
    local = _conv3d(local.unsqueeze(2), module.dep_conv).squeeze(2)

    q, k, v = qkv.chunk(3, dim=1)
    q = F.normalize(q.reshape(b, module.num_heads, module.head_dim, h * w), dim=-1)
    k = F.normalize(k.reshape(b, module.num_heads, module.head_dim, h * w), dim=-1)
    v = v.reshape(b, module.num_heads, module.head_dim, h * w)
    attention = ((q @ k.transpose(-2, -1)) * module.temperature).softmax(dim=-1)
    global_features = (attention @ v).reshape(b, c, h, w)
    global_features = _conv3d(global_features.unsqueeze(2), module.project_out).squeeze(2)
    return global_features + local


def test_cafm_matches_direct_local_and_global_reference() -> None:
    """CAFM should agree with a direct implementation of both source branches."""
    torch.manual_seed(7)
    module = CAFMAttention(dim=8, num_heads=2, bias=False).double()
    x = torch.randn(2, 8, 3, 4, dtype=torch.float64)

    torch.testing.assert_close(module(x), _reference_cafm(module, x), rtol=1e-12, atol=1e-12)


def test_cafm_has_no_internal_identity_residual() -> None:
    """Zero CAFM parameters should produce zero rather than returning its input."""
    module = CAFMAttention(dim=8, num_heads=2, bias=False)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    x = torch.randn(1, 8, 3, 4)
    output = module(x)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert not torch.equal(output, x)


@pytest.mark.parametrize(("dim", "num_heads"), ((128, 2), (256, 4), (384, 6)))
def test_cafm_preserves_shape_and_has_finite_gradients(dim: int, num_heads: int) -> None:
    """Supported scale configurations should preserve NCHW shape and backpropagate finite gradients."""
    module = CAFMAttention(dim=dim, num_heads=num_heads, bias=False)
    x = torch.randn(1, dim, 3, 5, requires_grad=True)
    output = module(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.parametrize(("dim", "num_heads"), ((128, 2), (256, 4), (384, 6)))
def test_cafm_temperature_kernel_and_grouping_contracts(dim: int, num_heads: int) -> None:
    """The module should retain the official projection, kernel, bias, and grouped-convolution contracts."""
    module = CAFMAttention(dim=dim, num_heads=num_heads, bias=False)
    head_dim = dim // num_heads

    assert module.temperature.shape == (num_heads, 1, 1)
    assert module.qkv.kernel_size == (1, 1, 1)
    assert module.qkv.in_channels == dim and module.qkv.out_channels == 3 * dim
    assert module.qkv.groups == 1 and module.qkv.bias is None
    assert module.qkv_dwconv.kernel_size == (3, 3, 3)
    assert module.qkv_dwconv.padding == (1, 1, 1)
    assert module.qkv_dwconv.groups == 3 * dim and module.qkv_dwconv.bias is None
    assert module.project_out.kernel_size == (1, 1, 1)
    assert module.project_out.in_channels == dim and module.project_out.out_channels == dim
    assert module.project_out.bias is None
    assert module.fc.kernel_size == (1, 1, 1)
    assert module.fc.in_channels == 3 * num_heads and module.fc.out_channels == 9
    assert module.fc.bias is not None
    assert module.dep_conv.kernel_size == (3, 3, 3)
    assert module.dep_conv.padding == (1, 1, 1)
    assert module.dep_conv.in_channels == 9 * head_dim and module.dep_conv.out_channels == dim
    assert module.dep_conv.groups == head_dim and module.dep_conv.bias is not None


def test_singleton_depth_outer_kernel_planes_have_zero_gradients() -> None:
    """Only the center depth plane should learn when a 3D kernel processes a singleton depth axis."""
    torch.manual_seed(11)
    module = CAFMAttention(dim=8, num_heads=2, bias=False)
    x = torch.randn(1, 8, 4, 5, requires_grad=True)
    module(x).square().mean().backward()

    for layer in (module.qkv_dwconv, module.dep_conv):
        gradient = layer.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[:, :, 0]) == 0
        assert torch.count_nonzero(gradient[:, :, 2]) == 0
        assert torch.count_nonzero(gradient[:, :, 1]) > 0


@pytest.mark.parametrize(
    ("dim", "num_heads"),
    ((0, 1), (-1, 1), (True, 1), (8, 0), (8, -1), (8, True), (10, 3)),
)
def test_cafm_rejects_invalid_head_and_channel_configurations(dim: int, num_heads: int) -> None:
    """Invalid dimensions and non-divisible head configurations should fail clearly."""
    with pytest.raises(ValueError):
        CAFMAttention(dim=dim, num_heads=num_heads)


def test_cafm_rejects_invalid_runtime_tensor_contracts() -> None:
    """CAFM should reject non-NCHW inputs and unexpected channel counts."""
    module = CAFMAttention(dim=8, num_heads=2)
    with pytest.raises(ValueError, match="4D NCHW"):
        module(torch.randn(1, 8, 4))
    with pytest.raises(ValueError, match="initialized for 8 channels"):
        module(torch.randn(1, 7, 4, 4))


def test_c2psa_cafm_preserves_outer_and_ffn_state_key_paths() -> None:
    """Outer C2PSA and FFN keys should load while all attention keys remain intentionally architecture-specific."""
    torch.manual_seed(13)
    baseline = C2PSA(c1=512, c2=512, n=1, e=0.5)
    variant = C2PSA_CAFM(c1=512, c2=512, n=1, e=0.5)
    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    reusable_keys = {key for key in baseline_state if key.startswith(("cv1.", "cv2.", "m.0.ffn."))}

    assert reusable_keys
    assert all(key in variant_state for key in reusable_keys)
    assert all(baseline_state[key].shape == variant_state[key].shape for key in reusable_keys)
    load_result = variant.load_state_dict(baseline_state, strict=False)
    assert load_result.missing_keys
    assert load_result.unexpected_keys
    assert all(key.startswith("m.0.attn.") for key in load_result.missing_keys)
    assert all(key.startswith("m.0.attn.") for key in load_result.unexpected_keys)
    for key in reusable_keys:
        torch.testing.assert_close(variant.state_dict()[key], baseline_state[key])

    assert len(variant.m) == 1
    assert all(isinstance(block, PSABlockCAFM) for block in variant.m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cafm_cuda_fp32_autocast_fp16_and_strict_deterministic_backward() -> None:
    """CAFM should support CUDA FP32, FP16 autocast, and strict deterministic backward execution."""
    device = torch.device("cuda:0")
    module = CAFMAttention(dim=128, num_heads=2, bias=False).to(device)
    x = torch.randn(1, 128, 4, 5, device=device, requires_grad=True)
    output = module(x)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    module.zero_grad(set_to_none=True)
    x = torch.randn(1, 128, 4, 5, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module(x)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        module = CAFMAttention(dim=64, num_heads=1, bias=False).to(device)
        source = torch.randn(1, 64, 3, 4, device=device)

        def run_backward() -> tuple[torch.Tensor, torch.Tensor]:
            module.zero_grad(set_to_none=True)
            sample = source.detach().clone().requires_grad_(True)
            result = module(sample)
            result.square().mean().backward()
            assert sample.grad is not None
            return result.detach(), sample.grad.detach()

        output_1, gradient_1 = run_backward()
        output_2, gradient_2 = run_backward()
        torch.testing.assert_close(output_1, output_2, rtol=0, atol=0)
        torch.testing.assert_close(gradient_1, gradient_2, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
