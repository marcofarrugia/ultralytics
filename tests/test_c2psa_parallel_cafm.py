# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2PSA, C2PSA_ParallelCAFM, CAFMAttention
from ultralytics.nn.modules.block import PSABlock, PSABlockParallelCAFM


def _conv3d(x: torch.Tensor, layer: nn.Conv3d) -> torch.Tensor:
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


def _load_baseline_block_state(
    baseline: PSABlock, variant: PSABlockParallelCAFM
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load the baseline attention and FFN paths into a parallel block."""
    result = variant.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {
        "gamma",
        *(f"cafm.{key}" for key in variant.cafm.state_dict()),
    }
    return result


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


@pytest.mark.parametrize("shortcut", (True, False))
def test_parallel_block_is_baseline_equivalent_at_zero_gate(shortcut: bool) -> None:
    """Copied baseline paths should produce exactly the baseline output while the CAFM gate is zero."""
    torch.manual_seed(13)
    baseline = PSABlock(c=64, attn_ratio=0.5, num_heads=1, shortcut=shortcut).eval()
    variant = PSABlockParallelCAFM(c=64, attn_ratio=0.5, num_heads=1, shortcut=shortcut).eval()
    _load_baseline_block_state(baseline, variant)
    assert variant.gamma.ndim == 0
    assert variant.gamma.requires_grad
    assert variant.gamma.item() == 0.0

    x = torch.randn(2, 64, 4, 5)
    with torch.no_grad():
        expected = baseline(x)
        actual = variant(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_parallel_attention_paths_receive_the_same_representation() -> None:
    """Attention and CAFM should receive the same pre-residual tensor object."""
    block = PSABlockParallelCAFM(c=64, num_heads=1).eval()
    observed: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            observed[name] = args[0]

        return hook

    attention_hook = block.attn.register_forward_pre_hook(capture("attn"))
    cafm_hook = block.cafm.register_forward_pre_hook(capture("cafm"))
    x = torch.randn(1, 64, 3, 4)
    try:
        block(x)
    finally:
        attention_hook.remove()
        cafm_hook.remove()

    assert observed["attn"] is x
    assert observed["cafm"] is x


def test_zero_gate_learns_before_cafm_parameters_then_activates_cafm_gradients() -> None:
    """The first backward should update gamma while holding CAFM weights fixed, then activate CAFM gradients."""
    torch.manual_seed(17)
    block = PSABlockParallelCAFM(c=64, num_heads=1, shortcut=False)
    block.ffn = nn.Identity()
    optimizer = torch.optim.SGD(block.parameters(), lr=1e-3)
    x = torch.randn(1, 64, 3, 4)
    with torch.no_grad():
        target = block.cafm(x).detach()

    loss = (block(x) * target).sum()
    loss.backward()
    assert block.gamma.grad is not None and torch.isfinite(block.gamma.grad)
    assert block.gamma.grad.abs().item() > 0
    assert all(parameter.grad is not None for parameter in block.cafm.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in block.cafm.parameters())

    optimizer.step()
    assert block.gamma.item() != 0.0

    optimizer.zero_grad(set_to_none=True)
    second_loss = (block(x) * target).sum()
    second_loss.backward()
    cafm_gradients = [parameter.grad for parameter in block.cafm.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in cafm_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in cafm_gradients)


def test_parallel_c2psa_preserves_all_baseline_state_paths_and_zero_gate_output() -> None:
    """Outer, Attention, and FFN paths should load without remapping and reproduce the baseline at initialization."""
    torch.manual_seed(19)
    baseline = C2PSA(c1=128, c2=128, n=2, e=0.5).eval()
    variant = C2PSA_ParallelCAFM(c1=128, c2=128, n=2, e=0.5).eval()
    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()

    assert all(key in variant_state for key in baseline_state)
    assert all(baseline_state[key].shape == variant_state[key].shape for key in baseline_state)
    load_result = variant.load_state_dict(baseline_state, strict=False)
    assert not load_result.unexpected_keys
    assert all(key.endswith(".gamma") or ".cafm." in key for key in load_result.missing_keys)
    assert {key for key in load_result.missing_keys if key.endswith(".gamma")} == {
        "m.0.gamma",
        "m.1.gamma",
    }
    assert len(variant.m) == 2
    assert all(isinstance(block, PSABlockParallelCAFM) for block in variant.m)
    assert all(block.gamma.item() == 0.0 for block in variant.m)

    x = torch.randn(1, 128, 4, 5)
    with torch.no_grad():
        expected = baseline(x)
        actual = variant(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_parallel_cafm_cuda_fp32_autocast_fp16_and_strict_deterministic_backward() -> None:
    """The parallel block should support CUDA FP32, FP16 autocast, and strict deterministic backward execution."""
    device = torch.device("cuda:0")
    block = PSABlockParallelCAFM(c=128, num_heads=2).to(device)
    x = torch.randn(1, 128, 4, 5, device=device, requires_grad=True)
    output = block(x)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    block.zero_grad(set_to_none=True)
    x = torch.randn(1, 128, 4, 5, device=device, dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = block(x)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    assert block.gamma.dtype == torch.float32 and block.gamma.ndim == 0
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        block = PSABlockParallelCAFM(c=64, num_heads=1).to(device)
        source = torch.randn(1, 64, 3, 4, device=device)

        def run_backward() -> tuple[torch.Tensor, torch.Tensor]:
            block.zero_grad(set_to_none=True)
            sample = source.detach().clone().requires_grad_(True)
            result = block(sample)
            result.square().mean().backward()
            assert sample.grad is not None
            return result.detach(), sample.grad.detach()

        output_1, gradient_1 = run_backward()
        output_2, gradient_2 = run_backward()
        torch.testing.assert_close(output_1, output_2, rtol=0, atol=0)
        torch.testing.assert_close(gradient_1, gradient_2, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
