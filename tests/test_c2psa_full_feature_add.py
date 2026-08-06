# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules import C2PSA, C2PSAFullFeatureAdd
from ultralytics.nn.modules.block import PSABlock


@pytest.mark.parametrize(("channels", "repeats"), ((64, 1), (128, 2)))
def test_full_feature_add_constructor_contracts(channels: int, repeats: int) -> None:
    """The block should remain full-width through projection, PSA processing, fusion, and residual scaling."""
    module = C2PSAFullFeatureAdd(channels, channels, n=repeats)

    assert module.c == channels
    assert module.cv1.conv.in_channels == channels
    assert module.cv1.conv.out_channels == channels
    assert module.cv2.conv.in_channels == 2 * channels
    assert module.cv2.conv.out_channels == channels
    assert len(module.m) == repeats
    assert all(isinstance(block, PSABlock) for block in module.m)
    assert all(block.attn.num_heads == channels // 64 for block in module.m)
    assert all(block.add for block in module.m)
    assert module.gamma.shape == (channels,)
    assert module.gamma.requires_grad
    torch.testing.assert_close(module.gamma, torch.full_like(module.gamma, 0.01), rtol=0, atol=0)


def test_full_feature_add_rejects_channel_change() -> None:
    """The outer identity residual requires identical input and output channel counts."""
    with pytest.raises(AssertionError):
        C2PSAFullFeatureAdd(64, 128)


def test_full_feature_add_routes_complete_width_and_preserves_shape() -> None:
    """Hooks should observe C channels through cv1 and PSA, then 2C channels at cv2."""
    channels = 64
    module = C2PSAFullFeatureAdd(channels, channels, n=1).eval()
    observed: dict[str, tuple[int, ...]] = {}

    def capture_input(name: str):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            observed[name] = tuple(args[0].shape)

        return hook

    def capture_output(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed["cv1_output"] = tuple(output.shape)

    hooks = (
        module.cv1.register_forward_pre_hook(capture_input("cv1_input")),
        module.cv1.register_forward_hook(capture_output),
        module.m[0].register_forward_pre_hook(capture_input("psa_input")),
        module.cv2.register_forward_pre_hook(capture_input("cv2_input")),
    )
    x = torch.randn(2, channels, 4, 5)
    try:
        with torch.no_grad():
            output = module(x)
    finally:
        for hook in hooks:
            hook.remove()

    assert observed == {
        "cv1_input": (2, channels, 4, 5),
        "cv1_output": (2, channels, 4, 5),
        "psa_input": (2, channels, 4, 5),
        "cv2_input": (2, 2 * channels, 4, 5),
    }
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_zero_outer_scale_is_exact_identity() -> None:
    """A zero gamma vector should disable the complete fused branch exactly."""
    module = C2PSAFullFeatureAdd(64, 64, n=1).eval()
    with torch.no_grad():
        module.gamma.zero_()
    x = torch.randn(2, 64, 4, 5)

    with torch.no_grad():
        output = module(x)

    torch.testing.assert_close(output, x, rtol=0, atol=0)


def test_forward_matches_project_concat_fuse_scaled_residual_equation() -> None:
    """The implementation should match y = x + gamma * cv2([z; M(z)])."""
    torch.manual_seed(7)
    module = C2PSAFullFeatureAdd(64, 64, n=1).eval()
    x = torch.randn(2, 64, 4, 5)

    with torch.no_grad():
        z = module.cv1(x)
        attended = module.m(z)
        fused = module.cv2(torch.cat((z, attended), dim=1))
        expected = x + module.gamma.view(1, -1, 1, 1) * fused
        actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_scale_backpropagates_finite_nonzero_gradients() -> None:
    """Gamma and the active 0.01-scaled branch should both receive usable gradients."""
    torch.manual_seed(11)
    module = C2PSAFullFeatureAdd(64, 64, n=1)
    x = torch.randn(2, 64, 4, 5, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.gamma.grad is not None and torch.isfinite(module.gamma.grad).all()
    assert torch.count_nonzero(module.gamma.grad) > 0

    branch_gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if not name.startswith("gamma")
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in branch_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in branch_gradients)


def test_baseline_c2psa_contract_remains_split_and_unscaled() -> None:
    """The existing baseline class should retain its half-width split contract and have no outer gamma."""
    baseline = C2PSA(128, 128, n=2, e=0.5)

    assert baseline.c == 64
    assert baseline.cv1.conv.in_channels == 128
    assert baseline.cv1.conv.out_channels == 128
    assert baseline.cv2.conv.in_channels == 128
    assert baseline.cv2.conv.out_channels == 128
    assert len(baseline.m) == 2
    assert not hasattr(baseline, "gamma")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_feature_add_cuda_fp32_and_fp16_autocast() -> None:
    """The block should remain finite with CUDA FP32 and FP16 autocast forward/backward execution."""
    device = torch.device("cuda:0")
    module = C2PSAFullFeatureAdd(128, 128, n=1).to(device)

    fp32_input = torch.randn(1, 128, 4, 5, device=device, requires_grad=True)
    fp32_output = module(fp32_input)
    assert torch.isfinite(fp32_output).all()
    fp32_output.square().mean().backward()
    assert fp32_input.grad is not None and torch.isfinite(fp32_input.grad).all()

    module.zero_grad(set_to_none=True)
    fp16_input = torch.randn(1, 128, 4, 5, device=device, dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fp16_output = module(fp16_input)
    assert torch.isfinite(fp16_output).all()
    fp16_output.float().square().mean().backward()
    assert fp16_input.grad is not None and torch.isfinite(fp16_input.grad).all()
    assert module.gamma.dtype == torch.float32
