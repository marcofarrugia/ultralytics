# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules import C2PSA, C2PSAFullFeatureFusion
from ultralytics.nn.modules.block import PSABlock


@pytest.mark.parametrize(("fusion", "cv2_width"), (("concat", 2), ("no_concat", 1)))
@pytest.mark.parametrize(("channels", "repeats"), ((64, 1), (128, 2)))
def test_full_feature_fusion_constructor_contracts(
    fusion: str, cv2_width: int, channels: int, repeats: int
) -> None:
    """Both modes should retain the complete feature width and configure only the final fusion projection differently."""
    module = C2PSAFullFeatureFusion(channels, channels, n=repeats, fusion=fusion)

    assert module.c == channels
    assert module.fusion == fusion
    assert module.cv1.conv.in_channels == channels
    assert module.cv1.conv.out_channels == channels
    assert module.cv2.conv.in_channels == cv2_width * channels
    assert module.cv2.conv.out_channels == channels
    assert len(module.m) == repeats
    assert all(isinstance(block, PSABlock) for block in module.m)
    assert all(block.attn.num_heads == channels // 64 for block in module.m)
    assert all(block.add for block in module.m)
    assert not hasattr(module, "gamma")


def test_full_feature_fusion_defaults_to_concat() -> None:
    """Direct construction without a mode should use the documented concatenative fusion default."""
    module = C2PSAFullFeatureFusion(64, 64)

    assert module.fusion == "concat"
    assert module.cv2.conv.in_channels == 128


def test_full_feature_fusion_rejects_unsupported_mode() -> None:
    """An unsupported YAML fusion value should fail instead of silently selecting another tensor path."""
    with pytest.raises(ValueError, match="expected 'concat' or 'no_concat'"):
        C2PSAFullFeatureFusion(64, 64, fusion="add")


def test_full_feature_fusion_rejects_channel_change() -> None:
    """The full-feature block should preserve its input channel width."""
    with pytest.raises(AssertionError):
        C2PSAFullFeatureFusion(64, 128)


@pytest.mark.parametrize(("fusion", "cv2_channels"), (("concat", 128), ("no_concat", 64)))
def test_full_feature_fusion_routes_complete_width(fusion: str, cv2_channels: int) -> None:
    """Hooks should observe all C channels through cv1 and PSA and the mode-specific width at cv2."""
    channels = 64
    module = C2PSAFullFeatureFusion(channels, channels, n=1, fusion=fusion).eval()
    observed: dict[str, tuple[int, ...]] = {}

    def capture_input(name: str):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            observed[name] = tuple(args[0].shape)

        return hook

    def capture_cv1_output(
        _module: torch.nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        observed["cv1_output"] = tuple(output.shape)

    hooks = (
        module.cv1.register_forward_pre_hook(capture_input("cv1_input")),
        module.cv1.register_forward_hook(capture_cv1_output),
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
        "cv2_input": (2, cv2_channels, 4, 5),
    }
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("fusion", ("concat", "no_concat"))
def test_forward_matches_registered_fusion_equation(fusion: str) -> None:
    """Each mode should exactly implement its registered full-feature fusion equation."""
    torch.manual_seed(7)
    module = C2PSAFullFeatureFusion(64, 64, n=1, fusion=fusion).eval()
    x = torch.randn(2, 64, 4, 5)

    with torch.no_grad():
        z = module.cv1(x)
        attended = module.m(z)
        cv2_input = torch.cat((z, attended), dim=1) if fusion == "concat" else attended
        expected = module.cv2(cv2_input)
        actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fusion", ("concat", "no_concat"))
def test_zero_fusion_projection_does_not_expose_outer_identity(fusion: str) -> None:
    """Zeroing cv2 should produce zero rather than reveal an unregistered outer x skip."""
    module = C2PSAFullFeatureFusion(64, 64, n=1, fusion=fusion).eval()
    with torch.no_grad():
        module.cv2.conv.weight.zero_()
    x = torch.randn(2, 64, 4, 5)

    with torch.no_grad():
        output = module(x)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert not torch.equal(output, x)


@pytest.mark.parametrize("fusion", ("concat", "no_concat"))
def test_full_feature_fusion_forward_backward_is_finite(fusion: str) -> None:
    """Both full-feature tensor paths should propagate finite, nonzero CPU gradients."""
    torch.manual_seed(11)
    module = C2PSAFullFeatureFusion(64, 64, n=1, fusion=fusion)
    x = torch.randn(2, 64, 4, 5, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape and torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    for component in (module.cv1, module.m, module.cv2):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_pretrained_projection_shape_compatibility_is_mode_specific() -> None:
    """Only no-concat should retain both baseline projection shapes; concat should retain cv1 alone."""
    baseline = C2PSA(128, 128, n=1)
    concat = C2PSAFullFeatureFusion(128, 128, n=1, fusion="concat")
    no_concat = C2PSAFullFeatureFusion(128, 128, n=1, fusion="no_concat")

    assert concat.cv1.conv.weight.shape == baseline.cv1.conv.weight.shape
    assert no_concat.cv1.conv.weight.shape == baseline.cv1.conv.weight.shape
    assert concat.cv2.conv.weight.shape != baseline.cv2.conv.weight.shape
    assert no_concat.cv2.conv.weight.shape == baseline.cv2.conv.weight.shape


def test_baseline_c2psa_contract_remains_split_and_unscaled() -> None:
    """The baseline class should retain its half-width split path without a fusion mode or gamma."""
    baseline = C2PSA(128, 128, n=2, e=0.5)

    assert baseline.c == 64
    assert baseline.cv1.conv.in_channels == 128
    assert baseline.cv1.conv.out_channels == 128
    assert baseline.cv2.conv.in_channels == 128
    assert baseline.cv2.conv.out_channels == 128
    assert len(baseline.m) == 2
    assert not hasattr(baseline, "fusion")
    assert not hasattr(baseline, "gamma")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("fusion", ("concat", "no_concat"))
def test_full_feature_fusion_cuda_fp32_and_fp16_autocast(fusion: str) -> None:
    """Both modes should support finite CUDA FP32 and FP16-autocast forward/backward execution."""
    device = torch.device("cuda:0")
    module = C2PSAFullFeatureFusion(128, 128, n=1, fusion=fusion).to(device)

    fp32_input = torch.randn(1, 128, 4, 5, device=device, requires_grad=True)
    fp32_output = module(fp32_input)
    fp32_output.square().mean().backward()
    assert torch.isfinite(fp32_output).all()
    assert fp32_input.grad is not None and torch.isfinite(fp32_input.grad).all()

    module.zero_grad(set_to_none=True)
    fp16_input = torch.randn(1, 128, 4, 5, device=device, dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fp16_output = module(fp16_input)
    fp16_output.float().square().mean().backward()
    assert torch.isfinite(fp16_output).all()
    assert fp16_input.grad is not None and torch.isfinite(fp16_input.grad).all()
