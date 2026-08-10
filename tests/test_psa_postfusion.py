# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import C2PSAFullFeatureFusion, C3k2, C3k2FusionBottleneckPSA
from ultralytics.nn.modules.block import PSABlock
from ultralytics.nn.tasks import DetectionModel


MODEL_DIR = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "11"
SCALE_CONTRACTS = {
    "n": (64, 16, 1, 128, 32, 1, 339_248),
    "s": (128, 32, 1, 256, 64, 1, 1_348_704),
    "m": (256, 64, 1, 512, 128, 2, 1_243_584),
    "l": (256, 64, 1, 512, 128, 2, 2_502_848),
    "x": (384, 96, 1, 768, 192, 3, 5_618_976),
}


@pytest.mark.parametrize(
    ("ratio", "match"),
    ((0.0, "bottleneck_ratio must be"), (-0.5, "bottleneck_ratio must be"), (1.01, "bottleneck_ratio must be")),
)
def test_fusion_bottleneck_rejects_out_of_range_ratio(ratio: float, match: str) -> None:
    """Only positive bottleneck ratios no larger than one are valid."""
    with pytest.raises(ValueError, match=match):
        C3k2FusionBottleneckPSA(16, 8, bottleneck_ratio=ratio)


def test_fusion_bottleneck_rejects_zero_rounded_attention_width() -> None:
    """A small valid bottleneck ratio must still leave at least one attention channel."""
    with pytest.raises(ValueError, match="produces zero attention channels"):
        C3k2FusionBottleneckPSA(16, 8, bottleneck_ratio=0.01)


def test_fusion_bottleneck_rejects_nonpositive_attention_ratio() -> None:
    """Attention must retain a positive key-to-head dimension ratio."""
    with pytest.raises(ValueError, match="attn_ratio must be positive"):
        C3k2FusionBottleneckPSA(64, 64, attn_ratio=0.0)


def test_fusion_bottleneck_rejects_nondivisible_adaptive_heads() -> None:
    """An arbitrary width that cannot be evenly divided by its adaptive head count must fail clearly."""
    with pytest.raises(ValueError, match="must be divisible"):
        C3k2FusionBottleneckPSA(200, 200)


def test_fusion_bottleneck_rejects_zero_key_dimension() -> None:
    """A positive attention width must still leave at least one key channel per head."""
    with pytest.raises(ValueError, match="produce a zero key dimension"):
        C3k2FusionBottleneckPSA(8, 4, bottleneck_ratio=0.25)


@pytest.mark.parametrize(
    ("channels", "expected_heads"), ((16, 1), (32, 1), (64, 1), (96, 1), (128, 2), (192, 3))
)
def test_adaptive_head_schedule(channels: int, expected_heads: int) -> None:
    """Quarter-width scale contracts should retain roughly 64 channels per head with a one-head minimum."""
    module = C3k2FusionBottleneckPSA(channels, channels)

    assert module.attention_channels == channels
    assert module.num_heads == expected_heads
    assert isinstance(module.psa, PSABlock)
    assert module.psa.attn.num_heads == expected_heads
    assert module.psa.add
    assert not hasattr(module, "gamma")


@pytest.mark.parametrize("c3k", (False, True))
def test_quarter_width_changes_only_inherited_cv2_state(c3k: bool) -> None:
    """Calling the baseline constructor first should preserve cv1 and internal blocks exactly under a matched seed."""
    torch.manual_seed(19)
    baseline = C3k2(128, 128, n=2, c3k=c3k)
    torch.manual_seed(19)
    ablation = C3k2FusionBottleneckPSA(128, 128, n=2, c3k=c3k, bottleneck_ratio=0.25)
    baseline_state = baseline.state_dict()
    ablation_state = ablation.state_dict()

    for key, value in baseline_state.items():
        assert key in ablation_state
        if key.startswith("cv2."):
            continue
        assert value.shape == ablation_state[key].shape
        torch.testing.assert_close(ablation_state[key], value, rtol=0, atol=0)
    assert baseline_state["cv2.conv.weight"].shape != ablation_state["cv2.conv.weight"].shape
    assert baseline_state["cv2.bn.weight"].shape != ablation_state["cv2.bn.weight"].shape
    assert ablation.cv2.conv.out_channels == 32
    assert ablation.cv3.conv.in_channels == 32 and ablation.cv3.conv.out_channels == 128
    extra_keys = set(ablation_state) - set(baseline_state)
    assert extra_keys and all(key.startswith(("psa.", "cv3.")) for key in extra_keys)


def _postfusion_equation(
    module: C3k2FusionBottleneckPSA, x: torch.Tensor, use_split: bool
) -> torch.Tensor:
    """Evaluate the intended post-fusion equation independently of the module forward method."""
    if use_split:
        split = module.cv1(x).split((module.c, module.c), 1)
        branches = [split[0], split[1]]
    else:
        branches = list(module.cv1(x).chunk(2, 1))
    branches.extend(block(branches[-1]) for block in module.m)
    fused = module.cv2(torch.cat(branches, 1))
    return module.cv3(module.psa(fused))


@pytest.mark.parametrize("bottleneck_ratio", (1.0, 0.25))
@pytest.mark.parametrize("forward_name", ("forward", "forward_split"))
def test_forward_matches_postfusion_attention_equation(
    bottleneck_ratio: float, forward_name: str
) -> None:
    """Both forwards should attend after internal concatenation and cv2 fusion, without an outer skip or concat."""
    torch.manual_seed(23)
    module = C3k2FusionBottleneckPSA(
        64, 64, n=2, bottleneck_ratio=bottleneck_ratio
    ).eval()
    x = torch.randn(2, 64, 7, 11)

    with torch.no_grad():
        expected = _postfusion_equation(module, x, use_split=forward_name == "forward_split")
        actual = getattr(module, forward_name)(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == x.shape and torch.isfinite(actual).all()


def test_zero_output_projection_does_not_expose_outer_identity() -> None:
    """Zeroing cv3 should produce zero rather than reveal an unregistered outer input or fused-feature skip."""
    module = C3k2FusionBottleneckPSA(64, 64, bottleneck_ratio=0.25).eval()
    with torch.no_grad():
        module.cv3.conv.weight.zero_()
    x = torch.randn(2, 64, 7, 11)

    with torch.no_grad():
        output = module(x)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert not torch.equal(output, x)


def test_postfusion_psa_rectangular_forward_backward_is_finite() -> None:
    """Global PSA should preserve a rectangular tensor and propagate finite, nonzero gradients."""
    torch.manual_seed(29)
    module = C3k2FusionBottleneckPSA(64, 64, n=2, bottleneck_ratio=0.25)
    x = torch.randn(2, 64, 7, 11, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape and torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


@pytest.mark.parametrize(("scale", "contract"), SCALE_CONTRACTS.items())
def test_scale_generic_model_contracts_and_parameter_delta(scale: str, contract: tuple[int, ...]) -> None:
    """Every scale should retain routing while applying quarter-width post-fusion PSA at P3 and P4."""
    p3_output, p3_attention, p3_heads, p4_output, p4_attention, p4_heads, expected_delta = contract
    ablation_path = MODEL_DIR / f"yolo11{scale}-psa-postfusion-p3p4quarter-p5full.yaml"
    baseline_path = MODEL_DIR / f"yolo11{scale}.yaml"
    ablation = DetectionModel(str(ablation_path), verbose=False)
    baseline = DetectionModel(str(baseline_path), verbose=False)
    p5, p3, p4 = ablation.model[10], ablation.model[16], ablation.model[19]

    assert isinstance(p5, C2PSAFullFeatureFusion)
    assert p5.fusion == "no_concat"
    assert p5.cv1.conv.in_channels == p5.cv1.conv.out_channels == p5.c
    assert p5.cv2.conv.in_channels == p5.cv2.conv.out_channels == p5.c
    assert isinstance(p3, C3k2FusionBottleneckPSA)
    assert isinstance(p4, C3k2FusionBottleneckPSA)
    assert type(ablation.model[22]) is C3k2
    assert ablation.model[23].f == [16, 19, 22]
    assert p3.bottleneck_ratio == p4.bottleneck_ratio == 0.25
    assert p3.attention_channels == p3_attention and p4.attention_channels == p4_attention
    assert p3.num_heads == p3_heads and p4.num_heads == p4_heads
    assert p3.cv2.conv.out_channels == p3_attention and p4.cv2.conv.out_channels == p4_attention
    assert p3.cv3.conv.in_channels == p3_attention and p3.cv3.conv.out_channels == p3_output
    assert p4.cv3.conv.in_channels == p4_attention and p4.cv3.conv.out_channels == p4_output
    expected_c3k = scale in "mlx"
    assert (type(p3.m[0]).__name__ == "C3k") is expected_c3k
    assert (type(p4.m[0]).__name__ == "C3k") is expected_c3k
    ablation_parameters = sum(parameter.numel() for parameter in ablation.parameters())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert ablation_parameters - baseline_parameters == expected_delta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_postfusion_psa_cuda_fp32_and_fp16_autocast() -> None:
    """The complete projected module should preserve finite CUDA gradients in FP32 and FP16 autocast."""
    device = torch.device("cuda:0")
    module = C3k2FusionBottleneckPSA(64, 64, bottleneck_ratio=0.25).to(device)

    fp32_input = torch.randn(1, 64, 7, 11, device=device, requires_grad=True)
    fp32_output = module(fp32_input)
    fp32_output.square().mean().backward()
    assert torch.isfinite(fp32_output).all()
    assert fp32_input.grad is not None and torch.isfinite(fp32_input.grad).all()

    module.zero_grad(set_to_none=True)
    fp16_input = torch.randn(1, 64, 7, 11, device=device, dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fp16_output = module(fp16_input)
    fp16_output.float().square().mean().backward()
    assert torch.isfinite(fp16_output).all()
    assert fp16_input.grad is not None and torch.isfinite(fp16_input.grad).all()
