# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import C2PSA, C3k2, C3k2FusionBottleneckCrossPSA, CrossPSA
from ultralytics.nn.tasks import DetectionModel


MODEL_DIR = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "11"
SCALE_CONTRACTS = {
    "n": (64, 128, 2, 2, 60_544),
    "s": (128, 256, 2, 2, 235_776),
    "m": (256, 512, 4, 4, 930_304),
    "l": (256, 512, 4, 4, 864_768),
    "x": (384, 768, 6, 6, 1_936_128),
}


@pytest.mark.parametrize(
    ("channels", "expected_heads", "head_dim", "key_dim"),
    ((32, 2, 16, 8), (64, 2, 32, 16), (128, 2, 64, 32), (256, 4, 64, 32), (384, 6, 64, 32)),
)
def test_cross_psa_default_head_schedule(
    channels: int, expected_heads: int, head_dim: int, key_dim: int
) -> None:
    """The scale-generic default should retain roughly 64 value channels per even directional head group."""
    module = CrossPSA(channels)

    assert module.attn.num_heads == expected_heads
    assert module.attn.head_dim == head_dim
    assert module.attn.key_dim == key_dim
    assert module.attn.stripe_width == 5
    assert module.add
    assert not any("gamma" in name for name, _ in module.named_parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"c": 64, "num_heads": 3}, "positive even"),
        ({"c": 64, "num_heads": 6}, "divisible"),
        ({"c": 64, "stripe_width": 0}, "stripe_width must be positive"),
        ({"c": 64, "attn_ratio": 0.0}, "attn_ratio must be positive"),
        ({"c": 63}, "no even head divisor"),
    ),
)
def test_cross_psa_rejects_invalid_configuration(kwargs: dict, match: str) -> None:
    """Invalid stripe and head configurations should fail instead of selecting another attention path."""
    with pytest.raises(ValueError, match=match):
        CrossPSA(**kwargs)


@pytest.mark.parametrize("horizontal", (True, False))
def test_padded_keys_do_not_dilute_stripe_attention(horizontal: bool) -> None:
    """Masked padding should not contribute zero-valued keys or values to a partial stripe."""
    module = CrossPSA(8, num_heads=2, stripe_width=5).eval()
    attention = module.attn
    q = torch.zeros(1, 1, attention.key_dim, 4, 3)
    k = torch.zeros_like(q)
    v = torch.ones(1, 1, attention.head_dim, 4, 3)

    with torch.no_grad():
        output = attention._stripe_attention(q, k, v, horizontal)

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=1e-6)


def test_horizontal_and_vertical_stripes_partition_different_axes() -> None:
    """Uniform logits should average values only within the selected horizontal or vertical stripe."""
    module = CrossPSA(8, num_heads=2, stripe_width=3).eval()
    attention = module.attn
    q = torch.zeros(1, 1, attention.key_dim, 6, 4)
    k = torch.zeros_like(q)
    rows = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6, 1)
    row_values = rows.expand(1, 1, attention.head_dim, 6, 4)
    columns = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 1, 4)
    column_values = columns.expand(1, 1, attention.head_dim, 6, 4)

    with torch.no_grad():
        horizontal = attention._stripe_attention(q, k, row_values, True)
        vertical = attention._stripe_attention(q, k, column_values, False)

    expected_rows = torch.tensor((1.0, 1.0, 1.0, 4.0, 4.0, 4.0)).view(1, 1, 1, 6, 1)
    expected_columns = torch.tensor((1.0, 1.0, 1.0, 3.0)).view(1, 1, 1, 1, 4)
    torch.testing.assert_close(horizontal, expected_rows.expand_as(horizontal), rtol=0, atol=1e-6)
    torch.testing.assert_close(vertical, expected_columns.expand_as(vertical), rtol=0, atol=1e-6)


def test_cross_psa_rectangular_forward_backward_is_finite() -> None:
    """Both padding directions should preserve shape and finite gradients on a non-divisible rectangle."""
    torch.manual_seed(11)
    module = CrossPSA(32)
    x = torch.randn(2, 32, 7, 11, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape and torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("ratio", "match"),
    ((0.0, "bottleneck_ratio must be"), (-0.5, "bottleneck_ratio must be"), (1.01, "bottleneck_ratio must be")),
)
def test_fusion_bottleneck_rejects_out_of_range_ratio(ratio: float, match: str) -> None:
    """Only positive ratios no larger than one are valid."""
    with pytest.raises(ValueError, match=match):
        C3k2FusionBottleneckCrossPSA(16, 8, bottleneck_ratio=ratio)


def test_fusion_bottleneck_rejects_zero_rounded_attention_width() -> None:
    """A small valid ratio must still leave at least one attention channel."""
    with pytest.raises(ValueError, match="produces zero attention channels"):
        C3k2FusionBottleneckCrossPSA(16, 8, bottleneck_ratio=0.01)


@pytest.mark.parametrize("c3k", (False, True))
def test_full_width_preserves_all_inherited_c3k2_state(c3k: bool) -> None:
    """At ratio one, calling the baseline constructor first should preserve every inherited state tensor exactly."""
    torch.manual_seed(17)
    baseline = C3k2(128, 128, n=2, c3k=c3k)
    torch.manual_seed(17)
    ablation = C3k2FusionBottleneckCrossPSA(128, 128, n=2, c3k=c3k, bottleneck_ratio=1.0)
    baseline_state = baseline.state_dict()
    ablation_state = ablation.state_dict()

    assert all(key in ablation_state for key in baseline_state)
    for key, value in baseline_state.items():
        assert value.shape == ablation_state[key].shape
        torch.testing.assert_close(ablation_state[key], value, rtol=0, atol=0)
    extra_keys = set(ablation_state) - set(baseline_state)
    assert extra_keys and all(key.startswith(("cross_psa.", "cv3.")) for key in extra_keys)
    assert ablation.cv2.conv.out_channels == 128


@pytest.mark.parametrize("c3k", (False, True))
def test_half_width_changes_only_inherited_cv2_state(c3k: bool) -> None:
    """At ratio one-half, cv1 and the C3k2 blocks should remain exact while cv2 changes shape intentionally."""
    torch.manual_seed(19)
    baseline = C3k2(128, 128, n=2, c3k=c3k)
    torch.manual_seed(19)
    ablation = C3k2FusionBottleneckCrossPSA(128, 128, n=2, c3k=c3k, bottleneck_ratio=0.5)
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
    assert ablation.cv2.conv.out_channels == 64
    assert ablation.cv3.conv.in_channels == 64 and ablation.cv3.conv.out_channels == 128
    extra_keys = set(ablation_state) - set(baseline_state)
    assert extra_keys and all(key.startswith(("cross_psa.", "cv3.")) for key in extra_keys)


def _postfusion_equation(
    module: C3k2FusionBottleneckCrossPSA, x: torch.Tensor, use_split: bool
) -> torch.Tensor:
    """Evaluate the intended post-fusion equation independently of the module forward method."""
    if use_split:
        split = module.cv1(x).split((module.c, module.c), 1)
        branches = [split[0], split[1]]
    else:
        branches = list(module.cv1(x).chunk(2, 1))
    branches.extend(block(branches[-1]) for block in module.m)
    fused = module.cv2(torch.cat(branches, 1))
    return module.cv3(module.cross_psa(fused))


@pytest.mark.parametrize("bottleneck_ratio", (1.0, 0.5))
@pytest.mark.parametrize("forward_name", ("forward", "forward_split"))
def test_forward_matches_postfusion_attention_equation(bottleneck_ratio: float, forward_name: str) -> None:
    """Both forwards should attend only after internal branch concatenation and cv2 fusion, with no outer skip."""
    torch.manual_seed(23)
    module = C3k2FusionBottleneckCrossPSA(64, 64, n=2, bottleneck_ratio=bottleneck_ratio).eval()
    x = torch.randn(2, 64, 7, 11)

    with torch.no_grad():
        expected = _postfusion_equation(module, x, use_split=forward_name == "forward_split")
        actual = getattr(module, forward_name)(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == x.shape and torch.isfinite(actual).all()


@pytest.mark.parametrize(("scale", "contract"), SCALE_CONTRACTS.items())
def test_scale_generic_model_contracts_and_parameter_delta(scale: str, contract: tuple[int, ...]) -> None:
    """Every scale should preserve routing while using full-width P3 and half-width P4 post-fusion attention."""
    p3_channels, p4_channels, p3_heads, p4_heads, expected_delta = contract
    ablation_path = MODEL_DIR / f"yolo11{scale}-crosspsa-postfusion-p3full-p4half.yaml"
    baseline_path = MODEL_DIR / f"yolo11{scale}.yaml"
    ablation = DetectionModel(str(ablation_path), verbose=False)
    baseline = DetectionModel(str(baseline_path), verbose=False)
    p3, p4 = ablation.model[16], ablation.model[19]

    assert type(ablation.model[10]) is C2PSA
    assert isinstance(p3, C3k2FusionBottleneckCrossPSA)
    assert isinstance(p4, C3k2FusionBottleneckCrossPSA)
    assert type(ablation.model[22]) is C3k2
    assert ablation.model[23].f == [16, 19, 22]
    assert p3.bottleneck_ratio == 1.0 and p4.bottleneck_ratio == 0.5
    assert p3.attention_channels == p3_channels
    assert p4.attention_channels == p4_channels // 2
    assert p3.cv2.conv.out_channels == p3_channels
    assert p4.cv2.conv.out_channels == p4_channels // 2
    assert p3.cv3.conv.in_channels == p3_channels and p3.cv3.conv.out_channels == p3_channels
    assert p4.cv3.conv.in_channels == p4_channels // 2 and p4.cv3.conv.out_channels == p4_channels
    assert p3.cross_psa.attn.num_heads == p3_heads
    assert p4.cross_psa.attn.num_heads == p4_heads
    expected_c3k = scale in "mlx"
    assert (type(p3.m[0]).__name__ == "C3k") is expected_c3k
    assert (type(p4.m[0]).__name__ == "C3k") is expected_c3k
    ablation_parameters = sum(parameter.numel() for parameter in ablation.parameters())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert ablation_parameters - baseline_parameters == expected_delta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cross_psa_cuda_fp32_and_fp16_autocast() -> None:
    """The complete projected module should preserve finite CUDA gradients in FP32 and FP16 autocast."""
    device = torch.device("cuda:0")
    module = C3k2FusionBottleneckCrossPSA(64, 64, bottleneck_ratio=0.5).to(device)
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
