# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import C2PSA, C3k2, C3k2CrossPSA, CrossPSA
from ultralytics.nn.tasks import DetectionModel


MODEL_DIR = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "11"
SCALE_CONTRACTS = {
    "n": (32, 64, 2, 2, 38_048),
    "s": (64, 128, 2, 2, 147_776),
    "m": (128, 256, 2, 4, 582_272),
    "l": (128, 256, 2, 4, 582_272),
    "x": (192, 384, 4, 6, 1_303_488),
}


@pytest.mark.parametrize(
    ("channels", "expected_heads", "head_dim", "key_dim"),
    ((32, 2, 16, 8), (64, 2, 32, 16), (128, 2, 64, 32), (192, 4, 48, 24), (256, 4, 64, 32), (384, 6, 64, 32)),
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


@pytest.mark.parametrize("c3k", (False, True))
def test_c3k2_cross_psa_preserves_original_state_keys_and_initial_values(c3k: bool) -> None:
    """Calling the original constructor first should leave every pretrained-compatible tensor unchanged."""
    torch.manual_seed(17)
    baseline = C3k2(128, 128, n=2, c3k=c3k)
    torch.manual_seed(17)
    ablation = C3k2CrossPSA(128, 128, n=2, c3k=c3k)
    baseline_state = baseline.state_dict()
    ablation_state = ablation.state_dict()

    assert all(key in ablation_state for key in baseline_state)
    for key, value in baseline_state.items():
        assert value.shape == ablation_state[key].shape
        torch.testing.assert_close(ablation_state[key], value, rtol=0, atol=0)
    extra_keys = set(ablation_state) - set(baseline_state)
    assert extra_keys and all(key.startswith("cross_psa.") for key in extra_keys)
    assert sum(isinstance(module, CrossPSA) for module in ablation.modules()) == 1


def test_c3k2_cross_psa_forward_matches_final_branch_refinement_equation() -> None:
    """CrossPSA should replace only the final processed branch before the original concatenation and cv2 fusion."""
    torch.manual_seed(23)
    module = C3k2CrossPSA(64, 64, n=2).eval()
    x = torch.randn(2, 64, 7, 11)

    with torch.no_grad():
        y = list(module.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in module.m)
        y[-1] = module.cross_psa(y[-1])
        expected = module.cv2(torch.cat(y, 1))
        actual = module(x)
        split_actual = module.forward_split(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(split_actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("scale", "contract"), SCALE_CONTRACTS.items())
def test_scale_generic_model_contracts_and_parameter_delta(scale: str, contract: tuple[int, ...]) -> None:
    """Every scale should preserve routing while adding exactly one correctly sized CrossPSA at P3 and P4."""
    p3_channels, p4_channels, p3_heads, p4_heads, expected_delta = contract
    ablation_path = MODEL_DIR / f"yolo11{scale}-crosspsa-neck-p3-p4.yaml"
    baseline_path = MODEL_DIR / f"yolo11{scale}.yaml"
    ablation = DetectionModel(str(ablation_path), verbose=False)
    baseline = DetectionModel(str(baseline_path), verbose=False)
    p3, p4 = ablation.model[16], ablation.model[19]

    assert type(ablation.model[10]) is C2PSA
    assert isinstance(p3, C3k2CrossPSA) and isinstance(p4, C3k2CrossPSA)
    assert type(ablation.model[22]) is C3k2
    assert ablation.model[23].f == [16, 19, 22]
    assert p3.c == p3_channels and p4.c == p4_channels
    assert p3.cross_psa.attn.num_heads == p3_heads
    assert p4.cross_psa.attn.num_heads == p4_heads
    assert sum(isinstance(module, CrossPSA) for module in p3.modules()) == 1
    assert sum(isinstance(module, CrossPSA) for module in p4.modules()) == 1
    ablation_parameters = sum(parameter.numel() for parameter in ablation.parameters())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert ablation_parameters - baseline_parameters == expected_delta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cross_psa_cuda_fp32_and_fp16_autocast() -> None:
    """CrossPSA should preserve finite CUDA gradients in FP32 and FP16 autocast."""
    device = torch.device("cuda:0")
    module = CrossPSA(64).to(device)
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
