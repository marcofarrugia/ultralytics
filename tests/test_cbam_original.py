# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import CBAMDetect, CBAMOriginal, Detect


def test_cbam_original_matches_source_equations():
    """CBAMOriginal should use shared average/max channel attention before max/mean spatial attention."""
    torch.manual_seed(0)
    module = CBAMOriginal(32).eval()
    x = torch.randn(2, 32, 11, 13)
    mlp_inputs = []
    handle = module.channel_mlp.register_forward_hook(lambda _module, inputs, _output: mlp_inputs.append(inputs[0]))

    with torch.no_grad():
        actual = module(x)
        repeated = module(x)
        pool_size = (x.size(2), x.size(3))
        avg_descriptor = F.avg_pool2d(x, pool_size, stride=pool_size)
        max_descriptor = F.max_pool2d(x, pool_size, stride=pool_size)
        channel_logits = module.channel_mlp(avg_descriptor) + module.channel_mlp(max_descriptor)
        channel_refined = x * torch.sigmoid(channel_logits).unsqueeze(-1).unsqueeze(-1)
        spatial_descriptor = torch.cat(
            (
                torch.max(channel_refined, dim=1, keepdim=True)[0],
                torch.mean(channel_refined, dim=1, keepdim=True),
            ),
            dim=1,
        )
        expected = channel_refined * torch.sigmoid(module.spatial(spatial_descriptor))

    handle.remove()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    assert len(mlp_inputs) == 6  # Two calls in each module pass and two calls in the explicit reference equation.
    torch.testing.assert_close(mlp_inputs[0], avg_descriptor)
    torch.testing.assert_close(mlp_inputs[1], max_descriptor)
    assert not any(
        isinstance(child, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)) for child in module.modules()
    )
    assert module.spatial.conv.kernel_size == (7, 7)
    assert module.spatial.conv.bias is None
    assert module.spatial.bn.eps == pytest.approx(1e-5)
    assert module.spatial.bn.momentum == pytest.approx(0.01)


@pytest.mark.parametrize(("channels", "size"), [(64, 80), (128, 40), (256, 20)])
def test_cbam_original_shape_and_gradients(channels, size):
    """CBAMOriginal should preserve NCHW shape and produce finite values and gradients."""
    module = CBAMOriginal(channels).train()
    x = torch.randn(2, channels, size, size, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the deterministic-backward check.")
def test_cbam_original_cuda_backward_is_deterministic():
    """CBAMOriginal should support strict deterministic CUDA backward with full-window pooling."""
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        module = CBAMOriginal(64).cuda().train()
        x = torch.randn(2, 64, 20, 20, device="cuda", requires_grad=True)
        output = module(x)
        output.square().mean().backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    finally:
        torch.use_deterministic_algorithms(deterministic_enabled, warn_only=warn_only_enabled)


@pytest.mark.parametrize(("channels", "ratio"), [(0, 16), (16, 0), (8, 16)])
def test_cbam_original_rejects_invalid_widths(channels, ratio):
    """CBAMOriginal should reject invalid channel and reduction configurations instead of creating an empty MLP."""
    with pytest.raises(ValueError):
        CBAMOriginal(channels, ratio)


def test_cbam_detect_preserves_detect_state_keys():
    """CBAMDetect should retain all baseline Detect parameter keys and add only CBAM state."""
    channels = (64, 128, 256)
    Detect.legacy = False
    CBAMDetect.legacy = False
    baseline = Detect(nc=14, ch=channels)
    target = CBAMDetect(nc=14, ch=channels)
    baseline_state = baseline.state_dict()
    target_state = target.state_dict()

    assert set(baseline_state).issubset(target_state)
    assert all(baseline_state[key].shape == target_state[key].shape for key in baseline_state)
    incompatible = target.load_state_dict(baseline_state, strict=False)
    assert not incompatible.unexpected_keys
    expected_missing = {
        key for key in target_state if key.startswith("cbam.") and not key.endswith("num_batches_tracked")
    }
    assert set(incompatible.missing_keys) == expected_missing


def test_cbam_detect_training_contract():
    """CBAMDetect should preserve the three-scale raw Detect output contract."""
    channels = (64, 128, 256)
    module = CBAMDetect(nc=14, ch=channels).train()
    features = [
        torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    outputs = module(features)

    assert [output.shape for output in outputs] == [(2, 78, 80, 80), (2, 78, 40, 40), (2, 78, 20, 20)]
    assert all(torch.isfinite(output).all() for output in outputs)
