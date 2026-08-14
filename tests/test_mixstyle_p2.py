# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import random
from copy import deepcopy

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, C3k2MixStyle, MixStyle
from ultralytics.utils import YAML


def _pinned_reference(x: torch.Tensor, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
    """Compact random-mixing reference from KaiyangZhou/mixstyle-release@16f7cf1."""
    if random.random() > p:
        return x
    batch_size = x.shape[0]
    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True)
    sig = (var + eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    normalized = (x - mu) / sig
    mixing_weight = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device, x.dtype)
    permutation = torch.randperm(batch_size, device=x.device)
    mixed_mu = mu * mixing_weight + mu[permutation] * (1 - mixing_weight)
    mixed_sig = sig * mixing_weight + sig[permutation] * (1 - mixing_weight)
    return normalized * mixed_sig + mixed_mu


def test_mixstyle_matches_pinned_reference_outputs_and_gradients():
    """Preserve the official arithmetic, detached statistics, random order, and gradient flow."""
    source = torch.randn(4, 7, 9, 11)
    actual_input = source.clone().requires_grad_(True)
    reference_input = source.clone().requires_grad_(True)
    transform = MixStyle(p=1.0, alpha=0.1, eps=1e-6).train()

    random.seed(17)
    torch.manual_seed(29)
    actual = transform(actual_input)
    random.seed(17)
    torch.manual_seed(29)
    expected = _pinned_reference(reference_input, p=1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    weights = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.detach(), source, rtol=0, atol=0)


def test_mixstyle_identity_paths(monkeypatch):
    """Return the original tensor object in evaluation, rejection, deactivation, and batch-one paths."""
    x = torch.randn(3, 8, 7, 9)
    transform = MixStyle(p=1.0)
    transform.eval()
    assert transform(x) is x

    transform.train()
    transform.p = 0.0
    monkeypatch.setattr("ultralytics.nn.modules.block.random.random", lambda: 0.5)
    assert transform(x) is x

    transform.p = 1.0
    transform.set_activation_status(False)
    assert transform(x) is x

    transform.set_activation_status(True)
    batch_one = x[:1]
    assert transform(batch_one) is batch_one


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p": -0.1}, "p"),
        ({"p": 1.1}, "p"),
        ({"alpha": 0.0}, "alpha"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_mixstyle_rejects_invalid_configuration(kwargs, message):
    """Reject invalid probability and numerical-stability settings clearly."""
    with pytest.raises(ValueError, match=message):
        MixStyle(**kwargs)


def test_mixstyle_rejects_non_bchw_training_input(monkeypatch):
    """Fail clearly when an activated training path receives something other than BCHW features."""
    monkeypatch.setattr("ultralytics.nn.modules.block.random.random", lambda: 0.0)
    with pytest.raises(ValueError, match="BCHW"):
        MixStyle(p=1.0).train()(torch.randn(2, 3, 8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixstyle_preserves_shape_dtype_device_and_input(dtype):
    """Keep the feature contract and avoid in-place modification on supported CPU dtypes."""
    x = torch.randn(4, 6, 8, 10, dtype=dtype)
    original = x.clone()
    random.seed(31)
    torch.manual_seed(37)
    output = MixStyle(p=1.0).train()(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_c3k2_wrapper_preserves_state_and_both_eval_forwards():
    """Retain all C3k2 parameter keys and make both wrapper forward paths exact identities in evaluation."""
    torch.manual_seed(41)
    baseline = C3k2(16, 32, n=2, c3k=False, e=0.25).eval()
    candidate = C3k2MixStyle(16, 32, n=2, c3k=False, e=0.25).eval()
    assert tuple(candidate.state_dict()) == tuple(baseline.state_dict())
    assert {key: value.shape for key, value in candidate.state_dict().items()} == {
        key: value.shape for key, value in baseline.state_dict().items()
    }
    assert sum(parameter.numel() for parameter in candidate.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(2, 16, 13, 17)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)
        torch.testing.assert_close(candidate.forward_split(x), baseline.forward_split(x), rtol=0, atol=0)


def test_yaml_changes_only_layer_two_module_and_executes_once_at_p2(monkeypatch):
    """Prove exact YAML isolation and runtime execution after the completed P2/4 backbone block."""
    baseline_yaml = YAML.load("ultralytics/cfg/models/11/yolo11.yaml")
    candidate_yaml = YAML.load("ultralytics/cfg/models/11/yolo11-mixstyle-p2.yaml")
    expected_yaml = deepcopy(baseline_yaml)
    expected_yaml["backbone"][2][2] = "C3k2MixStyle"
    assert candidate_yaml == expected_yaml

    model = YOLO("yolo11n-mixstyle-p2.yaml").model
    mixstyle_modules = [module for module in model.modules() if isinstance(module, MixStyle)]
    assert len(mixstyle_modules) == 1
    assert isinstance(model.model[2], C3k2MixStyle)
    assert mixstyle_modules[0] is model.model[2].mixstyle
    assert model.stride.tolist() == [8.0, 16.0, 32.0]

    class FixedBeta:
        @staticmethod
        def sample(shape):
            return torch.full(shape, 0.5)

    mixstyle = mixstyle_modules[0]
    mixstyle.p = 1.0
    mixstyle.beta = FixedBeta()
    monkeypatch.setattr("ultralytics.nn.modules.block.random.random", lambda: 0.0)
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda size, device=None: torch.arange(size - 1, -1, -1, device=device),
    )
    captures = []

    def capture(_module, inputs, output):
        captures.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = mixstyle.register_forward_hook(capture)
    model.train()
    images = torch.stack((torch.zeros(3, 64, 64), torch.ones(3, 64, 64)))
    with torch.no_grad():
        model(images)
    handle.remove()
    assert len(captures) == 1
    assert captures[0][0].shape[-2:] == (16, 16)
    assert not torch.equal(captures[0][0], captures[0][1])
    before_statistics = torch.cat(
        (captures[0][0].mean(dim=(2, 3)), captures[0][0].std(dim=(2, 3))), dim=1
    )
    after_statistics = torch.cat(
        (captures[0][1].mean(dim=(2, 3)), captures[0][1].std(dim=(2, 3))), dim=1
    )
    assert not torch.allclose(before_statistics, after_statistics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixstyle_cuda_amp_backward():
    """Keep CUDA autocast outputs and gradients finite when hardware is available."""
    device = torch.device("cuda:0")
    transform = MixStyle(p=1.0).to(device).train()
    x = torch.randn(4, 8, 16, 16, device=device, requires_grad=True)
    random.seed(43)
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = transform(x)
        loss = output.square().mean()
    loss.backward()
    assert output.device == device
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
