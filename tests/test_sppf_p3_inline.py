# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import deepcopy

import torch

from ultralytics.nn.modules import Detect, SPPF
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ROOT, YAML

BASELINE_YAML = ROOT / "cfg/models/11/yolo11.yaml"
ABLATION_YAML = ROOT / "cfg/models/11/yolo11-sppf-p3-inline.yaml"


def test_sppf_p3_inline_yaml_is_isolated_graph_change():
    """Verify the ablation only inserts the P3 SPPF and updates the shifted Detect inputs."""
    baseline = YAML.load(BASELINE_YAML)
    ablation = YAML.load(ABLATION_YAML)
    expected = deepcopy(baseline)
    expected["head"].insert(6, [-1, 1, "SPPF", [256, 5]])
    expected["head"][-1][0] = [17, 20, 23]

    assert ablation["backbone"] == expected["backbone"]
    assert ablation["head"] == expected["head"]
    assert ablation["nc"] == expected["nc"]
    assert ablation["scales"] == expected["scales"]


def test_sppf_p3_inline_model_graph():
    """Verify the stock SPPF instances, PAN continuation, detection inputs, and strides."""
    model = DetectionModel(ABLATION_YAML, ch=3, nc=80, verbose=False)
    sppf_layers = [(module.i, module) for module in model.model if isinstance(module, SPPF)]

    assert len(model.model) == 25
    assert [index for index, _ in sppf_layers] == [9, 17]
    assert all(module.m.kernel_size == 5 for _, module in sppf_layers)
    assert model.model[18].f == -1
    assert isinstance(model.model[24], Detect)
    assert model.model[24].f == [17, 20, 23]
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_sppf_p3_inline_output_feeds_pan_and_detect():
    """Prove that one P3 SPPF output feeds both the PAN descent and P3 Detect branch."""
    model = DetectionModel(ABLATION_YAML, ch=3, nc=80, verbose=False).eval()
    sppf_outputs = []
    pan_inputs = []
    detect_inputs = []

    hooks = [
        model.model[17].register_forward_hook(lambda _module, _args, output: sppf_outputs.append(output)),
        model.model[18].register_forward_pre_hook(lambda _module, args: pan_inputs.append(args[0])),
        model.model[24].register_forward_pre_hook(lambda _module, args: detect_inputs.append(tuple(args[0]))),
    ]
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64))
    finally:
        for hook in hooks:
            hook.remove()

    assert len(sppf_outputs) == len(pan_inputs) == len(detect_inputs) == 1
    assert pan_inputs[0] is sppf_outputs[0]
    assert detect_inputs[0][0] is sppf_outputs[0]


def test_sppf_p3_inline_keeps_baseline_constructible():
    """Confirm that the unchanged baseline architecture still constructs with its original graph."""
    baseline = DetectionModel(BASELINE_YAML, ch=3, nc=80, verbose=False)

    assert len(baseline.model) == 24
    assert baseline.model[-1].f == [16, 19, 22]
    assert baseline.stride.tolist() == [8.0, 16.0, 32.0]
