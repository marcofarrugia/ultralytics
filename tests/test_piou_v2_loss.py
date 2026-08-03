# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, PIoUv2BboxLoss, bbox_piou_v2_loss, v8DetectionPIoUv2Loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        ((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0), 8.99532565342349e-8),
        ((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0), 1.369827818555939),
        ((0.0, 0.0, 4.0, 4.0), (1.0, 1.0, 3.0, 3.0), 1.2337232674961545),
        ((2.5, 0.0, 4.5, 2.0), (0.0, 0.0, 2.0, 2.0), 1.7022689770841237),
        ((10.0, 10.0, 12.0, 12.0), (0.0, 0.0, 2.0, 2.0), 0.0525519543428038),
    ],
)
def test_bbox_piou_v2_loss_matches_frozen_reference(pred, target, expected):
    """Check identical, partial-overlap, enclosing, moderate, and very poor boxes against frozen values."""
    actual = bbox_piou_v2_loss(
        torch.tensor([pred], dtype=torch.float64),
        torch.tensor([target], dtype=torch.float64),
        xywh=False,
    )
    torch.testing.assert_close(actual, torch.tensor([[expected]], dtype=torch.float64), rtol=0, atol=1e-12)


def test_bbox_piou_v2_loss_xywh_equivalence_and_default_lambda():
    """Check coordinate-format equivalence and the fixed default focusing coefficient."""
    pred_xyxy = torch.tensor([[0.2, -0.3, 2.8, 3.1], [3.0, 2.0, 5.5, 7.0]], dtype=torch.float64)
    target_xyxy = torch.tensor([[0.0, 0.0, 3.0, 3.0], [2.5, 2.5, 6.0, 6.5]], dtype=torch.float64)

    xyxy_loss = bbox_piou_v2_loss(pred_xyxy, target_xyxy, xywh=False)
    xywh_loss = bbox_piou_v2_loss(xyxy2xywh(pred_xyxy), xyxy2xywh(target_xyxy), xywh=True)
    explicit_loss = bbox_piou_v2_loss(pred_xyxy, target_xyxy, xywh=False, focusing_lambda=1.3)

    torch.testing.assert_close(xyxy_loss, xywh_loss, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(xyxy_loss, explicit_loss, rtol=0, atol=0)


def test_bbox_piou_v2_loss_shapes_scale_invariance_and_edge_cases():
    """Check broadcasting, ordinary-box scale invariance, and tiny or degenerate target finiteness."""
    pred = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.1, 1.8, 2.1], [3000.0, 3000.0, 4000.0, 4000.0], [0.0, 0.0, 1e-4, 1e-4]],
        ]
    )
    target = torch.tensor([[[0.5, 0.5, 2.5, 2.5], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    loss = bbox_piou_v2_loss(pred, target, xywh=False)
    scaled_loss = bbox_piou_v2_loss(pred * 32, target * 32, xywh=False)

    assert loss.shape == (2, 3, 1)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss[..., :2, :], scaled_loss[..., :2, :], rtol=1e-5, atol=1e-6)


def test_bbox_piou_v2_loss_has_nonmonotonic_attention():
    """Check that PIoU v2 emphasizes medium-quality boxes over high- and extremely low-quality boxes."""
    penalty_values = torch.tensor([0.1, math.log(math.sqrt(2) * 1.3), 5.0], dtype=torch.float64)
    delta = 2 * penalty_values
    pred = torch.stack((-delta, -delta, 2 + delta, 2 + delta), dim=-1)
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float64).expand_as(pred)

    loss = bbox_piou_v2_loss(pred, target, xywh=False)
    piou_v1 = 2.0 - bbox_iou(pred, target, xywh=False) - torch.exp(-penalty_values.square()).unsqueeze(-1)
    attention = (loss / piou_v1).squeeze(-1)
    expected = torch.tensor([0.8845480701805427, 1.2866458274410604, 0.02627597717158436], dtype=torch.float64)

    torch.testing.assert_close(attention, expected, rtol=0, atol=1e-12)
    assert attention[1] > attention[0] > attention[2]


def test_bbox_piou_v2_loss_supplies_gradient_for_moderate_nonoverlap():
    """Check finite nonzero gradients for a moderately poor non-overlapping prediction."""
    pred = torch.tensor([[2.5, 0.0, 4.5, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    bbox_piou_v2_loss(pred, target, xywh=False).sum().backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_bbox_piou_v2_loss_gradcheck():
    """Check analytical gradients away from absolute-value and overlap boundaries."""
    pred = torch.tensor([[-2.7, -1.3, 1.4, 2.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-1.2, -0.8, 2.6, 1.7]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda boxes: bbox_piou_v2_loss(boxes, target, xywh=False),
        (pred,),
        atol=1e-4,
        rtol=1e-3,
    )


def test_piou_v2_bbox_loss_preserves_baseline_ciou_and_dfl_paths():
    """Prove the default class retains CIoU and the PIoU v2 subclass leaves DFL bit-identical."""
    reg_max = 4
    pred_dist = torch.linspace(-1.0, 1.0, 3 * 4 * reg_max).reshape(1, 3, 4 * reg_max)
    pred_bboxes = torch.tensor(
        [[[1.1, 0.9, 3.2, 2.8], [3.2, 2.9, 5.1, 5.2], [5.0, 5.0, 7.0, 7.0]]]
    )
    target_bboxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0], [3.0, 3.0, 5.0, 5.0], [5.0, 5.0, 7.0, 7.0]]])
    anchor_points = torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    target_scores = torch.tensor([[[0.8, 0.1], [0.5, 0.2], [0.0, 0.0]]])
    target_scores_sum = target_scores.sum()
    fg_mask = torch.tensor([[True, True, False]])

    baseline_iou, baseline_dfl = BboxLoss(reg_max)(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )
    piou_v2, piou_v2_dfl = PIoUv2BboxLoss(reg_max)(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    expected_ciou = (
        (1.0 - bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)) * weight
    ).sum() / target_scores_sum
    torch.testing.assert_close(baseline_iou, expected_ciou)
    assert torch.equal(baseline_dfl, piou_v2_dfl)
    assert not torch.isclose(baseline_iou, piou_v2)


@pytest.mark.parametrize("scale", ("n", "s", "m", "l", "x"))
def test_detection_model_routes_all_scales_to_piou_v2_criterion(scale):
    """Check every supported YOLO11 scale selects PIoU v2 and retains P3-P5 strides."""
    model = DetectionModel(f"yolo11{scale}.yaml", verbose=False)
    model.args = get_cfg()
    criterion = model.init_criterion()

    assert model.model[-1].__class__ is Detect
    assert type(criterion) is v8DetectionPIoUv2Loss
    assert type(criterion.bbox_loss) is PIoUv2BboxLoss
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_yolo11n_piou_v2_synthetic_loss_backward():
    """Check a complete synthetic YOLO11n criterion forward and backward produces finite values."""
    model = DetectionModel("yolo11n.yaml", verbose=False)
    model.args = get_cfg()
    model.train()
    predictions = model(torch.rand(1, 3, 64, 64))
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([0.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }

    loss_components, detached_components = model.init_criterion()(predictions, batch)
    loss_components.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]

    assert torch.isfinite(loss_components).all()
    assert torch.isfinite(detached_components).all()
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bbox_piou_v2_loss_cuda_amp():
    """Check low-precision CUDA inputs are promoted and produce finite loss and gradients under AMP."""
    pred = torch.tensor([[2.5, 0.0, 4.5, 2.0]], device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]], device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = bbox_piou_v2_loss(pred, target, xywh=False).sum()
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
