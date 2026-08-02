# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import (
    BboxLoss,
    InterpIoUBboxLoss,
    bbox_interpiou,
    v8DetectionInterpIoULoss,
)
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        ((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0), 0.9999996500000616),
        ((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0), 0.10383348698222727),
        ((0.0, 0.0, 4.0, 4.0), (1.0, 1.0, 3.0, 3.0), 0.21116856058761546),
        ((3.0, 3.0, 4.0, 4.0), (0.0, 0.0, 1.0, 1.0), -0.20852776400585282),
    ],
)
def test_bbox_interpiou_matches_frozen_static_reference(pred, target, expected):
    """Check identical, partial-overlap, enclosing, and non-overlap scores against frozen reference values."""
    actual = bbox_interpiou(
        torch.tensor([pred], dtype=torch.float64),
        torch.tensor([target], dtype=torch.float64),
        xywh=False,
        interpolation_alpha=0.98,
    )
    torch.testing.assert_close(actual, torch.tensor([[expected]], dtype=torch.float64), rtol=0, atol=1e-12)


def test_bbox_interpiou_xywh_equivalence_and_default_alpha():
    """Check equivalent coordinate formats and the fixed default interpolation coefficient."""
    pred_xyxy = torch.tensor([[0.2, -0.3, 2.8, 3.1], [3.0, 2.0, 5.5, 7.0]], dtype=torch.float64)
    target_xyxy = torch.tensor([[0.0, 0.0, 3.0, 3.0], [2.5, 2.5, 6.0, 6.5]], dtype=torch.float64)

    xyxy_score = bbox_interpiou(pred_xyxy, target_xyxy, xywh=False)
    xywh_score = bbox_interpiou(xyxy2xywh(pred_xyxy), xyxy2xywh(target_xyxy), xywh=True)
    explicit_score = bbox_interpiou(pred_xyxy, target_xyxy, xywh=False, interpolation_alpha=0.98)

    torch.testing.assert_close(xyxy_score, xywh_score, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(xyxy_score, explicit_score, rtol=0, atol=0)


def test_bbox_interpiou_shapes_scale_invariance_and_edge_cases():
    """Check broadcasting, approximate scale invariance, large/tiny boxes, and degenerate-box finiteness."""
    pred = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.1, 1.8, 2.1], [3000.0, 3000.0, 4000.0, 4000.0], [0.0, 0.0, 1e-4, 1e-4]],
        ]
    )
    target = torch.tensor([[[0.5, 0.5, 2.5, 2.5], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    score = bbox_interpiou(pred, target, xywh=False)
    scaled_score = bbox_interpiou(pred * 32, target * 32, xywh=False)

    assert score.shape == (2, 3, 1)
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score[..., :2, :], scaled_score[..., :2, :], rtol=1e-5, atol=1e-6)


def test_bbox_interpiou_supplies_gradient_for_nonoverlapping_boxes():
    """Check that interpolation supplies a finite nonzero gradient where standard IoU is flat."""
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    standard_pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], requires_grad=True)
    bbox_iou(standard_pred, target, xywh=False).sum().backward()
    assert standard_pred.grad is not None
    assert torch.count_nonzero(standard_pred.grad) == 0

    interp_pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], requires_grad=True)
    (1.0 - bbox_interpiou(interp_pred, target, xywh=False)).sum().backward()
    assert interp_pred.grad is not None
    assert torch.isfinite(interp_pred.grad).all()
    assert interp_pred.grad.abs().sum() > 0


def test_bbox_interpiou_gradcheck():
    """Check analytical gradients away from min/max and clamp boundaries."""
    pred = torch.tensor([[-2.7, -1.3, 1.4, 2.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-1.2, -0.8, 2.6, 1.7]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda boxes: bbox_interpiou(boxes, target, xywh=False, interpolation_alpha=0.98),
        (pred,),
        atol=1e-4,
        rtol=1e-3,
    )


def test_interpiou_bbox_loss_preserves_baseline_ciou_and_dfl_paths():
    """Prove the default class retains CIoU and the InterpIoU subclass leaves DFL unchanged."""
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
    interpiou, interpiou_dfl = InterpIoUBboxLoss(reg_max)(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    expected_ciou = (
        (1.0 - bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)) * weight
    ).sum() / target_scores_sum
    torch.testing.assert_close(baseline_iou, expected_ciou)
    torch.testing.assert_close(baseline_dfl, interpiou_dfl)
    assert not torch.isclose(baseline_iou, interpiou)


def test_detection_model_routes_to_static_interpiou_criterion():
    """Check that an ordinary YOLO11 Detect head selects the static InterpIoU criterion."""
    model = DetectionModel("yolo11n.yaml", verbose=False)
    model.args = get_cfg()
    criterion = model.init_criterion()

    assert model.model[-1].__class__ is Detect
    assert type(criterion) is v8DetectionInterpIoULoss
    assert type(criterion.bbox_loss) is InterpIoUBboxLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bbox_interpiou_cuda_amp():
    """Check finite static InterpIoU loss and gradients for FP16 tensors under CUDA autocast."""
    pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = (1.0 - bbox_interpiou(pred, target, xywh=False)).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
