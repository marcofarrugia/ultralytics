# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import (
    BboxLoss,
    DynamicInterpIoUBboxLoss,
    bbox_dynamic_interpiou,
    v8DetectionDynamicInterpIoULoss,
)
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh


def _iou_reference(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return standard IoU using the arithmetic employed by the official Dynamic InterpIoU implementation."""
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    return inter / (w1 * h1 + w2 * h2 - inter + eps)


def _fixed_interpiou_reference(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    interpolation_alpha: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return an xyxy InterpIoU score while treating the supplied coefficient as fixed."""
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
    iou = _iou_reference(pred_bboxes, target_bboxes, eps)

    bi_x1 = (1 - interpolation_alpha) * b1_x1 + interpolation_alpha * b2_x1
    bi_y1 = (1 - interpolation_alpha) * b1_y1 + interpolation_alpha * b2_y1
    bi_x2 = (1 - interpolation_alpha) * b1_x2 + interpolation_alpha * b2_x2
    bi_y2 = (1 - interpolation_alpha) * b1_y2 + interpolation_alpha * b2_y2
    inter_i = (torch.min(bi_x2, b2_x2) - torch.max(bi_x1, b2_x1)).clamp(0) * (
        torch.min(bi_y2, b2_y2) - torch.max(bi_y1, b2_y1)
    ).clamp(0)
    wi, hi = bi_x2 - bi_x1 + eps, bi_y2 - bi_y1 + eps
    w2, h2 = b2_x2 - b2_x1 + eps, b2_y2 - b2_y1 + eps
    iou_i = inter_i / (wi * hi + w2 * h2 - inter_i + eps)
    return iou + iou_i - 1


@pytest.mark.parametrize(
    ("pred", "target", "expected_score", "expected_alpha"),
    [
        ((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0), 0.9999996500000616, 0.60),
        ((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0), -0.0992954689455816, 0.8571428673469380),
        ((0.0, 0.0, 4.0, 4.0), (1.0, 1.0, 3.0, 3.0), -0.1100001021374882, 0.7500000109374996),
        ((3.0, 3.0, 4.0, 4.0), (0.0, 0.0, 1.0, 1.0), -0.1116046116492420, 0.99),
    ],
)
def test_bbox_dynamic_interpiou_matches_frozen_reference(pred, target, expected_score, expected_alpha):
    """Check frozen scores and the max, interior, and min dynamic-coefficient regimes."""
    pred_tensor = torch.tensor([pred], dtype=torch.float64)
    target_tensor = torch.tensor([target], dtype=torch.float64)
    actual = bbox_dynamic_interpiou(pred_tensor, target_tensor, xywh=False)
    alpha = torch.tensor([[expected_alpha]], dtype=torch.float64)
    reference = _fixed_interpiou_reference(pred_tensor, target_tensor, alpha)

    torch.testing.assert_close(actual, torch.tensor([[expected_score]], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)


def test_bbox_dynamic_interpiou_xywh_equivalence_and_default_bounds():
    """Check equivalent coordinate formats and the pre-registered default coefficient bounds."""
    pred_xyxy = torch.tensor([[0.2, -0.3, 2.8, 3.1], [3.0, 2.0, 5.5, 7.0]], dtype=torch.float64)
    target_xyxy = torch.tensor([[0.0, 0.0, 3.0, 3.0], [2.5, 2.5, 6.0, 6.5]], dtype=torch.float64)

    xyxy_score = bbox_dynamic_interpiou(pred_xyxy, target_xyxy, xywh=False)
    xywh_score = bbox_dynamic_interpiou(xyxy2xywh(pred_xyxy), xyxy2xywh(target_xyxy), xywh=True)
    explicit_score = bbox_dynamic_interpiou(
        pred_xyxy,
        target_xyxy,
        xywh=False,
        interpolation_alpha_min=0.60,
        interpolation_alpha_max=0.99,
    )

    torch.testing.assert_close(xyxy_score, xywh_score, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(xyxy_score, explicit_score, rtol=0, atol=0)


def test_bbox_dynamic_interpiou_shapes_scale_invariance_and_edge_cases():
    """Check broadcasting, approximate scale invariance, large/tiny boxes, and degenerate-box finiteness."""
    pred = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.1, 1.8, 2.1], [3000.0, 3000.0, 4000.0, 4000.0], [0.0, 0.0, 1e-4, 1e-4]],
        ]
    )
    target = torch.tensor([[[0.5, 0.5, 2.5, 2.5], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    score = bbox_dynamic_interpiou(pred, target, xywh=False)
    scaled_score = bbox_dynamic_interpiou(pred * 32, target * 32, xywh=False)

    assert score.shape == (2, 3, 1)
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score[..., :2, :], scaled_score[..., :2, :], rtol=1e-5, atol=1e-6)


def test_bbox_dynamic_interpiou_supplies_gradient_for_nonoverlapping_boxes():
    """Check that interpolation supplies a finite nonzero gradient where standard IoU is flat."""
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    standard_pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], requires_grad=True)
    bbox_iou(standard_pred, target, xywh=False).sum().backward()
    assert standard_pred.grad is not None
    assert torch.count_nonzero(standard_pred.grad) == 0

    dynamic_pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], requires_grad=True)
    (1.0 - bbox_dynamic_interpiou(dynamic_pred, target, xywh=False)).sum().backward()
    assert dynamic_pred.grad is not None
    assert torch.isfinite(dynamic_pred.grad).all()
    assert dynamic_pred.grad.abs().sum() > 0


def test_bbox_dynamic_interpiou_stops_gradient_through_coefficient():
    """Check that mid-range adaptation changes the forward value but is detached during backpropagation."""
    target = torch.tensor([[0.7, 0.7, 2.7, 2.7]], dtype=torch.float64)
    dynamic_pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float64, requires_grad=True)
    dynamic_score = bbox_dynamic_interpiou(dynamic_pred, target, xywh=False)
    dynamic_score.sum().backward()

    reference_pred = dynamic_pred.detach().clone().requires_grad_(True)
    with torch.no_grad():
        alpha = torch.clamp(1 - _iou_reference(reference_pred, target), min=0.60, max=0.99)
    reference_score = _fixed_interpiou_reference(reference_pred, target, alpha)
    reference_score.sum().backward()

    assert 0.60 < alpha.item() < 0.99
    torch.testing.assert_close(dynamic_score, reference_score, rtol=0, atol=1e-12)
    torch.testing.assert_close(dynamic_pred.grad, reference_pred.grad, rtol=1e-7, atol=1e-9)


def test_bbox_dynamic_interpiou_gradcheck_on_clamped_coefficient():
    """Check gradients where the detached dynamic coefficient remains clamped and locally constant."""
    pred = torch.tensor([[0.1, 0.1, 2.1, 2.1]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda boxes: bbox_dynamic_interpiou(boxes, target, xywh=False),
        (pred,),
        atol=1e-4,
        rtol=1e-3,
    )


def test_dynamic_interpiou_bbox_loss_preserves_baseline_ciou_and_dfl_paths():
    """Prove the default class retains CIoU and the dynamic subclass leaves DFL unchanged."""
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
    dynamic_iou, dynamic_dfl = DynamicInterpIoUBboxLoss(reg_max)(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    expected_ciou = (
        (1.0 - bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)) * weight
    ).sum() / target_scores_sum
    torch.testing.assert_close(baseline_iou, expected_ciou)
    torch.testing.assert_close(baseline_dfl, dynamic_dfl)
    assert not torch.isclose(baseline_iou, dynamic_iou)


def test_detection_model_routes_to_dynamic_interpiou_criterion():
    """Check that an ordinary YOLO11 Detect head selects the Dynamic InterpIoU criterion."""
    model = DetectionModel("yolo11n.yaml", verbose=False)
    model.args = get_cfg()
    criterion = model.init_criterion()

    assert model.model[-1].__class__ is Detect
    assert type(criterion) is v8DetectionDynamicInterpIoULoss
    assert type(criterion.bbox_loss) is DynamicInterpIoUBboxLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bbox_dynamic_interpiou_cuda_amp():
    """Check finite Dynamic InterpIoU loss and gradients for FP16 tensors under CUDA autocast."""
    pred = torch.tensor([[3.0, 3.0, 4.0, 4.0]], device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = (1.0 - bbox_dynamic_interpiou(pred, target, xywh=False)).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
