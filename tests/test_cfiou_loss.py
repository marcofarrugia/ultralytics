# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.utils.loss import BboxLoss, CFIoUBboxLoss, bbox_cfiou_loss
from ultralytics.utils.metrics import bbox_iou


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        ((-2.0, -2.0, 2.0, 2.0), (-2.0, -2.0, 2.0, 2.0), 0.0),
        ((-1.0, -1.0, 1.0, 1.0), (-2.0, -2.0, 2.0, 2.0), 0.8125),
        ((-4.0, -4.0, 4.0, 4.0), (-2.0, -2.0, 2.0, 2.0), 1.375),
        ((-4.0, -4.0, 4.0, 4.0), (-1.0, -2.0, 3.0, 2.0), 1.3828125),
    ],
)
def test_bbox_cfiou_loss_matches_paper_examples(pred, target, expected):
    """Reproduce the identical, enclosing, and offset examples reported in the CFIoU paper."""
    actual = bbox_cfiou_loss(torch.tensor([pred]), torch.tensor([target]))
    torch.testing.assert_close(actual, torch.tensor([[expected]]), rtol=0, atol=1e-6)


def test_bbox_cfiou_loss_shapes_scale_invariance_and_edge_cases():
    """Check batched shape preservation, scale invariance, non-overlap, and degenerate-box finiteness."""
    pred = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.1, 1.8, 2.1], [6.0, 5.0, 7.0, 8.0], [0.0, 0.0, 1e-4, 1e-4]],
        ]
    )
    target = torch.tensor(
        [
            [[0.5, 0.5, 2.5, 2.5], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 2e-4, 2e-4]],
        ]
    )

    loss = bbox_cfiou_loss(pred, target)
    scaled_loss = bbox_cfiou_loss(pred * 32, target * 32)

    assert loss.shape == (2, 3, 1)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss[:, :2], scaled_loss[:, :2], rtol=1e-5, atol=1e-6)


def test_bbox_cfiou_loss_gradients_on_both_foreground_branches():
    """Check finite gradients when centers coincide and when they differ."""
    target = torch.tensor([[-2.0, -2.0, 2.0, 2.0]])
    for values in ((-1.0, -1.0, 1.0, 1.0), (-1.2, -1.0, 1.0, 1.0)):
        pred = torch.tensor([values], requires_grad=True)
        bbox_cfiou_loss(pred, target).sum().backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


def test_bbox_cfiou_loss_gradcheck():
    """Check analytical gradients away from min/max and center-gate boundaries."""
    pred = torch.tensor([[-2.7, -1.3, 1.4, 2.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-1.2, -0.8, 2.6, 1.7]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda boxes: bbox_cfiou_loss(boxes, target), (pred,), atol=1e-4, rtol=1e-3)


def test_cfiou_bbox_loss_preserves_baseline_ciou_and_dfl_paths():
    """Prove the default class retains CIoU and the CFIoU subclass leaves DFL unchanged."""
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
    cfiou, cfiou_dfl = CFIoUBboxLoss(reg_max)(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    expected_ciou = (
        (1.0 - bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)) * weight
    ).sum() / target_scores_sum
    torch.testing.assert_close(baseline_iou, expected_ciou)
    torch.testing.assert_close(baseline_dfl, cfiou_dfl)
    assert not torch.isclose(baseline_iou, cfiou)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bbox_cfiou_loss_cuda_amp():
    """Check low-precision CUDA inputs are promoted and produce finite loss and gradients under AMP."""
    pred = torch.tensor([[-160.0, -160.0, 160.0, 160.0]], device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.tensor([[-80.0, -80.0, 80.0, 80.0]], device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = bbox_cfiou_loss(pred, target).sum()
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
