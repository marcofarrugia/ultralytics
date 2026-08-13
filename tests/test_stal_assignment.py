# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.utils.loss import v8DetectionSTALLoss
from ultralytics.utils.tal import STALTaskAlignedAssigner, TaskAlignedAssigner


def _reference_stal_mask(xy_centers, gt_bboxes, mask_gt, min_stride=8.0, reference_size=16.0, eps=1e-9):
    """Return a compact, independent implementation of the paper's candidate-only STAL geometry."""
    original_lt, original_rb = gt_bboxes.chunk(2, dim=-1)
    centers = (original_lt + original_rb) / 2
    enlarge = ((original_rb - original_lt) < min_stride) & mask_gt.bool()
    half_reference = torch.full_like(centers, reference_size / 2)
    lt = torch.where(enlarge, centers - half_reference, original_lt).unsqueeze(2)
    rb = torch.where(enlarge, centers + half_reference, original_rb).unsqueeze(2)
    return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)


def _grid_centers(device="cpu", dtype=torch.float32):
    return torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]], device=device, dtype=dtype)


def _pyramid_centers(device="cpu", dtype=torch.float32):
    """Return the 8,400 P3/P4/P5 anchor centres used by a 640-pixel YOLO11 input."""
    levels = []
    for stride in (8, 16, 32):
        axis = torch.arange(640 // stride, device=device, dtype=dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        levels.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2) * stride)
    return torch.cat(levels)


def test_stal_enlarges_width_and_height_independently_but_not_at_exact_threshold():
    """Cover ordinary, one-axis tiny, two-axis tiny, and exact-eight-pixel candidate geometry."""
    xy_centers = _grid_centers()
    gt_bboxes = torch.tensor(
        [
            [
                [2.0, 2.0, 14.0, 14.0],  # ordinary 12x12
                [5.0, 2.0, 11.0, 14.0],  # width-only 6x12
                [2.0, 5.0, 14.0, 11.0],  # height-only 12x6
                [5.0, 5.0, 11.0, 11.0],  # both dimensions 6x6
                [4.0, 5.0, 12.0, 11.0],  # exact width 8, tiny height 6
                [5.0, 4.0, 11.0, 12.0],  # tiny width 6, exact height 8
            ]
        ]
    )
    mask_gt = torch.ones((1, 6, 1), dtype=torch.bool)

    vanilla = TaskAlignedAssigner.select_candidates_in_gts(xy_centers, gt_bboxes)
    stal = STALTaskAlignedAssigner().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)

    assert torch.equal(stal[:, 0], vanilla[:, 0])
    assert vanilla[0, 1:4].sum() == 0
    assert torch.equal(stal[0, 1:4].sum(1), torch.full((3,), 4))
    assert vanilla[0, 4:].sum() == 0
    assert stal[0, 4:].sum() == 0  # dimensions equal to 8 must not be enlarged
    assert not (vanilla.bool() & ~stal).any()


def test_stal_preserves_boundary_aligned_ordinary_mask_and_never_removes_candidates():
    """Regress the float32 round-trip failure found by the full pre-training assignment diagnostic."""
    xy_centers = _pyramid_centers()
    gt_bboxes = torch.tensor(
        [
            [
                [236.0, 361.0, 279.1666564941406, 404.1666564941406],  # ordinary box with an anchor-aligned edge
                [237.0, 361.0, 243.0, 404.1666564941406],  # width-only tiny; preserve the y edges exactly
                [236.0, 397.0, 279.1666564941406, 403.0],  # height-only tiny; preserve the x edges exactly
                [237.0, 397.0, 243.0, 403.0],  # both dimensions tiny
            ]
        ],
        dtype=torch.float32,
    )
    mask_gt = torch.ones((1, 4, 1), dtype=torch.bool)

    vanilla = TaskAlignedAssigner.select_candidates_in_gts(xy_centers, gt_bboxes).bool()
    stal = STALTaskAlignedAssigner().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)
    changed = stal ^ vanilla
    tiny_objects = ((gt_bboxes[..., 2:] - gt_bboxes[..., :2]) < 8).any(-1, keepdim=True)

    assert torch.equal(stal[:, 0], vanilla[:, 0])
    assert not (vanilla & ~stal).any()
    assert not (changed & ~tiny_objects).any()


def test_stal_matches_reference_preserves_inputs_and_keeps_padding_inactive():
    """Match the pinned upstream formula while proving padded boxes and caller-owned tensors remain untouched."""
    xy_centers = _grid_centers()
    gt_bboxes = torch.tensor(
        [[[5.0, 5.0, 11.0, 11.0], [0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 14.0, 14.0]]]
    )
    mask_gt = torch.tensor([[[True], [False], [True]]])
    original = gt_bboxes.clone()

    actual = STALTaskAlignedAssigner().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)
    expected = _reference_stal_mask(xy_centers, gt_bboxes, mask_gt)

    assert torch.equal(actual, expected)
    assert not actual[0, 1].any()
    torch.testing.assert_close(gt_bboxes, original, rtol=0, atol=0)


def test_stal_recovers_a_tiny_box_with_no_vanilla_candidate():
    """Prove that a grid-misaligned 6x6 target changes from zero candidates to four candidates."""
    xy_centers = _grid_centers()
    gt_bboxes = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

    vanilla = TaskAlignedAssigner.select_candidates_in_gts(xy_centers, gt_bboxes)
    stal = STALTaskAlignedAssigner().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)

    assert vanilla.sum() == 0
    assert stal.sum() == 4


def test_stal_uses_original_boxes_for_scoring_and_targets(monkeypatch):
    """Spy on the full assigner path to prove the surrogate does not escape candidate containment."""
    assigner = STALTaskAlignedAssigner(topk=4, num_classes=1)
    xy_centers = _grid_centers()
    gt_bboxes = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])
    original = gt_bboxes.clone()
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)
    pd_scores = torch.full((1, 4, 1), 0.75)
    pd_bboxes = torch.tensor([[[4.0, 4.0, 12.0, 12.0]]] * 4).reshape(1, 4, 4)
    gt_labels = torch.zeros((1, 1, 1))
    observed = {}

    get_box_metrics = assigner.get_box_metrics
    get_targets = assigner.get_targets

    def spy_get_box_metrics(scores, boxes, labels, targets, mask):
        observed["scoring"] = targets.clone()
        return get_box_metrics(scores, boxes, labels, targets, mask)

    def spy_get_targets(labels, targets, target_gt_idx, fg_mask):
        observed["targets"] = targets.clone()
        return get_targets(labels, targets, target_gt_idx, fg_mask)

    monkeypatch.setattr(assigner, "get_box_metrics", spy_get_box_metrics)
    monkeypatch.setattr(assigner, "get_targets", spy_get_targets)
    _, _, _, fg_mask, _ = assigner(pd_scores, pd_bboxes, xy_centers, gt_labels, gt_bboxes, mask_gt)

    assert fg_mask.any()
    torch.testing.assert_close(observed["scoring"], original, rtol=0, atol=0)
    torch.testing.assert_close(observed["targets"], original, rtol=0, atol=0)
    torch.testing.assert_close(gt_bboxes, original, rtol=0, atol=0)


class _DummyDetectionModel(nn.Module):
    """Minimal model contract needed to construct the detection criterion."""

    def __init__(self, strides):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.model = [SimpleNamespace(stride=torch.tensor(strides), nc=14, reg_max=16)]


def test_stal_loss_preserves_baseline_settings_and_rejects_nonstandard_pyramids():
    """Lock the registered STAL-16 arm to the standard P3/P4/P5 YOLO11 detection pyramid."""
    criterion = v8DetectionSTALLoss(_DummyDetectionModel([8.0, 16.0, 32.0]))

    assert isinstance(criterion.assigner, STALTaskAlignedAssigner)
    assert criterion.assigner.topk == 10
    assert criterion.assigner.alpha == 0.5
    assert criterion.assigner.beta == 6.0
    assert criterion.assigner.strides == (8.0, 16.0, 32.0)

    with pytest.raises(ValueError, match=r"requires the standard P3/P4/P5 detection strides"):
        v8DetectionSTALLoss(_DummyDetectionModel([4.0, 8.0, 16.0, 32.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stal_candidate_selection_cuda_autocast():
    """Check that candidate selection matches the reference under CUDA float16 autocast."""
    xy_centers = _grid_centers(device="cuda", dtype=torch.float16)
    gt_bboxes = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]], device="cuda", dtype=torch.float16)
    mask_gt = torch.ones((1, 1, 1), device="cuda", dtype=torch.bool)

    with torch.autocast("cuda", dtype=torch.float16):
        actual = STALTaskAlignedAssigner().select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)
        expected = _reference_stal_mask(xy_centers, gt_bboxes, mask_gt)

    assert torch.equal(actual, expected)
