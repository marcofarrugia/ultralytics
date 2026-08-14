# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from itertools import product

import cv2
import numpy as np
import pytest

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data.augment import RandomHSV, UnderwaterColorRandomTransfer, v8_transforms
from ultralytics.data.dataset import YOLODataset


def _pinned_reference(img: np.ndarray, transform: UnderwaterColorRandomTransfer) -> np.ndarray:
    """Compact reference form of LEFTeyex/UnitModule@039c015 UCRT."""
    hue_enabled = np.random.rand() < transform.hue_prob
    saturation_enabled = np.random.rand() < transform.saturation_prob
    value_enabled = np.random.rand() < transform.value_prob
    if not any((hue_enabled, saturation_enabled, value_enabled)):
        return img

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    if hue_enabled:
        hue_mean = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0])
        hue_gain = np.random.uniform(-1, 1) * transform.hue_delta
        hue_min, hue_max = transform.underwater_hue_interval
        if hue_min < hue_mean < hue_max:
            hue_gain = np.clip(hue_mean + hue_gain, hue_min, hue_max) - hue_mean
        else:
            hue_gain = np.abs(hue_gain)
            if hue_mean >= hue_max:
                hue_gain = -hue_gain
        img_hsv[..., 0] = (img_hsv[..., 0] + np.array(hue_gain, dtype=np.int16)) % 180
    if saturation_enabled:
        gain = np.array(np.random.uniform(-1, 1) * transform.saturation_delta, dtype=np.int16)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + gain, 0, 255)
    if value_enabled:
        gain = np.array(np.random.uniform(-1, 1) * transform.value_delta, dtype=np.int16)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + gain, 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _solid_hsv(hue: int, saturation: int = 255, value: int = 255) -> np.ndarray:
    """Create a solid BGR image whose OpenCV hue is exactly controllable."""
    hsv = np.full((12, 16, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_ucrt_matches_pinned_reference_for_all_gate_combinations():
    """Match upstream arithmetic and random-call order across all eight independent H/S/V gate combinations."""
    image = np.random.default_rng(1234).integers(0, 256, (31, 47, 3), dtype=np.uint8)
    transform = UnderwaterColorRandomTransfer()
    seeds_by_gates = {}
    for seed in range(1000):
        np.random.seed(seed)
        gates = tuple(np.random.rand(3) < 0.5)
        seeds_by_gates.setdefault(gates, seed)
        if len(seeds_by_gates) == 8:
            break

    assert set(seeds_by_gates) == set(product((False, True), repeat=3))
    for seed in seeds_by_gates.values():
        np.random.seed(seed)
        expected = _pinned_reference(image.copy(), transform)
        labels = {"img": image.copy()}
        np.random.seed(seed)
        actual = transform(labels)["img"]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("hue", "multiplier", "expected_gain"),
    [(0, -0.8, 4), (18, -0.8, 4), (19, -1.0, -1), (50, -0.8, -4), (50, 0.8, 4), (115, 1.0, 1), (116, 0.8, -4), (150, -0.8, -4)],
)
def test_ucrt_hue_direction_and_boundaries(monkeypatch, hue, multiplier, expected_gain):
    """Use strict interior bounds and direct out-of-range or exact-boundary hues toward the underwater interval."""
    transform = UnderwaterColorRandomTransfer()
    monkeypatch.setattr(transform, "_random_multiplier", lambda: multiplier)
    assert int(transform._get_hue_gain(_solid_hsv(hue))) == expected_gain


def test_ucrt_modulo_clipping_and_annotation_preservation(monkeypatch):
    """Wrap hue, clip saturation/value, replace only the image, and leave geometry-related fields untouched."""
    image = _solid_hsv(179, saturation=250, value=20)
    original = image.copy()
    instances = object()
    cls = np.array([[3.0]], dtype=np.float32)
    segments = [np.array([[1.0, 2.0]], dtype=np.float32)]
    keypoints = np.array([[[3.0, 4.0, 1.0]]], dtype=np.float32)
    labels = {
        "img": image,
        "instances": instances,
        "cls": cls,
        "segments": segments,
        "keypoints": keypoints,
        "metadata": {"source": "sentinel"},
    }
    transform = UnderwaterColorRandomTransfer(hue_prob=1.0, saturation_prob=1.0, value_prob=1.0)
    monkeypatch.setattr(transform, "_get_hue_gain", lambda _: np.array(4, dtype=np.int16))
    monkeypatch.setattr(transform, "_get_saturation_gain", lambda: np.array(30, dtype=np.int16))
    monkeypatch.setattr(transform, "_get_value_gain", lambda: np.array(-30, dtype=np.int16))

    source_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    source_hsv[..., 0] = (source_hsv[..., 0] + 4) % 180
    source_hsv[..., 1] = np.clip(source_hsv[..., 1] + 30, 0, 255)
    source_hsv[..., 2] = np.clip(source_hsv[..., 2] - 30, 0, 255)
    expected = cv2.cvtColor(source_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    result = transform(labels)
    assert result is labels
    np.testing.assert_array_equal(result["img"], expected)
    np.testing.assert_array_equal(image, original)
    assert result["instances"] is instances
    assert result["cls"] is cls
    assert result["segments"] is segments
    assert result["keypoints"] is keypoints
    assert result["metadata"] == {"source": "sentinel"}


def test_ucrt_inactive_is_exact_identity():
    """Return the original mapping and image object when all three independent gates are disabled."""
    image = np.random.default_rng(9).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    labels = {"img": image, "unchanged": object()}
    result = UnderwaterColorRandomTransfer(0.0, 0.0, 0.0)(labels)
    assert result is labels
    assert result["img"] is image
    assert result["unchanged"] is labels["unchanged"]


def test_ucrt_is_seed_deterministic():
    """Use Ultralytics' seeded NumPy stream without introducing an untracked generator."""
    image = np.random.default_rng(19).integers(0, 256, (29, 37, 3), dtype=np.uint8)
    transform = UnderwaterColorRandomTransfer()
    np.random.seed(77)
    first = transform({"img": image.copy()})["img"]
    np.random.seed(77)
    second = transform({"img": image.copy()})["img"]
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hue_prob": -0.01}, "hue_prob"),
        ({"saturation_prob": 1.01}, "saturation_prob"),
        ({"value_prob": -1.0}, "value_prob"),
        ({"hue_delta": -1}, "hue_delta"),
        ({"saturation_delta": -1}, "saturation_delta"),
        ({"value_delta": -1}, "value_delta"),
    ],
)
def test_ucrt_rejects_invalid_configuration(kwargs, message):
    """Fail clearly instead of silently accepting invalid probabilities or displacement limits."""
    with pytest.raises(ValueError, match=message):
        UnderwaterColorRandomTransfer(**kwargs)


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (np.zeros((8, 8, 3), dtype=np.float32), TypeError, "uint8"),
        (np.zeros((8, 8), dtype=np.uint8), ValueError, "three-channel"),
        (np.zeros((8, 8, 4), dtype=np.uint8), ValueError, "three-channel"),
        ([[[0, 0, 0]]], TypeError, "numpy array"),
    ],
)
def test_ucrt_rejects_unsupported_images(image, error, message):
    """Require the uint8 three-channel BGR contract used by the detection pipeline."""
    with pytest.raises(error, match=message):
        UnderwaterColorRandomTransfer()({"img": image})


def test_training_pipeline_replaces_random_hsv():
    """Place UCRT in the standard training slot without retaining baseline RandomHSV."""

    class StubDataset:
        def __init__(self):
            self.data = {}
            self.use_keypoints = False
            self.buffer = []
            self.cache = None

    transforms = v8_transforms(StubDataset(), imgsz=640, hyp=get_cfg(DEFAULT_CFG)).transforms
    assert sum(isinstance(transform, UnderwaterColorRandomTransfer) for transform in transforms) == 1
    assert not any(isinstance(transform, RandomHSV) for transform in transforms)


def test_non_augmented_dataset_pipeline_excludes_ucrt_and_random_hsv():
    """Keep validation and inference preprocessing free of either stochastic color transform."""
    dataset = YOLODataset.__new__(YOLODataset)
    dataset.augment = False
    dataset.imgsz = 640
    dataset.use_segments = False
    dataset.use_keypoints = False
    dataset.use_obb = False
    transforms = dataset.build_transforms(get_cfg(DEFAULT_CFG)).transforms
    assert not any(isinstance(transform, (UnderwaterColorRandomTransfer, RandomHSV)) for transform in transforms)
