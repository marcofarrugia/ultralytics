"""Audit COCO-pretrained weight transfer into a YOLO detection ablation.

The audit is read-only: it constructs temporary CPU models, runs Ultralytics'
normal loader, and reports parameter coverage. It never re-keys or saves weights.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel, guess_model_scale, load_checkpoint
from ultralytics.utils.torch_utils import intersect_dicts


_LAYER_KEY = re.compile(r"^model\.(\d+)(?:\.|$)")
_FIRST_CONV_KEY = "model.0.conv.weight"


def _layer_index(key: str) -> int | None:
    """Return the top-level YOLO layer index in a state-dictionary key."""
    match = _LAYER_KEY.match(key)
    return int(match.group(1)) if match else None


def _replace_layer_index(key: str, index: int) -> str:
    """Replace the top-level layer index in a state-dictionary key."""
    match = _LAYER_KEY.match(key)
    if not match:
        raise ValueError(f"State key has no YOLO layer index: {key!r}")
    return f"{key[:match.start(1)]}{index}{key[match.end(1):]}"


def _module_type(module: torch.nn.Module) -> str:
    """Return the parser-recorded type used to compare top-level layers."""
    return str(getattr(module, "type", f"{type(module).__module__}.{type(module).__qualname__}"))


def _short_type(module: torch.nn.Module) -> str:
    return _module_type(module).rsplit(".", 1)[-1]


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parameter_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(parameter.shape) for name, parameter in module.named_parameters()}


def _compatible_parameters(source: torch.nn.Module, target: torch.nn.Module) -> tuple[int, int]:
    """Return compatible parameter elements and tensor count for two layers."""
    source_shapes = _parameter_shapes(source)
    elements = tensors = 0
    for name, parameter in target.named_parameters():
        if source_shapes.get(name) == tuple(parameter.shape):
            elements += parameter.numel()
            tensors += 1
    return elements, tensors


def _add_scores(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _infer_layer_map(
    source_layers: list[torch.nn.Module],
    target_layers: list[torch.nn.Module],
) -> tuple[dict[int, int | None], bool]:
    """Infer an order-preserving map and flag distinct equally optimal maps."""
    ns, nt = len(source_layers), len(target_layers)
    zero = (0, 0, 0, 0)
    scores = [[zero for _ in range(nt + 1)] for _ in range(ns + 1)]
    signatures = [[{()} for _ in range(nt + 1)] for _ in range(ns + 1)]
    actions = [["" for _ in range(nt + 1)] for _ in range(ns + 1)]
    for i in range(1, ns + 1):
        actions[i][0] = "skip_source"
    for j in range(1, nt + 1):
        actions[0][j] = "skip_target"

    for i in range(1, ns + 1):
        for j in range(1, nt + 1):
            source_index, target_index = i - 1, j - 1
            candidates = [
                (scores[i - 1][j], signatures[i - 1][j], 0, "skip_source"),
                (scores[i][j - 1], signatures[i][j - 1], 1, "skip_target"),
            ]
            source_layer, target_layer = source_layers[source_index], target_layers[target_index]
            if _module_type(source_layer) == _module_type(target_layer):
                elements, tensors = _compatible_parameters(source_layer, target_layer)
                increment = (1, elements, tensors, -abs(source_index - target_index))
                matched_signatures = {
                    signature + ((target_index, source_index),)
                    for signature in signatures[i - 1][j - 1]
                }
                candidates.append((_add_scores(scores[i - 1][j - 1], increment), matched_signatures, 2, "match"))

            best_score = max(candidate[0] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_score]
            scores[i][j] = best_score
            distinct_signatures = set().union(*(candidate[1] for candidate in best))
            signatures[i][j] = set(sorted(distinct_signatures)[:2])
            actions[i][j] = max(best, key=lambda candidate: candidate[2])[3]

    mapping: dict[int, int | None] = {index: None for index in range(nt)}
    i, j = ns, nt
    while i or j:
        action = actions[i][j]
        if action == "match":
            mapping[j - 1] = i - 1
            i -= 1
            j -= 1
        elif action == "skip_source":
            i -= 1
        elif action == "skip_target":
            j -= 1
        else:
            raise RuntimeError("Could not reconstruct the semantic layer alignment.")
    return mapping, len(signatures[ns][nt]) > 1


def _load_layer_map(
    path: Path,
    source_layers: list[torch.nn.Module],
    target_layers: list[torch.nn.Module],
) -> dict[int, int | None]:
    """Load and validate a complete target-index to source-index JSON map."""
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("The semantic map must be a JSON object.")

    mapping = {int(key): None if value is None else int(value) for key, value in raw.items()}
    expected = set(range(len(target_layers)))
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"The semantic map must cover every target layer; missing={missing}, extra={extra}.")

    source_indices = [index for index in mapping.values() if index is not None]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("A source layer may appear only once in the semantic map.")
    if any(index < 0 or index >= len(source_layers) for index in source_indices):
        raise ValueError("The semantic map contains an out-of-range source layer.")
    for target_index, source_index in mapping.items():
        if (
            source_index is not None
            and _module_type(target_layers[target_index]) != _module_type(source_layers[source_index])
        ):
            raise ValueError(
                f"Mapped layers have different types: target {target_index} "
                f"({_short_type(target_layers[target_index])}) and source {source_index} "
                f"({_short_type(source_layers[source_index])}). Use null for a replacement layer."
            )
    return mapping


def _has_module_prefix(name: str, prefix: str) -> bool:
    """Return whether a module or state key belongs to an exact module prefix."""
    return name == prefix or name.startswith(f"{prefix}.")


def _validate_non_overlapping_prefixes(prefixes: list[str], label: str) -> None:
    """Reject nested prefixes that would make an override ambiguous."""
    for index, left in enumerate(prefixes):
        for right in prefixes[index + 1 :]:
            if _has_module_prefix(left, right) or _has_module_prefix(right, left):
                raise ValueError(
                    f"The submodule map contains overlapping {label} prefixes: "
                    f"{left!r} and {right!r}."
                )


def _load_submodule_map(
    path: Path,
    source: torch.nn.Module,
    target: torch.nn.Module,
) -> dict[str, str]:
    """Load reviewed target-prefix to source-prefix correspondence overrides."""
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("The submodule map must be a JSON object.")

    mapping: dict[str, str] = {}
    for target_prefix, source_prefix in raw.items():
        if (
            not isinstance(target_prefix, str)
            or not target_prefix
            or not isinstance(source_prefix, str)
            or not source_prefix
        ):
            raise ValueError("Submodule-map keys and values must be non-empty strings.")
        mapping[target_prefix] = source_prefix

    source_prefixes = list(mapping.values())
    if len(source_prefixes) != len(set(source_prefixes)):
        raise ValueError("A source submodule may appear only once in the submodule map.")
    _validate_non_overlapping_prefixes(list(mapping), "target")
    _validate_non_overlapping_prefixes(source_prefixes, "source")

    source_modules = dict(source.named_modules())
    target_modules = dict(target.named_modules())
    for target_prefix, source_prefix in mapping.items():
        if target_prefix not in target_modules:
            raise ValueError(f"Target submodule does not exist: {target_prefix!r}.")
        if source_prefix not in source_modules:
            raise ValueError(f"Source submodule does not exist: {source_prefix!r}.")
        if _module_type(target_modules[target_prefix]) != _module_type(source_modules[source_prefix]):
            raise ValueError(
                f"Mapped submodules have different types: target {target_prefix!r} "
                f"({_short_type(target_modules[target_prefix])}) and source {source_prefix!r} "
                f"({_short_type(source_modules[source_prefix])})."
            )
    return mapping


def _intended_source_key(
    target_key: str,
    layer_map: dict[int, int | None],
    submodule_map: dict[str, str],
) -> str | None:
    """Resolve a target key using submodule overrides before layer correspondence."""
    for target_prefix, source_prefix in submodule_map.items():
        if _has_module_prefix(target_key, target_prefix):
            return f"{source_prefix}{target_key[len(target_prefix):]}"

    target_index = _layer_index(target_key)
    source_index = layer_map.get(target_index) if target_index is not None else None
    return _replace_layer_index(target_key, source_index) if source_index is not None else None


def _partial_first_conv_elements(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    exact_keys: set[str],
) -> int:
    """Count values copied by BaseModel.load's first-convolution exception."""
    if _FIRST_CONV_KEY in exact_keys or _FIRST_CONV_KEY not in source_state or _FIRST_CONV_KEY not in target_state:
        return 0
    source, target = source_state[_FIRST_CONV_KEY], target_state[_FIRST_CONV_KEY]
    if source.ndim != 4 or target.ndim != 4 or source.shape[2:] != target.shape[2:]:
        return 0
    return (
        min(source.shape[0], target.shape[0])
        * min(source.shape[1], target.shape[1])
        * target.shape[2]
        * target.shape[3]
    )


def _verify_actual_load(
    source: DetectionModel,
    target: DetectionModel,
    before: dict[str, torch.Tensor],
    exact_keys: set[str],
    partial_elements: int,
) -> None:
    """Run the real loader and ensure its effects match the audit model."""
    source_state = _state_copy(source)
    target.load(source, verbose=False)
    after = _state_copy(target)
    failed = [key for key in exact_keys if not torch.equal(after[key], source_state[key])]
    if failed:
        raise RuntimeError(f"Ultralytics did not copy expected exact tensors: {failed[:3]}")

    allowed = set(exact_keys)
    if partial_elements:
        allowed.add(_FIRST_CONV_KEY)
        source_conv, target_conv = source_state[_FIRST_CONV_KEY], after[_FIRST_CONV_KEY]
        c1, c2 = min(source_conv.shape[0], target_conv.shape[0]), min(source_conv.shape[1], target_conv.shape[1])
        if not torch.equal(target_conv[:c1, :c2], source_conv[:c1, :c2]):
            raise RuntimeError("Ultralytics did not perform the expected partial first-convolution copy.")
    unexpected = [key for key in after if key not in allowed and not torch.equal(after[key], before[key])]
    if unexpected:
        raise RuntimeError(f"Ultralytics changed unexpected tensors: {unexpected[:3]}")


def _audit_transfer(
    source: DetectionModel,
    target: DetectionModel,
    layer_map: dict[int, int | None],
    submodule_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure actual and semantic transfer from source into target."""
    submodule_map = submodule_map or {}
    source_state, target_state = _state_copy(source), _state_copy(target)
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    target_buffers = dict(target.named_buffers())
    total = sum(parameter.numel() for parameter in target_parameters.values())
    if not total:
        raise ValueError("The target model has no parameters.")

    exact_keys = set(intersect_dicts(source_state, target_state))
    exact_parameter_keys = exact_keys.intersection(target_parameters)
    partial = _partial_first_conv_elements(source_state, target_state, exact_keys)
    actual = sum(target_parameters[key].numel() for key in exact_parameter_keys) + partial

    semantic_source: dict[str, str] = {}
    semantic_full: dict[str, str] = {}
    shape_mismatches: list[dict[str, Any]] = []
    new_keys: list[str] = []
    for target_key, target_parameter in target_parameters.items():
        source_key = _intended_source_key(target_key, layer_map, submodule_map)
        if source_key is None:
            new_keys.append(target_key)
            continue
        semantic_source[target_key] = source_key
        source_parameter = source_parameters.get(source_key)
        if source_parameter is None:
            new_keys.append(target_key)
        elif source_parameter.shape == target_parameter.shape:
            semantic_full[target_key] = source_key
        else:
            shape_mismatches.append(
                {
                    "target_key": target_key,
                    "source_key": source_key,
                    "target_shape": tuple(target_parameter.shape),
                    "source_shape": tuple(source_parameter.shape),
                    "elements": target_parameter.numel(),
                }
            )

    semantic_elements = sum(target_parameters[key].numel() for key in semantic_full)
    if partial and semantic_source.get(_FIRST_CONV_KEY) == _FIRST_CONV_KEY:
        semantic_elements += partial
    correct_keys = {key for key in exact_parameter_keys if semantic_full.get(key) == key}
    correct = sum(target_parameters[key].numel() for key in correct_keys)
    if partial and semantic_source.get(_FIRST_CONV_KEY) == _FIRST_CONV_KEY:
        correct += partial
    missed_keys = {key for key, source_key in semantic_full.items() if source_key != key}
    missed = sum(target_parameters[key].numel() for key in missed_keys)

    collision_state_keys: dict[str, str | None] = {}
    for key in exact_keys:
        intended = _intended_source_key(key, layer_map, submodule_map)
        if intended != key:
            collision_state_keys[key] = intended

    groups: dict[tuple[int, int | None], dict[str, Any]] = {}
    for key, intended in sorted(collision_state_keys.items()):
        target_index = _layer_index(key)
        if target_index is None:
            continue
        semantic_index = _layer_index(intended) if intended is not None else None
        group = groups.setdefault(
            (target_index, semantic_index),
            {
                "target_layer": target_index,
                "loaded_source_layer": target_index,
                "semantic_source_layer": semantic_index,
                "parameter_keys": [],
                "buffer_keys": [],
                "correspondence_keys": {},
            },
        )
        group["correspondence_keys"][key] = intended
        if key in target_parameters:
            group["parameter_keys"].append(key)
        elif key in target_buffers:
            group["buffer_keys"].append(key)

    _verify_actual_load(source, target, target_state, exact_keys, partial)
    return {
        "total": total,
        "actual": actual,
        "correct": correct,
        "semantic": semantic_elements,
        "missed": missed,
        "partial": partial,
        "shape_mismatches": shape_mismatches,
        "new_keys": new_keys,
        "collisions": list(groups.values()),
        "collision_parameter_count": sum(len(group["parameter_keys"]) for group in groups.values()),
    }


def _pct(elements: int, total: int) -> float:
    return 100.0 * elements / total if total else 0.0


def _identity_baseline_valid(
    source: DetectionModel,
    baseline: DetectionModel,
    report: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate that baseline non-transfer is solely dataset-dependent."""
    issues: list[str] = []
    source_layers, baseline_layers = list(source.model), list(baseline.model)
    if len(source_layers) != len(baseline_layers):
        issues.append("source and baseline layer counts differ")
    for index, (source_layer, baseline_layer) in enumerate(zip(source_layers, baseline_layers)):
        if _module_type(source_layer) != _module_type(baseline_layer):
            issues.append(f"layer {index} changes type")

    detect_index = len(baseline_layers) - 1
    if not baseline_layers or not isinstance(baseline_layers[-1], Detect):
        issues.append("the baseline does not end in a Detect layer")
    allowed_shape_layers = {detect_index}
    for mismatch in report["shape_mismatches"]:
        index = _layer_index(mismatch["target_key"])
        if index not in allowed_shape_layers and mismatch["target_key"] != _FIRST_CONV_KEY:
            issues.append(f"unexpected baseline shape mismatch: {mismatch['target_key']}")
    if report["new_keys"]:
        issues.append(f"baseline has {len(report['new_keys'])} parameters without source counterparts")
    if report["collision_parameter_count"]:
        issues.append("baseline contains semantic key collisions")
    return not issues, issues


def _mapping_ranges(mapping: dict[int, int | None]) -> list[tuple[int, int, int, int]]:
    """Compress non-identity consecutive mappings for readable output."""
    pairs = [(target, source) for target, source in sorted(mapping.items()) if source is not None and target != source]
    if not pairs:
        return []
    ranges: list[tuple[int, int, int, int]] = []
    ts = te = pairs[0][0]
    ss = se = pairs[0][1]
    for target, source in pairs[1:]:
        if target == te + 1 and source == se + 1:
            te, se = target, source
        else:
            ranges.append((ts, te, ss, se))
            ts = te = target
            ss = se = source
    ranges.append((ts, te, ss, se))
    return ranges


def _format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _metric_line(label: str, elements: int, total: int) -> str:
    """Format an element count and percentage as one aligned table row."""
    return f"{label:<34}{elements:>14,d} ({_pct(elements, total):.3f}%)"


def _print_report(
    model_path: Path,
    weights_path: Path,
    nc: int,
    channels: int,
    source: DetectionModel,
    target: DetectionModel,
    mapping: dict[int, int | None],
    submodule_mapping: dict[str, str],
    ambiguous: bool,
    baseline_valid: bool,
    baseline_issues: list[str],
    baseline: dict[str, Any],
    ablation: dict[str, Any],
    baseline_parameters: dict[str, torch.nn.Parameter],
) -> None:
    """Print only information useful for deciding whether to redesign."""
    baseline_pct = _pct(baseline["actual"], baseline["total"])
    actual_pct = _pct(ablation["actual"], ablation["total"])
    expected_shape_loss = baseline["total"] - baseline["actual"]

    induced_mismatches = []
    for mismatch in ablation["shape_mismatches"]:
        baseline_parameter = baseline_parameters.get(mismatch["source_key"])
        if baseline_parameter is None or tuple(baseline_parameter.shape) != mismatch["target_shape"]:
            induced_mismatches.append(mismatch)
    induced_shape_elements = sum(item["elements"] for item in induced_mismatches)
    new_elements = sum(dict(target.named_parameters())[key].numel() for key in ablation["new_keys"])

    print("\nYOLO11 pretrained-transfer audit")
    print("=" * 35)
    print(f"Checkpoint:       {weights_path}")
    print(f"Ablation YAML:    {model_path}")
    print(f"Dataset classes:  {nc}")
    print(f"Input channels:   {channels}")
    print(f"{'Baseline validation:':<34}{'VALID' if baseline_valid else 'WARNING':>14}")
    print(_metric_line("Baseline reference transfer:", baseline["actual"], baseline["total"]))
    if expected_shape_loss:
        print(_metric_line("Expected dataset-specific loss:", expected_shape_loss, baseline["total"]))
    print()
    print(_metric_line("Actual ablation transfer:", ablation["actual"], ablation["total"]))
    print(f"{'Additional loss vs baseline:':<34}{baseline_pct - actual_pct:>14.3f} percentage points")

    if ambiguous:
        print(f"{'Semantic alignment:':<34}{'AMBIGUOUS':>14}")
        print(f"{'Semantic metrics:':<34}{'not reported; supply --semantic-map':>14}")
    else:
        print(_metric_line("Semantically correct transfer:", ablation["correct"], ablation["total"]))
        print(_metric_line("Semantic compatibility:", ablation["semantic"], ablation["total"]))
        print(_metric_line("Missed because keys moved:", ablation["missed"], ablation["total"]))
        print(_metric_line("Ablation-induced shape loss:", induced_shape_elements, ablation["total"]))
        print(_metric_line("New/unmatched parameters:", new_elements, ablation["total"]))
        print(f"{'Suspicious exact collisions:':<34}{ablation['collision_parameter_count']:>14,d}")
    if ablation["partial"]:
        print(_metric_line("Partial first-conv values:", ablation["partial"], ablation["total"]))

    if not baseline_valid:
        print("\nBaseline warning")
        for issue in baseline_issues:
            print(f"  - {issue}")
        print("  The ablation comparison may be unreliable.")

    ranges = _mapping_ranges(mapping) if not ambiguous else []
    mapped_source = {index for index in mapping.values() if index is not None}
    unmatched_source = sorted(set(range(len(source.model))) - mapped_source) if not ambiguous else []
    unmatched_target = [index for index, source_index in mapping.items() if source_index is None]
    if ambiguous:
        unmatched_target = []
    if ranges or unmatched_source or unmatched_target:
        print("\nLayer correspondence changes")
        for ts, te, ss, se in ranges:
            print(f"  target {_format_range(ts, te)} -> source {_format_range(ss, se)}")
        for index in unmatched_source:
            print(f"  source {index} ({_short_type(source.model[index])}): unmatched")
        for index in unmatched_target:
            print(f"  target {index} ({_short_type(target.model[index])}): new or replaced")

    if submodule_mapping:
        print("\nSubmodule correspondence overrides")
        for target_prefix, source_prefix in sorted(submodule_mapping.items()):
            print(f"  target {target_prefix} -> source {source_prefix}")

    if induced_mismatches and not ambiguous:
        print("\nAblation-induced shape changes")
        for index in sorted({_layer_index(item["target_key"]) for item in induced_mismatches}):
            count = sum(1 for item in induced_mismatches if _layer_index(item["target_key"]) == index)
            print(f"  target layer {index} ({_short_type(target.model[index])}): {count} parameter tensors")

    if ablation["collisions"] and not ambiguous:
        print("\nSuspicious collisions")
        for collision in ablation["collisions"]:
            target_index = collision["target_layer"]
            loaded_type = _short_type(source.model[target_index]) if target_index < len(source.model) else "missing"
            for key in collision["parameter_keys"]:
                correspondence_key = collision["correspondence_keys"][key]
                semantic_index = _layer_index(correspondence_key) if correspondence_key is not None else None
                print(f"  {key}")
                print(f"    loader source:  layer {target_index} {loaded_type}")
                print(f"    target:         layer {target_index} {_short_type(target.model[target_index])}")
                if correspondence_key is None:
                    print("    correspondence: no source layer")
                elif semantic_index is None:
                    print(f"    correspondence: {correspondence_key}")
                else:
                    print(
                        f"    correspondence: source layer {semantic_index} "
                        f"{_short_type(source.model[semantic_index])} ({correspondence_key})"
                    )

    print("\nDiagnosis")
    if ambiguous:
        print("  Semantic correspondence is ambiguous; review or supply an explicit map before deciding.")
    elif ablation["collision_parameter_count"]:
        print("  Standard loading transfers parameters between different semantic components.")
    if not ambiguous and ablation["missed"]:
        print("  Semantically compatible parameters are missed because their layer indices moved.")
    if (
        not ambiguous
        and not ablation["collision_parameter_count"]
        and not ablation["missed"]
        and not induced_mismatches
    ):
        print("  No architecture-induced transfer problem was identified.")


def audit(
    model_path: Path,
    weights_path: Path,
    data_path: Path,
    semantic_map_path: Path | None = None,
    submodule_map_path: Path | None = None,
) -> None:
    """Build source, baseline, and ablation models and print the audit."""
    # Scaled standard names such as yolo11n.yaml can resolve to the shared
    # yolo11.yaml through yaml_model_load, so model_path need not itself exist.
    for label, path in (("checkpoint", weights_path), ("dataset YAML", data_path)):
        if not path.is_file():
            raise FileNotFoundError(f"The {label} does not exist: {path}")
    if semantic_map_path is not None and not semantic_map_path.is_file():
        raise FileNotFoundError(f"The semantic map does not exist: {semantic_map_path}")
    if submodule_map_path is not None and not submodule_map_path.is_file():
        raise FileNotFoundError(f"The submodule map does not exist: {submodule_map_path}")

    data = check_det_dataset(str(data_path), autodownload=False)
    nc, channels = int(data["nc"]), int(data.get("channels", 3))
    source, _ = load_checkpoint(str(weights_path), device="cpu")
    if not isinstance(source, DetectionModel):
        raise TypeError(f"The checkpoint contains {type(source).__name__}, not a YOLO DetectionModel.")
    source = source.float().cpu()

    source_scale = str(source.yaml.get("scale", ""))
    target_scale = guess_model_scale(model_path)
    if source_scale and target_scale != source_scale:
        raise ValueError(
            f"Model scale mismatch: checkpoint scale={source_scale!r}, YAML scale={target_scale or '<missing>'!r}. "
            "Include the scale in the ablation filename, for example yolo11m_example.yaml."
        )

    baseline_model = DetectionModel(deepcopy(source.yaml), ch=channels, nc=nc, verbose=False).float().cpu()
    target_model = DetectionModel(str(model_path), ch=channels, nc=nc, verbose=False).float().cpu()
    baseline_map = {index: index for index in range(len(baseline_model.model))}
    baseline_report = _audit_transfer(source, baseline_model, baseline_map)
    baseline_valid, baseline_issues = _identity_baseline_valid(source, baseline_model, baseline_report)

    source_layers, target_layers = list(source.model), list(target_model.model)
    if semantic_map_path is None:
        mapping, ambiguous = _infer_layer_map(source_layers, target_layers)
    else:
        mapping = _load_layer_map(semantic_map_path, source_layers, target_layers)
        ambiguous = False
    submodule_mapping = (
        _load_submodule_map(submodule_map_path, source, target_model) if submodule_map_path is not None else {}
    )
    ablation_report = _audit_transfer(source, target_model, mapping, submodule_mapping)
    _print_report(
        model_path,
        weights_path,
        nc,
        channels,
        source,
        target_model,
        mapping,
        submodule_mapping,
        ambiguous,
        baseline_valid,
        baseline_issues,
        baseline_report,
        ablation_report,
        dict(baseline_model.named_parameters()),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit COCO-pretrained transfer into one YOLO detection ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python audit_pretrained_transfer.py --model yolo11m_sppf_removed.yaml "
            "--weights yolo11m.pt --data C:\\Datasets\\UOD_Dataset\\data_split.yaml\n"
            "  python audit_pretrained_transfer.py --model yolo11m_reordered.yaml "
            "--weights yolo11m.pt --data C:\\Datasets\\UOD_Dataset\\data_split.yaml "
            "--semantic-map yolo11m_reordered_map.json"
        ),
    )
    parser.add_argument("--model", type=Path, required=True, help="Ablation model YAML.")
    parser.add_argument("--weights", type=Path, required=True, help="Local pretrained checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Target dataset YAML.")
    parser.add_argument(
        "--semantic-map",
        type=Path,
        help="Optional complete JSON mapping of target layer indices to source indices or null.",
    )
    parser.add_argument(
        "--submodule-map",
        type=Path,
        help="Optional JSON mapping of reviewed target submodule prefixes to source submodule prefixes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit(args.model, args.weights, args.data, args.semantic_map, args.submodule_map)


if __name__ == "__main__":
    main()
