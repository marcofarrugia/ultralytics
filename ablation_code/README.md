# Ablation Utilities

## Pretrained-weight transfer audit

`audit_pretrained_transfer.py` is a read-only diagnostic utility for examining how Ultralytics loads pretrained YOLO11 detection weights into an architectural ablation.

The utility distinguishes ordinary key-and-shape transfer from transfer that agrees with the inferred or explicitly supplied layer correspondence. This helps identify weight loss caused by shifted layer indices, changed parameter shapes, replacement modules, or misleading same-key matches.

All models are constructed temporarily on the CPU, and results are printed to the terminal without any file modification.

## Requirements

Run the utility using the Python environment associated with this repository. It relies on internal APIs from this current Ultralytics fork and is not guaranteed to work with other Ultralytics versions.

The script requires:

- Python 3.10 or later;
- the dependencies declared by this repository;
- a local pretrained YOLO detection checkpoint;
- the ablation model YAML;
- the target dataset YAML.

No training or GPU is required. The checkpoint and dataset configuration must already be available locally.

## Usage

Run the command from the repository root:

```text
python "ablation_code/audit_pretrained_transfer.py" --model "path/to/yolo11m-ablation.yaml" --weights "path/to/yolo11m.pt" --data "path/to/data.yaml"
```

The scale encoded in the model filename must match the checkpoint scale. For example, a `yolo11m.pt` checkpoint requires a model filename containing the `m` scale, such as `yolo11m-ablation.yaml`.

### Arguments

| Argument | Description |
|---|---|
| `--model` | Model YAML for the ablation being audited. |
| `--weights` | Local pretrained YOLO detection checkpoint. |
| `--data` | Target dataset YAML used to obtain the number of classes and input channels. |
| `--semantic-map` | Optional JSON file defining the correspondence from every target layer to its source layer. |
| `--submodule-map` | Optional JSON file defining reviewed target-to-source submodule-prefix correspondence. |

## How the audit works

The utility:

1. loads the pretrained source model on the CPU;
2. reconstructs an unmodified dataset-specific baseline using the source architecture;
3. constructs the requested ablation using the same dataset class count and input channels;
4. runs the actual Ultralytics model-loading operation on the temporary target models;
5. verifies which tensors were copied by the loader;
6. compares actual loading with the inferred or supplied semantic layer correspondence;
7. applies any explicitly reviewed submodule correspondence before the top-level layer correspondence.

By default, the tool infers an order-preserving correspondence between top-level layers with matching module types. If more than one equally suitable correspondence exists, it reports the alignment as `AMBIGUOUS` and does not report semantic metrics.

## Interpreting the output

| Output | Meaning |
|---|---|
| `Baseline validation` | Whether the reconstructed baseline differs from the pretrained source only in expected dataset-dependent areas, such as the detection head. |
| `Baseline reference transfer` | Parameter elements transferred into the unmodified dataset-specific baseline. |
| `Expected dataset-specific loss` | Baseline non-transfer caused by expected dataset-dependent shape changes. |
| `Actual ablation transfer` | Parameter elements copied into the ablation by the normal Ultralytics loader. |
| `Additional loss vs baseline` | Reduction in actual transfer relative to the baseline reference, expressed in percentage points. |
| `Semantically correct transfer` | Actually transferred parameter elements that also agree with the inferred or supplied layer correspondence. |
| `Semantic compatibility` | Parameter elements whose semantic counterparts have compatible names and shapes, whether or not the normal loader copies them. |
| `Missed because keys moved` | Semantically compatible parameter elements not copied because their top-level layer indices changed. |
| `Ablation-induced shape loss` | Non-transfer caused by parameter-shape changes introduced by the ablation. |
| `New/unmatched parameters` | Parameters belonging to new or replaced components without a compatible source counterpart. |
| `Suspicious exact collisions` | Same-key and same-shape transfers where the loader source does not agree with the semantic correspondence. |
| `Partial first-conv values` | Values copied through Ultralytics' partial first-convolution loading exception. |

Transfer rows report both the parameter-element count and its percentage of the corresponding target model.

A high transfer percentage alone does not prove that the transferred weights belong to the correct semantic components. In particular, inspect any reported collisions and correspondence changes.

## Explicit semantic maps

Use `--semantic-map` only when the automatic correspondence is ambiguous or when the architecture requires an explicitly reviewed mapping:

```text
python "ablation_code/audit_pretrained_transfer.py" --model "path/to/yolo11m-ablation.yaml" --weights "path/to/yolo11m.pt" --data "path/to/data.yaml" --semantic-map "path/to/layer-map.json"
```

The JSON object must map every top-level target-layer index to either:

- the corresponding source-layer index; or
- `null` when the target layer is new or replaces a different module.

For illustration, a complete map for a hypothetical three-layer target could be:

```json
{
  "0": 0,
  "1": null,
  "2": 1
}
```

Every target layer must be present, each non-null source layer may be used only once, and mapped source and target layers must have matching module types.

## Explicit submodule maps

Use `--submodule-map` when a replacement layer contains individually reviewed submodules that correspond to parts of the source layer:

```text
python "ablation_code/audit_pretrained_transfer.py" --model "path/to/yolo11m-ablation.yaml" --weights "path/to/yolo11m.pt" --data "path/to/data.yaml" --semantic-map "path/to/layer-map.json" --submodule-map "path/to/submodule-map.json"
```

For example, a replacement at layer 9 may retain compatible `cv1` and `cv2` bookend projections while introducing a new internal operation:

```json
{
  "model.9.cv1": "model.9.cv1",
  "model.9.cv2": "model.9.cv2"
}
```

Submodule mappings take precedence over the enclosing layer mapping. Therefore, layer 9 can remain mapped to `null`, while the two explicitly mapped submodules receive correspondence and every unmapped child remains new.

Both prefixes must exist in `named_modules()`, mapped submodules must have matching types, prefixes may not overlap, and a source submodule may be used only once. Parameter-name or shape differences remain visible in the normal audit results.

An explicit mapping records a reviewed correspondence claim. Matching module types and tensor shapes alone do not prove conceptual equivalence, particularly when a fusion layer receives features produced by a changed internal operation.

### Included SPPF examples

Two SPPF-removal semantic-map examples are included:

- `yolo11m_sppf_identity_map_example.json` describes the index-preserving design used by `ablation/sppf-remove`. Target layer 9 is the replacement `nn.Identity` layer and therefore maps to `null`; all downstream layer indices remain unchanged.
- `yolo11m_sppf_removed_map_example.json` illustrates a different design in which source SPPF layer 9 is physically deleted. Consequently, target layers 9--22 correspond to source layers 10--23.

The current `ablation/sppf-remove` implementation uses the first, index-preserving design. The direct-removal map is included only to illustrate how an explicit map represents shifted layer indices.

## Expected diagnostic messages

Two `Overriding model.yaml nc=...` messages may be printed. This is expected because the utility constructs both the dataset-specific baseline and the ablation using the target dataset class count.

If `Semantic alignment: AMBIGUOUS` is reported, review the architecture and provide an explicit semantic map before interpreting semantic transfer.

If `Baseline validation: WARNING` is reported, investigate the listed baseline issues before relying on the ablation comparison.

## Limitations

This utility evaluates pretrained-weight transfer only. It does not establish:

- model accuracy or convergence;
- whether the ablation improves detection performance;
- inference latency, memory use, parameters, or FLOPs;
- export compatibility;
- the conceptual correctness of an automatically inferred correspondence.

It does not remap parameter keys, force incompatible parameters into the target model, save modified weights, or approve an ablation automatically.

## Reproducibility

When reporting an audit, record:

- the repository commit or submission tag;
- the ablation model YAML;
- the checkpoint name and source;
- the dataset YAML and class count;
- whether an explicit semantic map was used;
- whether an explicit submodule map was used;
- the complete command and terminal output.

Checkpoints, datasets, and local experiment artefacts are not distributed with this utility.
