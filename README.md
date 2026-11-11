# Underwater YOLO11: A Structured Ablation Approach for Improving Object Detection

This repository contains the implementations and supplementary materials for the Master's dissertation. The study assessed how targeted modifications to YOLO11m affect underwater object-detection accuracy and computational cost.

## Repository Basis

This repository is a research fork based on [Ultralytics YOLO v8.3.244](https://github.com/ultralytics/ultralytics/releases/tag/v8.3.244).

Ultralytics provides the underlying YOLO11 training, validation, inference and model-construction framework. The common experimental baseline is maintained on the `YOLO11-baseline` branch.

## Ablation Study

The main study compared 19 seed-0 YOLO11m variants individually with the COCO-pretrained YOLO11m baseline (B0) on the validation set. The four loss variants used the baseline architecture and replaced its CIoU bounding-box regression term.

| Category | ID | Evaluated variant |
|---|---|---|
| SPPF | C1 | SPPF removal using an identity layer |
| SPPF | C2 | DW-HDC-SPPF |
| Detection scale | D1 | Terminal P2 detection with corrected pretrained-weight transfer |
| Upsampling | U1 | CARAFE |
| Upsampling | U2 | DySample+ |
| Upsampling | U3 | Depthwise Pixel Shuffle with ICNR |
| Attention | A1 | CBAM before Detect (P3–P5) |
| Attention | A2 | C2PSA attention replacement with CAFM |
| Attention | A3 | C2PSA with parallel CAFM |
| Attention | A4 | C2PSA full-width concat |
| Attention | A5 | C2PSA full-width non-concat |
| Attention | A6 | C2PSA full-width residual fusion |
| Attention | A7 | C2PSA with iRPE-K |
| Attention | A8 | C3k2 fusion-bottleneck CrossPSA |
| Attention | A9 | Post-fusion PSA |
| Bounding-box regression loss | L1 | CFIoU |
| Bounding-box regression loss | L2 | InterpIoU |
| Bounding-box regression loss | L3 | Dynamic InterpIoU |
| Bounding-box regression loss | L4 | PIoU-v2 |

## Evaluation Metrics

The primary accuracy metric was **COCO AP@[.50:.95]**, computed with `faster-coco-eval` and averaged over IoU thresholds from 0.50 to 0.95. The independent COCO results also include AP<sub>50</sub>, AP<sub>75</sub> and AP for small, medium and large objects. **Native Ultralytics mAP<sub>50–95</sub>** was reported separately and used for checkpoint selection.

Computational cost was measured using parameter counts, estimated GFLOPs and per-image forward-pass and pipeline times from batched validator runs. The timed YOLO11m pipeline covers device transfer, precision conversion and normalisation, the forward pass, and non-maximum suppression. Throughput estimates were derived from the corresponding per-image times.

## Experimental Protocol

The seed-0 YOLO11m comparisons used the same dataset splits, 640 × 640 inputs, a training batch size of 64 and a common augmentation policy. Models were initialised with compatible weights from the official `yolo11m.pt` checkpoint. The dissertation reports that all training runs completed the full 300 epochs with early stopping disabled.

The runs requested deterministic execution (`deterministic=True`). The DySample+ (U2) training log flags a nondeterministic CUDA grid-sampling operation in the backward pass.

## Branch Organisation

Ablation implementations use the `ablation/<descriptive-name>` branch convention.

| Variant | Implementation branch |
|---|---|
| C2 | `ablation/sppf-dw-hdc-sppf` |
| A1 | `ablation/attention-cbam-detect-p3-p5` |
| A4 and A5 | `ablation/c2psa-full-feature-fusion` |

A4 and A5 select distinct model YAMLs on their shared branch. The supplement maps every variant to its implementation branch, selected YAML and executed notebook. [Training and validation notebook index](dissertation/README.md#training-and-validation-notebook-index)

## Training and Evaluation

The supplement contains the executed Google Colab notebooks for training, validation and held-out evaluation. Reproducing training and model-specific evaluation requires the corresponding dataset images, YOLO labels and checkpoints. The common held-out analysis uses the supplied COCO annotations and saved predictions. [Dissertation supplement](dissertation/README.md)

### Validation and Held-Out Results

On validation, four variants exceeded B0 in overall independent COCO AP. All 19 had lower small-object COCO AP and scored at or below B0 in native Ultralytics mAP<sub>50–95</sub>. The validation-only ROC-weighted VIKOR ranking placed A1 first and C2 second. [Validation ranking workbook](dissertation/analysis/yolo11m_ablation_vikor_analysis.xlsx)

The held-out panel used the frozen seed-0 checkpoints for B0, A1 and C2, with YOLO26m as a separate framework comparator. Independent COCO AP@[.50:.95] was 29.32% for B0, 30.42% for A1 (+1.10 percentage points) and 30.74% for C2 (+1.43 percentage points). The 95% paired cluster-bootstrap intervals for both overall AP differences lay above zero. The corresponding intervals for small-object AP included zero. [Held-out point estimates](dissertation/common-test-analysis/results/common_test_point_estimates.csv) · [Bootstrap summary](dissertation/common-test-analysis/results/paired_cluster_bootstrap_summary.csv)

## Installation

Clone the repository and select the required branch:

```bash
git clone https://github.com/marcofarrugia/ultralytics.git
cd ultralytics
git checkout ablation/<descriptive-name>
```

Create a Python environment and install the package in editable mode:

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e .
```

Activate the environment before installation where required.

GPU-enabled PyTorch should be installed using the version appropriate for the available CUDA environment.

## Dataset

The curated underwater dataset contains 16,626 images annotated for 14 classes. The fixed, group-aware split comprises 13,301 training images, 1,663 validation images and 1,662 test images. Images identified as related were assigned to the same split.

Dataset images and per-image YOLO labels are maintained separately. The supplement includes the dataset configuration, split manifests, group assignments and shared COCO test annotations. [Dataset metadata](dissertation/README.md#dataset-metadata)

## Reproducibility

Each reported experiment should be associated with:

- Git branch;
- model configuration and scale;
- pretrained checkpoint;
- dataset version and split;
- training configuration;
- software and hardware environment;
- random seed;
- evaluation results.

## Citation

If you use the architectural or bounding-box regression loss ablation implementations in this repository, please cite **Underwater YOLO11 Ablations** and the upstream **Ultralytics YOLO v8.3.244** release.

Citation metadata for the fork and its upstream foundation is provided in [CITATION.cff](CITATION.cff).

### Dissertation ablation implementations

```bibtex
@software{FarrugiaYOLO11Ablations,
  author = {Farrugia, Marco},
  title  = {{Underwater YOLO11 Ablations}},
  url    = {https://github.com/marcofarrugia/ultralytics},
  note   = {Custom fork of Ultralytics YOLO v8.3.244 containing
            architectural and bounding-box regression loss
            ablation implementations for underwater object detection}
}
```
### Upstream framework

```bibtex
@software{Ultralytics_v8_3_244,
  author  = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  title   = {{Ultralytics YOLO}},
  version = {8.3.244},
  url     = {https://github.com/ultralytics/ultralytics/releases/tag/v8.3.244}
}
```
When reporting an experiment, identify the ablation branch and the exact source revision where available, together with the model and training configuration. A repository citation identifies the software project; the experiment-specific records identify the implementation used.

## Licence

This repository is derived from Ultralytics and is distributed under the GNU Affero General Public License v3.0.

The dataset and dissertation document are not included in this repository and are subject to separate terms.

## Acknowledgements

This work builds upon the Ultralytics YOLO framework and research on real-time object detection, including the YOLO11 architecture and subsequent developments in efficient attention-based detection.
