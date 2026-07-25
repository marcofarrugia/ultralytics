# Underwater YOLO11: A Structured Ablation Approach for Improving Object Detection

This repository contains the source-code modifications developed for the Master’s dissertation:

> **Underwater YOLO11: A Structured Ablation Approach for Improving Object Detection**

The research investigates whether targeted modifications to YOLO11 can improve underwater object-detection accuracy while maintaining reasonable real-time inference performance.

## Repository Basis

This repository is a research fork of the [Ultralytics](https://github.com/ultralytics/ultralytics) repository.

Ultralytics provides the underlying YOLO11 training, validation, inference and model-construction framework. This fork contains only the modifications required for the dissertation experiments and is not an independent implementation of YOLO11.

The common experimental baseline is maintained on the `YOLO11-baseline` branch. The [ablation utilities](ablation_code/README.md) include the read-only pretrained-weight transfer audit and its semantic-map examples.

## Ablation Study

Each architectural modification is evaluated independently against a computationally unmodified YOLO11 baseline. Successful modifications may subsequently be combined and evaluated as a final model.

The ablations are organised into the following categories.

### SPPF

- Remove the Spatial Pyramid Pooling–Fast (`SPPF`) block.
- Replace SPPF max pooling with lightweight depthwise atrous spatial pyramid pooling using Hybrid Dilated Convolution.

### Upsampling

- CARAFE.
- DySample.
- Pixel Shuffle with ICNR initialisation.

### Attention

- CBAM.
- CAFM.
- Integration of alternative attention mechanisms into `C2PSA`.

### C2PSA

- Alternative channel projection and branch structures.
- Addition-based and concatenation-based feature fusion.
- Modified convolutional positional encoding.
- Relative positional bias.
- Application of C2PSA at additional feature-pyramid scales.

### Bounding-Box Loss

- CFIoU.
- DAPIoU.

## Evaluation Metrics

Average Precision metrics are calculated using the COCO evaluation protocol through the `faster-coco-eval` library.

The reported metrics are:

- **mAP<sub>50–95</sub>** — the primary accuracy metric, averaged over IoU thresholds from 0.50 to 0.95;
- **AP<sub>50</sub>** — Average Precision at IoU 0.50;
- **AP<sub>75</sub>** — Average Precision at IoU 0.75;
- **AP<sub>S</sub>**, **AP<sub>M</sub>** and **AP<sub>L</sub>** — Average Precision for small, medium and large objects;
- **number of trainable parameters**;
- **GFLOPs**;
- **forward-pass latency and throughput**;
- **end-to-end latency and throughput**, including preprocessing and postprocessing.

The scale-aware metrics are particularly relevant because underwater objects frequently appear at small scales and under low contrast, turbidity and substantial scale variation.

## Experimental Protocol

To support fair comparisons, the ablation experiments use a common:

- dataset split;
- image resolution;
- augmentation policy;
- training schedule;
- optimiser configuration;
- random seed and deterministic configuration;
- scale-matched COCO-pretrained initialisation;
- evaluation procedure.

The main experiments use YOLO11m. Implementation compatibility is also checked for YOLO11n, YOLO11s, YOLO11l and YOLO11x.

## Branch Organisation

Each ablation is developed on a separate branch using the convention:

```text
ablation/<descriptive-name>
```

Examples:

```text
ablation/sppf-remove
ablation/sppf-dw-aspp-hdc
ablation/upsample-carafe
ablation/upsample-dysample
ablation/attention-cbam
ablation/c2psa-cafm
ablation/loss-cfiou
```

Each branch should contain only the changes required for the corresponding ablation.

## Implementation Verification

Before experimental training, each ablation is checked to confirm that:

- the package imports successfully;
- the model constructs successfully;
- forward propagation completes without tensor-shape errors;
- outputs remain compatible with the detection head;
- the implementation works across all YOLO11 model scales;
- parameters are registered correctly;
- pretrained-weight transfer can be audited using the provided diagnostic utility;
- parameter and GFLOP changes are recorded.

These checks verify implementation correctness but do not replace experimental evaluation.

## Training and Evaluation

Training and dataset evaluation are performed through separate Google Colab notebooks maintained outside this repository.

This repository focuses on:

- architecture and loss-function implementation;
- model configuration;
- implementation auditing;
- model-construction and tensor-shape verification.

Datasets, model checkpoints and training outputs are not committed to this repository.

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

The experiments use a fixed underwater object-detection dataset described in the dissertation methodology.

The dataset, annotations and train-validation-test manifests are maintained separately. Access and citation information will be added following formal publication or archival deposit.

## Reproducibility

Each reported experiment should be associated with:

- Git branch and commit SHA;
- model configuration and scale;
- pretrained checkpoint;
- dataset version and split;
- training configuration;
- software and hardware environment;
- random seed;
- evaluation results.

The commit SHA identifies the exact implementation used for an experiment.

## Project Status

This repository is under active development as part of an academic dissertation. Implementations and experimental branches may change until the experiments are completed.

## Citation

Citation information for the dissertation, source code and dataset will be added following submission or publication.

Users of the underlying framework should also cite Ultralytics YOLO and the relevant YOLO literature.

## Licence

This repository is derived from Ultralytics and is distributed under the GNU Affero General Public License v3.0.

The dataset and dissertation document are not included in this repository and are subject to separate terms.

## Acknowledgements

This work builds upon the Ultralytics YOLO framework and research on real-time object detection, including the YOLO11 architecture and subsequent developments in efficient attention-based detection.
