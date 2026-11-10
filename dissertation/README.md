# Dissertation supplement

This folder contains the supplementary materials for the dissertation *Underwater YOLO11: A Structured Ablation Approach for Improving Object Detection*. Paths below are relative to this folder.

## Contents

| Folder | Contents |
|---|---|
| notebooks/ablations/ | Nineteen seed-0 YOLO11m ablation records |
| notebooks/baselines/ | Pretrained and scratch YOLO11m records; YOLO26m comparator |
| notebooks/test/ | Four model-specific held-out evaluation records |
| common-test-analysis/ | Common analysis notebook, shared COCO test ground truth and splits.csv |
| common-test-analysis/predictions/ | Aligned predictions in B0, A1, C2 and YOLO26m subfolders |
| common-test-analysis/results/ | Seven saved CSV outputs, including completed bootstrap draws |
| dataset/ | Dataset configuration and train/validation/test manifests |
| analysis/ | yolo11m_ablation_vikor_analysis.xlsx workbook |

Images, per-image YOLO label files and model checkpoints are not included. The source training and model-specific evaluation notebooks retain their Colab setup and original locations; running those notebooks requires the corresponding data, checkpoints and environment. The common test analysis inputs can be used without images or checkpoints.

## Training and validation notebook index

Files are under notebooks/ablations/. The selected YAML names below are the scale-specific names passed to Ultralytics. They may resolve to a shared YAML without the scale letter under ultralytics/cfg/models/11/ on the named branch.

| ID | Notebook filename | Implementation branch | Selected model YAML |
|---|---|---|---|
| C1 | YOLO11m_ablation_SPPF-remove_v4.ipynb | ablation/sppf-remove | yolo11m-sppf-remove.yaml |
| C2 | YOLO11m_ablation_DW-HDC-SPPF_v4.ipynb | ablation/sppf-dw-hdc-sppf | yolo11m-dw-hdc-sppf.yaml |
| D1 | YOLO11m_ablation_detect-p2-transfer-corrected_v4.ipynb | ablation/detect-p2 | yolo11m-p2.yaml |
| U1 | YOLO11m_ablation_CARAFE_v4.ipynb | ablation/upsample-carafe | yolo11m-carafe.yaml |
| U2 | YOLO11m_ablation_DySamplePlus_v4.ipynb | ablation/upsample-dysample | yolo11m-dysample-plus.yaml |
| U3 | YOLO11m_ablation_DW-Pixel Shuffle-ICNR_v4.ipynb | ablation/upsample-pixelshuffle-icnr | yolo11m-pixelshuffle-icnr.yaml |
| A1 | YOLO11m_ablation_CBAM_v4.ipynb | ablation/attention-cbam-detect-p3-p5 | yolo11m-cbam-detect-p3-p5.yaml |
| A2 | YOLO11m_ablation_C2PSA-CAFM_v4.ipynb | ablation/c2psa-cafm | yolo11m-c2psa-cafm.yaml |
| A3 | YOLO11m_ablation_C2PSA_ParallelCAFM_v4.ipynb | ablation/c2psa-parallel-cafm | yolo11m-c2psa-parallel-cafm.yaml |
| A4 | YOLO11m_ablation_Full-Feature-Fusion_concat_C2PSA_v4.ipynb | ablation/c2psa-full-feature-fusion | yolo11m-c2psa-full-feature-concat.yaml |
| A5 | YOLO11m_ablation_Full-Feature-Fusion_non-concat_C2PSA_v4.ipynb | ablation/c2psa-full-feature-fusion | yolo11m-c2psa-full-feature-no-concat.yaml |
| A6 | YOLO11m_ablation_Full-Width_Residual_C2PSA_v4.ipynb | ablation/c2psa-full-feature-add | yolo11m-c2psa-full-feature-add.yaml |
| A7 | YOLO11m_ablation_C2PSA+iRPE-K_v4.ipynb | ablation/c2psa-irpe-k | yolo11m-c2psa-irpe-k.yaml |
| A8 | YOLO11m_ablation_C3k2FusionBottleneckCrossPSA_v4.ipynb | ablation/attention-crosspsa-postfusion-p3full-p4half | yolo11m-crosspsa-postfusion-p3full-p4half.yaml |
| A9 | YOLO11m_ablation_Full-Feature-Fusion_non-concat_p3p4quarter-p5full_v4.ipynb | ablation/attention-psa-postfusion-p3p4quarter-p5full | yolo11m-psa-postfusion-p3p4quarter-p5full.yaml |
| L1 | YOLO11m_ablation_CFIoU_v4.ipynb | ablation/loss-cfiou | yolo11m.yaml |
| L2 | YOLO11m_ablation_InterpIoU_v4.ipynb | ablation/loss-interpiou | yolo11m.yaml |
| L3 | YOLO11m_ablation_dynamic-InterpIoU_v4.ipynb | ablation/loss-dynamic-interpiou | yolo11m.yaml |
| L4 | YOLO11m_ablation_PIoU-v2_v4.ipynb | ablation/loss-piou | yolo11m.yaml |

Baseline files are under notebooks/baselines/. The pretrained YOLO11m notebook serves both as B0 and as the pretrained member of the initialisation (pretrained / from scratch) comparison.

| Role | Notebook filename | Implementation | Selected model YAML |
|---|---|---|---|
| B0 / pretrained initialisation | YOLO11m_baseline_training_v4.ipynb | YOLO11-baseline | yolo11m.yaml |
| Scratch initialisation | YOLO11m_baseline_training_scratch_v3.ipynb | YOLO11-baseline | yolo11m.yaml |
| YOLO26m framework comparator | YOLO26m_baseline_training_v4.ipynb | official Ultralytics v8.4.115 | yolo26m.yaml |

The listed branches identify the relevant implementations. 

## Held-out notebooks and files

Model-specific notebooks are under notebooks/test/:

- YOLO11m_ablation_CBAM_Test_v4.ipynb
- YOLO11m_ablation_DW-HDC-SPPF_Test_v4.ipynb
- YOLO11m_baseline_Test_v4.ipynb
- YOLO26m_baseline_Test_v4.ipynb

The common analysis notebook is YOLO11m_Common_Test_Analysis_Cluster_Bootstrap_v4.ipynb under common-test-analysis/. That folder also contains:

- instances_test2017.json: shared test annotations.
- splits.csv: recorded split membership and leakage-risk group assignments.
- predictions/B0/predictions_coco_ids.json: YOLO11m baseline predictions.
- predictions/A1/predictions_coco_ids.json: CBAM before Detect predictions.
- predictions/C2/predictions_coco_ids.json: DW-HDC-SPPF predictions.
- predictions/YOLO26m/predictions_coco_ids.json: separate comparator predictions.

Saved files under common-test-analysis/results/:

- common_test_per_class_ap.csv
- common_test_point_estimates.csv
- paired_cluster_bootstrap_draws_seed20260824_n2000.csv
- paired_cluster_bootstrap_summary.csv
- prediction_validation_summary.csv
- runtime_and_complexity.csv
- test_image_groups.csv

## Opening the common analysis

Use a Python/Jupyter environment with NumPy, pandas, Matplotlib and tqdm. The notebook retains its installation cell for faster-coco-eval==1.7.2. From the repository root, launch Jupyter with the common test analysis folder as the working directory:

```powershell
python -m jupyter lab "YOLO11m_Common_Test_Analysis_Cluster_Bootstrap_v4.ipynb"
```

The notebook writes to `results/`. It reuses completed replicates when the supplied bootstrap-draw CSV is present. A fresh bootstrap calculation would require a separate working copy without that cached CSV.

## Dataset metadata

`dataset/data_split.yaml` defines the fourteen classes. Its `train/train.txt`, `train/val.txt` and `train/test.tx`t references are relative to the dataset root; these manifests are supplied under dataset/train/ in this package. Manifest entries retain their original relative image paths. The complete group-assignment record is supplied once as `common-test-analysis/splits.csv`.

