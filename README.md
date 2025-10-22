# YOLOv5-Seg + SHAP: Explainable Object Detection for Industrial Monitoring

This repository demonstrates a complete workflow for **explaining YOLOv5-Seg** object detections using **SHAP (SHapley Additive exPlanations)**. The goal is to make model decisions **transparent and auditable** for safety-critical, real-time monitoring scenarios. 

---

## Overview

* **Model:** YOLOv5-Seg (backbone: CSPDarknet-s; multi-scale head; proto + mask coefficients). 
* **Explainability:** SHAP on image *superpixels* to quantify each region’s contribution to a detection’s confidence/IoU-based score. 
* **Target score (OD2Score):**
  [
  \text{score} = \mathrm{clamp}\big(\text{conf}_{\text{box}} \times \mathrm{IoU}(\text{box}, \text{target}),\ 0,\ 1\big)
  ]
  A unified 0–1 metric that reflects both localization (IoU) and classification confidence. 

The notebook produces heatmaps and cropped visualizations that highlight **which regions increase or decrease** the model’s score for a specific detection. 

---

## Repository Structure

```
yolov5-shap-explainability/
├─ shap-analysis.ipynb        # Main notebook
├─ README.md
└─ requirements.txt           # Python dependencies

```

> Output files (e.g., `shap_heatmap.png`, `cropped_object_shap.png`, `shap_heatmap_top.png`) are created. 

---

## Requirements

* **Python ≤ 3.9**
* **PyTorch (CUDA ≤ 2.1)** or CPU (SHAP will run ~10× slower on CPU)
* **Packages:** `shap`, `numpy`, `pandas`, `matplotlib`, `pillow`
* **Hardware:** GPU ≥ 6GB VRAM recommended; ≥16GB RAM to avoid OOM on large SHAP runs. 

Install with:

```bash
pip install -r requirements.txt
# or, minimally:
pip install shap numpy pandas matplotlib pillow torch torchvision
```

---

## Quickstart (Notebook)

1. Open `shap-analysis.ipynb` (Kaggle/Colab or local).
2. Run the **setup** cells (installs packages, imports).
3. Provide an **input image** and a **YOLOv5-Seg model/weights** (e.g., `yolov5s-seg.pt`).
4. Run inference and pick the **target detection** you want to explain.
5. Execute SHAP cells to generate:

   * `results/shap_heatmap.png` — full-frame SHAP heatmap
   * `results/cropped_object_shap.png` — cropped object with superpixel scores (0–100)
   * `results/shap_heatmap_top.png` — top-|SHAP| superpixels emphasized 

---

## Configuration (Key Variables)

These variables are exposed in the notebook for speed/quality trade-offs and visualization control:

| Variable            | Meaning                                   | Notes                                     |                |                                            |
| ------------------- | ----------------------------------------- | ----------------------------------------- | -------------- | ------------------------------------------ |
| `IMG_SIZE`          | Inference image size                      | Larger → more detail (more superpixels).  |                |                                            |
| `SP_W`              | Superpixel granularity (e.g., block size) | Smaller → finer explanation, slower.      |                |                                            |
| `BATCHES`           | Number of SHAP mini-batches               | ↑BATCHES → more samples total.            |                |                                            |
| `NSAMPLES`          | Samples per batch                         | ↑NSAMPLES → better estimates, slower.     |                |                                            |
| `target_idx`        | Which detection to explain                | Select the object index of interest.      |                |                                            |
| `TOP_PCT`           | Top-                                      | SHAP                                      | % to highlight | E.g., 50 shows the most influential half.  |
| `FIG_W, FIG_H, DPI` | Figure size & resolution                  | Use `DPI=600` for slides/publication.     |                |                                            |

---

## How It Works

1. **Run YOLOv5-Seg** to obtain detections (boxes, masks, confidences).
2. **Define superpixels** (e.g., fixed-size 4×4 blocks) as features for SHAP. 
3. **Compute OD2Score** for the selected detection (confidence × IoU with target). 
4. **Sample masks** of visible/hidden superpixels and fit a **Kernel SHAP** surrogate to estimate each region’s contribution. 
5. **Visualize** contributions: red increases the score, blue decreases, gray ~ neutral. Thresholded overlays emphasize the most critical regions. 

---

## Interpreting the Outputs

* **`shap_heatmap.png`:** Red = regions that *increase* the score; Blue = regions that *decrease* it; Gray ≈ neutral. 
* **`cropped_object_shap.png`:** Object crop with per-superpixel scores normalized to 0–100; empirically, blocks with scores **>60%** tend to be critical. 
* **`shap_heatmap_top.png`:** Same heatmap but with a custom colormap and `TOP_PCT` filtering to emphasize the most influential regions. 

---

## Troubleshooting

| Symptom                       | Likely Cause                                               | Fix                                                                     |
| ----------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| `KeyError: 94` (class id)     | Class label outside your model’s label set (COCO mismatch) | Ensure `model.names` matches your dataset/classes.                      |
| `too many indices for tensor` | Removed/reshaped the wrong columns (e.g., mask-coef)       | Re-run setup cells; ensure YOLOv5-Seg proto + coef handling is intact.  |
| GPU OOM                       | Batches/samples too large; VRAM too small                  | Reduce `BATCHES`/`NSAMPLES`, or run CPU with smaller settings.          |

---

## Notes & Good Practices

* For presentations, set `DPI=600` to generate publication-quality figures. 
* On CPU-only machines, expect significantly longer runtimes; start with small `NSAMPLES` (e.g., 200) and increase gradually. 
* Keep heavy model checkpoints out of the repo; reference them via links.

