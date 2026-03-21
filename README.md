# ML-DL-Ops Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Sonam Sikarwar | B23CM1060**

[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/acinonyxxx/B23CM1060_best_model_ml-dl-ops_assignment_4/tree/main)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/isonam16/MLOps_Sonam_Sikarwar_B23CM1060/tree/Assignment-4)

---

## Overview

This assignment optimizes a custom **PyTorch Transformer** model for **English-to-Hindi translation** using **Ray Tune** paired with the **Optuna** search algorithm. The goal was to match or exceed the baseline BLEU score in significantly fewer epochs by finding an optimal hyperparameter configuration.

---

## Repository Structure

```
Assignment-4/
│
├── B23CM1060_ass_4_tuned_en_to_hi.ipynb   # Ray Tune + Optuna training notebook
├── en_to_hi.ipynb                          # Baseline training notebook
├── B23CM1060_ass_4_report.pdf             # 2-page report
├── B23CM1060_best_model.pth               # Best model weights (also on HuggingFace)
└── README.md
```

---

## Results at a Glance

| Metric | Baseline (100 epochs) | Best Tuned (30 epochs) | Improvement |
|---|---|---|---|
| Training Time | ~110 min | 25.3 min | **4.3× faster** |
| Final Loss | 0.0992 | 0.1082 | Comparable |
| BLEU Score | 71.47 | **90.38** | **+18.91 pts** |
| Epochs Used | 100 | 30 | **70% fewer** |

---

## Part 1 — Baseline

The baseline model was trained for **100 epochs** with the following fixed hyperparameters:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| Dropout | 0.1 |

- **Training Time:** ~110 minutes (~66 sec/epoch)  
- **Final Loss:** 0.0992  
- **BLEU Score:** 71.47

---

## Part 2 — Ray Tune + Optuna Setup

### Search Space (7 Hyperparameters)

| Hyperparameter | Range / Choices | Method |
|---|---|---|
| Learning Rate | 1e-5 to 1e-2 | `loguniform` |
| Batch Size | 16, 32, 64, 128 | `choice` |
| Attention Heads | 4, 8, 16 | `choice` |
| Feed-Forward Dim | 1024, 2048 | `choice` |
| Dropout | 0.1 to 0.4 | `uniform` |
| Optimizer | Adam, AdamW | `choice` |
| Weight Decay | 1e-5 to 1e-2 | `loguniform` |

### Scheduler

**ASHA Scheduler** was used alongside OptunaSearch to prune underperforming trials early:
- `max_t = 20`
- `grace_period = 5`
- `reduction_factor = 2`

**20 trials** were run, each capped at **30 epochs**. Total sweep time: **103.9 minutes**.

---

## Part 3 — Best Configuration

| Parameter | Best Value |
|---|---|
| Learning Rate | 0.000212 |
| Batch Size | 16 |
| Attention Heads | 16 |
| Feed-Forward Dim | 2048 |
| Dropout | 0.187 |
| Optimizer | AdamW |
| Weight Decay | 0.00271 |
| Best Trial Loss | 0.2063 |

Retrained with best config for **30 epochs**:
- **Training Time:** 25.3 minutes  
- **Final Loss:** 0.1082  
- **BLEU Score:** 90.38 ✅

---

## Model Weights

The best model weights are available on HuggingFace:

🤗 [B23CM1060\_best\_model\_ml-dl-ops\_assignment\_4](https://huggingface.co/acinonyxxx/B23CM1060_best_model_ml-dl-ops_assignment_4/tree/main)




## Key Insights

- **Small batch size (16)** introduced beneficial gradient noise, improving generalisation
- **AdamW** with weight decay suppressed overfitting within 30 epochs
- **16 attention heads** improved source-target alignment quality
- **CosineAnnealingLR** enabled fast early convergence followed by smooth fine-tuning
- **ASHA** made the 20-trial sweep feasible on a single GPU in under 2 hours

---

*ML-DL-Ops Assignment 4 | Sonam Sikarwar | B23CM1060*
