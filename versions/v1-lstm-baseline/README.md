# Version 1: LSTM Baseline

[← Back to Main](../../README.md) | [Next: V2 Transformer →](../v2-transformer/README.md)

---

## Overview

This version represents the initial approach using LSTM (Long Short-Term Memory) networks for sign language recognition. While the model trained successfully, this version contained a **data leakage issue** that was later identified and fixed in V2.

---

## Architecture

- **Model:** 2-layer LSTM with dropout
- **Input:** 30 frames × 225 features (MediaPipe landmarks)
- **Output:** 20 sign classes

---

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~96% (inflated due to data leakage) |
| Actual Test Accuracy | 40-47% |

**Note:** The high training accuracy was misleading. Data augmentation was applied before train/test split, causing augmented versions of the same video to appear in both sets.

---

## Files

### Scripts
| File | Description |
|------|-------------|
| `retrain_local.py` | Local retraining script |
| `data_augmentation.py` | Augmentation with data leakage (10x expansion) |
| `train_lstm_augmented.py` | Training on augmented data |

### Models
| File | Description |
|------|-------------|
| `sign_model_20.keras` | Initial local model |
| `best_model.h5` | GPU-trained model |
| `lstm_augmented.keras` | Model trained on augmented data |
| `norm_mean.npy`, `norm_std.npy` | Normalization parameters |

---

## Key Learnings

1. **Data leakage** — Augmenting before splitting inflates accuracy metrics
2. **LSTM limitations** — Struggled with signs that have similar starting motions
3. **Evaluation matters** — High training accuracy doesn't guarantee real-world performance

---

## What Changed in V2

The issues discovered here led to V2 improvements:
- Proper train/test split before augmentation
- Transformer architecture with attention mechanism
- Honest evaluation methodology

[Continue to Version 2 →](../v2-transformer/README.md)
