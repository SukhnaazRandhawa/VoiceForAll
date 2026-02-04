# Version 2: Transformer with Proper Evaluation

[← Back to Main](../../README.md) | [← Previous: V1 LSTM](../v1-lstm-baseline/README.md)

---

## Overview

This version addresses the data leakage issue from V1 and introduces a Transformer architecture. With proper evaluation methodology, we establish honest baselines and demonstrate that attention mechanisms significantly outperform LSTM for sign language recognition.

---

## Key Improvements Over V1

1. **Proper data splitting** — Train/test split before augmentation (no leakage)
2. **Fair comparison** — LSTM vs Transformer on identical data
3. **Transformer architecture** — Multi-head attention for temporal dependencies
4. **Real-time application** — Webcam-based sign recognition demo

---

## Architecture

### LSTM (Baseline)
- 2-layer LSTM with dropout
- Input: 30 frames × 225 features

### Transformer
- Multi-head self-attention (4 heads)
- Positional encoding for temporal information
- Input: 30 frames × 225 features

---

## Results

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| LSTM (proper split) | 57% | Honest baseline |
| Transformer | 81% | +24% improvement |

### Per-Sign Improvements

Signs that improved most with Transformer:
- Signs with similar starting motions now distinguished correctly
- Previously 0-20% accuracy → 80%+ accuracy on several classes

---

## Files

### Scripts
| File | Description |
|------|-------------|
| `data_augmentation_proper.py` | Augmentation after split (no leakage) |
| `train_lstm_proper.py` | LSTM training with proper methodology |
| `train_transformer.py` | Transformer model training |
| `analyse_confusion.py` | Confusion matrix analysis |

### Models
| File | Description |
|------|-------------|
| `lstm_proper.keras` | LSTM model (57% accuracy) |
| `transformer.keras` | Transformer model (81% accuracy) |
| `norm_mean_transformer.npy`, `norm_std_transformer.npy` | Normalization parameters |
| `X_train_proper.npy`, `y_train_proper.npy` | Training data (3,990 samples) |
| `X_test_proper.npy`, `y_test_proper.npy` | Test data (100 samples) |
| `top_20_words.npy` | Vocabulary (20 signs) |
| `confusion_matrix.png` | Model performance visualization |

### Application
| File | Description |
|------|-------------|
| `app.py` | Real-time webcam recognition demo |

---

## Running the Demo
```bash
# From the v2-transformer directory
python app.py
```

This launches a webcam interface for real-time sign recognition using the Transformer model.

---

## Real-World Testing Observations

While test accuracy is 81%, real-world performance varies due to:
- **Hand dominance** — Model trained mostly on right-handed signers
- **Occlusion** — Hands blocking each other reduces accuracy
- **Signing style** — Individual variations vs dataset signers

---

## Data Pipeline
```
WLASL-2000 (21,083 videos)
    ↓
Top 20 words selected (sufficient samples)
    ↓
Train/Test split (80/20)
    ↓
Augmentation on training set only
    ↓
MediaPipe feature extraction (225 features/frame)
    ↓
Model training
```

---

## Key Learnings

1. **Attention > Recurrence** — Transformers capture long-range dependencies better than LSTMs
2. **Methodology matters** — Proper splitting revealed true model performance
3. **Test ≠ Real-world** — 81% test accuracy doesn't guarantee practical usability
4. **Continuous testing** — Real-world validation essential alongside metrics
