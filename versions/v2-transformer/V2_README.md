# Version 2: Transformer with Proper Evaluation

[← Back to Main](../../README.md) | [← Previous: V1 LSTM](../v1-lstm-baseline/V1_README.md) | [Next: V3 PopSign LSTM →](../v3-popsign-lstm/V3_README.md)  

---

## Overview

V2 has two goals: **fix the data leakage bug from V1** and **test whether a Transformer architecture outperforms LSTM** on the same clean data.

The leakage fix alone was important, it revealed that V1's reported ~96% accuracy was completely misleading, and that the honest LSTM baseline was actually 57%. On top of that, the Transformer model trained in this version achieved 81% test accuracy, a genuine +24% improvement over LSTM.

However, despite strong test numbers, real-world webcam performance remained inconsistent, a recurring theme that would drive the direction of every subsequent version.

---

## What Changed From V1

| Issue in V1 | Fix in V2 |
|---|---|
| Augmentation applied before train/test split (data leakage) | Split first, then augment only the training set |
| Only one model architecture tested | LSTM and Transformer trained on identical data for fair comparison |
| Inflated test accuracy (~96%) | Honest test accuracy (LSTM: 57%, Transformer: 81%) |
| Normalisation computed over full dataset | Normalisation computed from training set only, then applied to test set |

---

## Architecture

### LSTM (Honest Baseline)

Same architecture as V1, kept identical so the only variable is the data split methodology:

```
Input(40, 225)
→ LSTM(128, return_sequences=True)
→ Dropout(0.4)
→ LSTM(64)
→ Dropout(0.4)
→ Dense(64, relu)
→ Dropout(0.4)
→ Dense(20, softmax)
```

### Transformer

A new architecture built around **self-attention**, the core idea being that instead of processing frames sequentially (like LSTM), the model can look at all 40 frames simultaneously and decide which ones are most relevant to each other.

```
Input(40, 225)
→ Dense(128)                        ← projects 225 features down to 128
→ + Positional Embedding(40, 128)   ← adds learned position information
→ [Transformer Block × 2]:
     MultiHeadAttention(4 heads, key_dim=64)
     → Dropout(0.3)
     → Add & LayerNorm               ← residual connection
     → Dense(128, relu)              ← feed-forward network
     → Dropout(0.3)
     → Dense(128)
     → Dropout(0.3)
     → Add & LayerNorm               ← residual connection
→ GlobalAveragePooling1D()           ← collapses 40 frames into one vector
→ Dense(64, relu)
→ Dropout(0.3)
→ Dense(20, softmax)
```

### Key Transformer Concepts

**Multi-Head Attention** : The model runs 4 parallel attention operations simultaneously. Each "head" learns to focus on different aspects of the signing sequence — one might focus on hand shape, another on movement direction, another on timing. Their outputs are combined.

**Positional Embedding** : Unlike LSTM which processes frames in order and naturally knows position, the Transformer sees all frames at once. Positional embeddings are learned vectors added to each frame's features to tell the model "this is frame 1", "this is frame 15", etc.

**Residual Connections (Add & LayerNorm)** : After each attention and feed-forward block, the input is added back to the output (`x = x + block(x)`). This creates a "shortcut" so gradients can flow directly through training without vanishing, making deep networks much easier to train.

**GlobalAveragePooling1D** : After the Transformer blocks, we have 40 vectors (one per frame). This layer averages them into a single vector that summarises the whole sequence before the final classification.

---

## Data Pipeline

```
WLASL-2000 dataset
        ↓
Filter to top 20 words by frequency
        ↓
Train/Test split (80/20, stratified)     ← THE KEY FIX from V1
        ↓
Augment ONLY the training set (10×)      ← test set stays clean
        ↓
Compute normalisation from training set  ← apply same stats to test set
        ↓
Train LSTM and Transformer separately
        ↓
Evaluate both on the same clean test set
```

### Why Normalise From Training Data Only?

In V1, normalisation was computed over the entire dataset (including test). This is another subtle form of leakage, the model indirectly "sees" statistics from the test set before evaluation. In V2, mean and std are computed from the training set only, then those exact same values are applied to normalise the test set. This is the correct approach.

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| LSTM V1 (with leakage) | ~96% | Misleading (not real) |
| LSTM V2 (honest) | 57% | True baseline |
| Transformer V2 | 81% | +24% over honest LSTM |
| Random chance | 5% | 1 in 20 classes |

### Per-Sign Analysis

The Transformer improved particularly on signs that LSTM struggled with, specifically signs with similar hand shapes or starting positions that require understanding the full motion arc to distinguish. Signs like `no`, `thin`, `deaf`, and `before` went from 0–20% accuracy under LSTM to significantly higher under the Transformer.

The `analyse_confusion.py` script was written specifically to investigate these failure cases, it produces a full confusion matrix showing exactly which signs get mistaken for which.

---

## Files

### Scripts

| File | Description |
|---|---|
| `scripts/data_augmentation_proper.py` | Splits data first, then augments only the training portion, fixes the V1 leakage bug |
| `scripts/train_lstm_proper.py` | Trains the same LSTM architecture as V1 on the clean split for an honest baseline |
| `scripts/train_transformer.py` | Builds and trains the Transformer model; includes per-class accuracy and focus on previously failing signs |
| `scripts/analyse_confusion.py` | Loads the LSTM model, generates predictions, builds a confusion matrix, and saves a heatmap to `Analysis/confusion_matrix.png` |

### Models & Saved Artefacts

| File | Description |
|---|---|
| `models/lstm_proper.keras` | LSTM trained on clean split, 57% honest accuracy |
| `models/transformer.keras` | Transformer model (81% test accuracy) |
| `models/norm_mean_proper.npy`, `norm_std_proper.npy` | Normalisation stats for the LSTM (from training set only) |
| `models/norm_mean_transformer.npy`, `models/norm_std_transformer.npy` | Normalisation stats for the Transformer |
| `models/X_train_proper.npy`, `models/y_train_proper.npy` | Augmented training set (~3,200 samples) |
| `models/X_test_proper.npy`, `models/y_test_proper.npy` | Clean test set (original videos only, no augmentation) |
| `models/top_20_words.npy` | Array of 20 sign class labels |
| `models/transformer_history.npy` | Training history (accuracy/loss per epoch) |

### Analysis

| File | Description |
|---|---|
| `Analysis/confusion_matrix.png` | Heatmap showing which signs get confused with which under the LSTM model |

### Application

| File | Description |
|---|---|
| `app.py` | Live webcam app, identical stability-based prediction loop as V1, but loads the Transformer model and its normalisation parameters |

---

## The App (`app.py`)

The app code is functionally identical to V1, same stability filter (8 consecutive matching predictions), same movement detection, same cooldown logic. The only difference is which model and normalisation files are loaded:

```python
model = keras.models.load_model('models/transformer.keras')
norm_mean = np.load('models/norm_mean_transformer.npy')
norm_std = np.load('models/norm_std_transformer.npy')
```

The commented-out code at the top of `app.py` is the earlier version that used the V1-style LSTM, it's kept for reference to show the evolution of the app.

---

## Key Learnings

1. **Fixing leakage revealed the true picture** : The jump from ~96% to 57% (LSTM honest) wasn't a regression; it was reality. The model hadn't improved, the measurement had just become honest.
2. **Transformers genuinely outperform LSTMs here** : The +24% gain on identical data is a real architectural advantage, not a data artefact. Attention mechanisms are better at capturing which frames in a signing sequence are most discriminative.
3. **Normalise from training data only** : Computing mean/std from the full dataset is a subtle leakage that's easy to miss. Always fit normalisation statistics on the training set, then transform everything else using those same statistics.
4. **Test accuracy ≠ real-world performance** : 81% on a held-out test set still translated to inconsistent webcam performance. The gap between controlled evaluation and real-world use became the central problem driving V3 onwards.

---

## What Changed in V3

The honest evaluation in V2 established that test accuracy alone is insufficient. V3 moved to the **PopSign dataset and methodology**, which was specifically designed for real-world sign recognition and provided a stronger foundation for practical usability.

[Continue to Version 3 →](../v3-popsign-lstm/V3_README.md)