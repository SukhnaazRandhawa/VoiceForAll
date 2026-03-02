# Version 1: LSTM Baseline

[← Back to Main](../../README.md) | [Next: V2 Transformer →](../v2-transformer/README.md)

---

## Overview

V1 is the starting point of the project, a classic LSTM-based sign language recognition system trained on the **top 20 most frequent signs** from the WLASL-2000 dataset. The goal was to establish a working end-to-end pipeline: extract MediaPipe landmarks from video, train a sequence model, and run live recognition through a webcam app.

The version produced a functioning system, but a **data leakage bug** was discovered post-training: augmentation was applied to the full dataset *before* the train/test split, meaning augmented copies of training videos leaked into the test set. This inflated reported accuracy and made the model appear stronger than it was in practice.

Despite this flaw, V1 was a critical first step, it validated the pipeline and produced the key learnings that shaped every version that followed.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Model | 2-layer LSTM with Dropout |
| Input shape | 40 frames × 225 features |
| Features | Left hand (63) + Right hand (63) + Pose (99) landmarks from MediaPipe Holistic |
| Output | 20 sign classes (softmax) |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |

### Model layers

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

---

## Dataset & Preprocessing

- **Dataset:** WLASL-2000 (filtered to top 20 words by frequency)
- **Features:** 225 MediaPipe landmarks per frame (left hand + right hand + pose)
- **Sequence length:** 40 frames per video
- **Normalization:** Global mean/std subtraction computed over the training set

### Data Augmentation (10× expansion)

Augmentation was applied to generate more training variety. The following transforms were used:

| Augmentation | Detail |
|---|---|
| Horizontal flip | Swaps left/right hands; flips all x-coordinates (`x → 1 - x`) |
| Gaussian noise | Two levels: σ = 0.01 and σ = 0.02 |
| Scaling | Two factors: 0.9× and 1.1× around centre (0.5) |
| 2D rotation | Two angles: −10° and +10° applied to (x, y) pairs |
| Time warp | Two speeds: 0.9× and 1.1× via frame interpolation |

> **Known bug:** All augmentations were applied to the *entire* dataset before the train/test split. This means augmented copies of the same source video appeared in both train and test sets, causing **data leakage** and inflated test metrics. This was corrected in V2.

---

## Results

| Metric | Value |
|--------|-------|
| Reported test accuracy (with leakage) | ~96% |
| Real-world estimated accuracy | 40–47% |

The gap between reported and real-world accuracy is entirely explained by data leakage. When the same video (or a lightly augmented copy of it) appears in both training and test splits, the model effectively memorises rather than generalises.

---

## Live Recognition App (`app.py`)

The webcam app implements a **stability-based prediction loop**:

1. Collect 40 frames of MediaPipe landmarks into a rolling buffer
2. Compute hand movement using variance of the last 10 frames
3. If movement exceeds threshold, run model inference
4. Only surface a prediction when the same word appears in **8 consecutive frames**
5. Apply a cooldown period of 20 frames after each confirmed sign to prevent repeats

### Key thresholds

| Parameter | Value | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum softmax confidence to log a prediction |
| `MOVEMENT_THRESHOLD` | 0.01 | Minimum hand variance to trigger inference |
| `STABILITY_FRAMES` | 8 | Consecutive matching predictions required |
| `COOLDOWN_FRAMES` | 20 | Frames to suppress new predictions after a detection |

---

## Files

### Scripts

| File | Description |
|------|-------------|
| `scripts/retrain_local.py` | Trains a fresh model on MacBook CPU using the top-20 filtered dataset |
| `scripts/data_augmentation.py` | Generates 10× augmented dataset (contains the data leakage bug) |
| `scripts/train_lstm_augmented.py` | Trains the LSTM on the augmented dataset with early stopping and model checkpointing |

### Tests

| File | Description |
|------|-------------|
| `tests/test_model_local.py` | Loads the saved model, verifies inference runs correctly, and prints a dummy prediction |

### Models & Saved Artefacts

| File | Description |
|------|-------------|
| `models/sign_model_20.keras` | Model trained locally via `retrain_local.py` |
| `models/best_model.h5` | Model trained on university GPU |
| `models/lstm_augmented.keras` | Best model from augmented training (used by `app.py`) |
| `models/top_20_words.npy` | Array of 20 sign class labels |
| `models/norm_mean.npy`, `norm_std.npy` | Normalisation stats from `retrain_local.py` |
| `models/norm_mean_aug.npy`, `norm_std_aug.npy` | Normalisation stats from augmented training |
| `models/X_augmented_20.npy` | Augmented feature array (shape: ~2000, 40, 225) |
| `models/y_augmented_20.npy` | Augmented labels array |
| `models/lstm_history.npy` | Training history (accuracy/loss per epoch) |

---

## Key Learnings

1. **Data leakage is easy to introduce and hard to spot** : augmenting before splitting is a common mistake that makes test metrics meaningless. Always split first, then augment only the training portion.
2. **225 features is noisier than expected** : combining hands and pose landmarks adds irrelevant body motion that hurts generalisation; later versions would test hand-only features.
3. **Stability filtering is essential for live use** : raw per-frame predictions are too noisy; requiring 8 consecutive consistent predictions significantly improved the usability of the webcam app.
4. **Real-world performance is the true benchmark** : high test accuracy means nothing if the model fails on live webcam input. Every version after this was evaluated against real-world usage, not just held-out data.

---

## What Changed in V2

The problems found here directly motivated the V2 design:

- **Fixed data leakage** — augmentation now applied only after train/test split
- **Switched to Transformer architecture** — attention mechanisms better capture which frames matter most for each sign
- **Improved evaluation methodology** — honest train/test separation and per-class accuracy reporting

[Continue to Version 2 →](../v2-transformer/README.md)