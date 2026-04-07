# SOP-002: Data Processing Pipeline

**Project:** VoiceForAll — ASL Sign Language to Speech Translation  
**Version:** All Versions (V1–V6)  
**Last Updated:** April 2026  
**Author:** Sukhnaaz Kaur

---

## Purpose

This SOP documents the standard data processing pipeline used across all six versions of VoiceForAll. It defines how raw video data was transformed into model-ready feature arrays, how this process evolved across versions, and what outputs to expect at each stage.

---

## Dataset

| Property | Value |
|---|---|
| Dataset | PopSign ASL v1.0 (V3–V6) / WLASL-2000 (V1–V2) |
| Source | https://signdata.cc.gatech.edu |
| Signs used | 248 (V6 final) |
| Total videos | ~110,540 (V6) |
| Signers | 47 Deaf adults |
| Recording device | Pixel 4A smartphone |

---

## Feature Extraction Pipeline (V6 Final)

### Stage 1 — Video to Landmarks

Each video is processed frame by frame using MediaPipe Holistic:
```python
mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

For each frame, only the dominant hand landmarks are extracted:
```python
def extract_hands_only(results):
    if results.right_hand_landmarks:
        # Use right hand as-is
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    elif results.left_hand_landmarks:
        # Mirror left hand to normalise handedness
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([1.0 - lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
```

**Output per frame:** 63 features (21 landmarks × 3 coordinates)

### Stage 2 — Sequence Normalisation

Each video is normalised to exactly 60 frames:

| Condition | Action |
|---|---|
| Video < 60 frames | Zero-pad at the end |
| Video > 60 frames | Truncate to first 60 frames |
| No hand detected | Fill frame with 63 zeros |

**Output per video:** `(60, 63)` array

### Stage 3 — Dataset Assembly

All processed videos are stacked into a single feature array:

| File | Shape | Description |
|---|---|---|
| `X_popsign_248.npy` | `(110540, 60, 63)` | Feature arrays for all videos |
| `y_popsign_248.npy` | `(110540,)` | Integer class labels |
| `label_map_hands_only_248.npy` | dict | `{0: 'after', 1: 'airplane', ...}` |

### Stage 4 — Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ensures equal class distribution in both splits
)
```

| Split | Samples |
|---|---|
| Training | 88,432 |
| Test | 22,108 |

---

## How Feature Extraction Evolved Across Versions

| Version | Features | Landmarks Used | Left-hand mirroring |
|---|---|---|---|
| V1 | 225 | Hands + Pose | No |
| V2 | 225 | Hands + Pose | No |
| V3 | 225 | Hands + Pose | No |
| V4 | 225 | Hands + Pose | No |
| V5 | 225 | Hands + Pose | No |
| **V6** | **63** | **Hands only** | **Yes** |

The switch from 225 to 63 features in V6 was the single most impactful change across the entire project. Removing 162 pose landmarks eliminated noise that described how the signer was sitting rather than what they were signing.

---

## Key Processing Rules

1. **Always split before any augmentation** — augmenting before splitting causes data leakage (discovered in V1, fixed in V2)
2. **Zero-padding goes at the end** — the Masking layer in the model ignores trailing zeros automatically
3. **Mirror left hand, do not discard** — left-handed signers produce valid training data once mirrored
4. **No normalisation applied** — raw MediaPipe coordinates are used directly; the Masking layer handles zero-padding cleanly without normalisation
5. **Stratified split** — ensures every sign class has proportional representation in both train and test sets

---

## Verification

After running data processing, confirm:

- [ ] Feature array shape is `(N, 60, 63)` for V6
- [ ] Label map contains expected number of classes
- [ ] No NaN or Inf values in feature arrays
- [ ] Class distribution is balanced across train and test splits
```python
import numpy as np
X = np.load('X_popsign_248.npy')
y = np.load('y_popsign_248.npy')
print(X.shape)          # Expected: (110540, 60, 63)
print(np.isnan(X).any()) # Expected: False
print(np.unique(y).shape) # Expected: (248,)
```