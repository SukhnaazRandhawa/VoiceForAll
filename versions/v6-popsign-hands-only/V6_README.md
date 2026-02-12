# V6: PopSign Exact - Hands Only Features

## Overview

This version replicates the **exact methodology** from the PopSign paper (NeurIPS 2023), achieving significantly better real-world performance than previous versions.

**Key Achievement:** 81.27% test accuracy and **87.5% real-world accuracy** (14/16 signs working)

---

## Architecture

### Model: Bidirectional LSTM (PopSign Paper Exact)
```
Input (60 frames × 63 features)
    ↓
Masking (mask_value=0.0)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.5)
    ↓
Bidirectional LSTM (128 units, return_sequences=False)
    ↓
Dropout (0.5)
    ↓
Dense (num_classes, softmax)
```

### Features: 63 Dimensions (Hands Only)

| Component | Landmarks | Dimensions |
|-----------|-----------|------------|
| Right hand (or left flipped) | 21 | 63 (21 × 3) |
| **Total** | **21** | **63** |

### Key Preprocessing: Left-Hand Flip
```python
if results.right_hand_landmarks:
    # Use right hand as-is
    for lm in results.right_hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
elif results.left_hand_landmarks:
    # Flip left hand to right (mirror x-coordinate)
    for lm in results.left_hand_landmarks.landmark:
        landmarks.extend([1.0 - lm.x, lm.y, lm.z])
```

---

## Results

### Test Accuracy

| Signs | Test Accuracy | PopSign Paper |
|-------|---------------|---------------|
| 16 | **81.27%** | 82-84% (250 signs) |

### Real-World Performance

| Sign | Confidence | Status |
|------|------------|--------|
| airplane | 99.9% |  Perfect |
| all | 90.4% |  Works |
| alligator | 99.5% |  Perfect |
| animal | 99.7% |  Perfect |
| another | 75.5% |  Works |
| apple | 99.5% |  Perfect |
| arm | 99.9% |  Perfect |
| aunt | 100.0% |  Perfect |
| bath | 99.1% |  Perfect |
| bed | 99.9% |  Perfect |
| blue | 100.0% |  Perfect |
| brother | 92.8% |  Works |
| bye | 70.1% |  Works |
| cloud | 99.6% |  Perfect |
| after | - |  Confused with blue |
| any | - |  Confused with aunt |

**Real-World Accuracy: 14/16 = 87.5%**

---

## Comparison with Previous Versions

| Version | Architecture | Features | Test Acc | Real-World |
|---------|--------------|----------|----------|------------|
| V3 | Regular LSTM | 225 | 72.8% | 54% (7/13) |
| V4 | BiLSTM | 225 | 72.6% | 19% (8/42) |
| V5 | Transformer | 225 | 70.8% | 24% (10/42) |
| **V6** | **BiLSTM** | **63** | **81.3%** | **87.5% (14/16)** |

---

## Key Findings

### 1. Hands-Only Features Are Superior

Removing pose landmarks (99 dimensions) **improved** accuracy:
- Less noise from irrelevant body position data
- Model focuses on hand shape and movement
- Matches the original PopSign paper methodology

### 2. Left-Hand Flip Is Important

Mirroring left-hand signs to right-hand orientation:
- Normalizes all signs to consistent hand position
- Reduces variation in training data
- Improves generalization

### 3. Test-to-Real-World Gap Eliminated

| Model | Test Accuracy | Real-World | Gap |
|-------|---------------|------------|-----|
| V4/V5 | 70-73% | 19-24% | ~50% |
| **V6** | **81.3%** | **87.5%** | **-6%** |

---

## Training Configuration

Following PopSign paper exactly:

| Parameter | Value |
|-----------|-------|
| Epochs | 40 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Early stopping | Patience 10 |
| Dropout | 0.5 |

---

## Files
```
v6-popsign-hands-only/
├── data/
│   └── label_map_hands_only.npy      # Sign labels
├── models/
│   └── popsign_exact_weights.npy     # Trained weights
├── scripts/
│   └── test_webcam_hands_only.py     # Real-time testing
└── V6_README.md                       # This file
```

---

## Usage

### Real-Time Testing
```bash
cd scripts
python test_webcam_hands_only.py
```

**Controls:**
- SPACE: Start/Stop recording
- Q: Quit

---

## References

- **PopSign Paper:** "PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones" (NeurIPS 2023)
- **Dataset:** https://signdata.cc.gatech.edu/
- **Architecture:** Bidirectional LSTM as described in PopSign paper

---

## Future Work

1. **Expand vocabulary** - Process all 250 PopSign signs
2. **Sentence generation** - Combine recognized signs into sentences
3. **Cross-linguistic transfer** - Apply methodology to BSL

---

