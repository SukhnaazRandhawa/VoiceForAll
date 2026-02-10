# Version 3: PopSign LSTM Baseline

## Overview

This version transitions from the WLASL dataset to the **PopSign ASL v1.0 dataset**, addressing the fundamental data scarcity issues discovered in V1 and V2. The key improvement is significantly more training data with diverse, verified Deaf signers.

## Why Switch to PopSign?

After analyzing V1 and V2 results, I identified that the core problem was **insufficient training data diversity**, not model architecture. The WLASL dataset had only ~20 videos per sign from unknown sources, making it impossible to learn generalizable patterns.

| Problem with WLASL | PopSign Solution |
|-------------------|------------------|
| ~20 videos per sign | ~450 videos per sign (22x more) |
| Scraped from YouTube | Collected from 47 verified Deaf signers |
| Unknown signer diversity | Controlled, diverse signing styles |
| Inconsistent video quality | Standardized smartphone recording |
| 40-47% real-world accuracy (V1/V2) | Improved real-world performance |

## Dataset Details

### Source
- **Dataset**: PopSign ASL v1.0
- **Institution**: Georgia Tech
- **Paper**: NeurIPS 2023
- **License**: CC BY 4.0 (free for research)
- **Website**: https://signdata.cc.gatech.edu

### Statistics
| Metric | Value |
|--------|-------|
| Signs Used | 13 (subset of 250 available) |
| Total Videos | 5,861 |
| Videos per Sign | 255 - 557 (avg ~450) |
| Signers | 47 Deaf adults |
| Recording Device | Pixel 4A smartphone (selfie camera) |
| Resolution | 1944 × 2592 |
| Frame Rate | 120 fps |

### Signs Included (13 signs)
```
TV, after, airplane, all, alligator, animal, 
another, any, apple, arm, aunt, awake, hello
```

### Videos Per Sign Distribution
| Sign | Videos | Sign | Videos |
|------|--------|------|--------|
| TV | 509 | any | 489 |
| after | 395 | apple | 459 |
| airplane | 255 | arm | 300 |
| all | 291 | aunt | 487 |
| alligator | 518 | awake | 507 |
| animal | 542 | hello | 552 |
| another | 557 | | |

## Feature Extraction

### MediaPipe Landmarks (225 features per frame)
| Body Part | Landmarks | Features |
|-----------|-----------|----------|
| Left Hand | 21 | 63 (x, y, z) |
| Right Hand | 21 | 63 (x, y, z) |
| Pose | 33 | 99 (x, y, z) |
| **Total** | **75** | **225** |

### Preprocessing
- Frame sampling: 60 frames per video
- Padding: Zero-padding for videos shorter than 60 frames
- Truncation: Cut videos longer than 60 frames
- No normalization applied

## Model Architecture

```
Input: (60 frames, 225 features)
    ↓
LSTM(128, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
BatchNormalization()
    ↓
LSTM(64, return_sequences=False)
    ↓
Dropout(0.3)
    ↓
BatchNormalization()
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(13, activation='softmax')
    ↓
Output: 13 class probabilities
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Early Stopping Patience | 10 |
| Train/Test Split | 80/20 (stratified) |

## Results

### Test Set Performance
| Metric | Value |
|--------|-------|
| Training Accuracy | 81.23% |
| Validation Accuracy (Best) | 77.75% |
| **Test Accuracy** | **72.80%** |

### Real-World Webcam Testing

After training, the model was tested in real-world conditions using a laptop webcam:

| Sign | Works? | Confidence | Notes |
|------|--------|------------|-------|
| arm |  Yes | 99.2% | Very reliable |
| hello |  Yes | 97.1% | Very reliable |
| apple |  Yes | 89% | Occasionally confused with aunt |
| animal |  Partial | 57.6% | Lower confidence |
| aunt |  Partial | 58% | Lower confidence |
| awake |  Partial | 41% | Confused with hello when hands move from eyes |
| TV |  Partial | 27.1% | Low confidence |
| after |  No | - | Predicted as arm (97.8%) |
| airplane |  No | - | Predicted as aunt/awake |
| all |  No | - | Predicted as any (59.6%) |
| alligator |  No | - | Predicted as arm (96.3%) |
| another |  No | - | Predicted as TV (30.7%) |
| any |  No | - | Predicted as animal/aunt |

**Real-World Accuracy: 7/13 signs = 54%**

### Confusion Matrix Analysis

Key findings from confusion matrix:

| Actual Sign | Often Confused With | Reason |
|-------------|---------------------|--------|
| after | arm | Similar arm movement |
| airplane | aunt, awake | Hand near face |
| all | any | Very similar hand shapes |
| alligator | arm | Arm extension similar |
| any | animal, aunt | Similar hand shapes |

**"another" was the dominant confuser** - many signs were incorrectly classified as "another" due to its broad motion patterns.

## Key Findings

### 1. Test Accuracy vs Real-World Gap
| Environment | Accuracy |
|-------------|----------|
| Test Set (PopSign data) | 72.80% |
| Real-World (webcam) | ~54% |
| **Gap** | **~19%** |

This gap indicates the model learned dataset-specific patterns rather than fully generalizable sign recognition.

### 2. Sign Distinctiveness Matters More Than Data Quantity

| Sign | Videos | Works? |
|------|--------|--------|
| arm | 300 (lowest) |  Best (99.2%) |
| another | 557 (highest) |  Fails |
| alligator | 518 |  Fails |

**Conclusion**: Visually distinctive signs work better, regardless of training data quantity.

### 3. Similar Signs Cause Confusion

Signs with similar characteristics get confused:
- **Hand near face**: aunt, awake, apple (partial confusion)
- **Arm movements**: arm, after, alligator (strong confusion)
- **Similar hand shapes**: all, any (strong confusion)

## Comparison: V1/V2 vs V3

| Metric | V1 (WLASL) | V2 (WLASL) | V3 (PopSign) |
|--------|------------|------------|--------------|
| Dataset | WLASL | WLASL | PopSign |
| Videos/Sign | ~20 | ~20 | ~450 |
| Total Videos | ~400 | ~400 | 5,861 |
| Architecture | LSTM | Transformer | LSTM |
| Test Accuracy | 96%* | TBD | 72.80% |
| Real-World Accuracy | 40-47% | TBD | ~54% |

*V1 had data leakage, inflating test accuracy

## Files

### Data
| File | Shape | Description |
|------|-------|-------------|
| `X_popsign.npy` | (5861, 60, 225) | Features for 13 signs |
| `Y_popsign.npy` | (5861,) | Labels |
| `label_map_popsign.npy` | dict | {0: 'TV', 1: 'after', ...} |

### Models
| File | Description |
|------|-------------|
| `best_model_popsign.keras` | Best model (72.80% test accuracy) |
| `popsign_weights.npy` | Weights for cross-platform compatibility |

### Scripts
| File | Description |
|------|-------------|
| `train_lstm_popsign.py` | Training script |
| `test_webcam_popsign.py` | Real-world webcam testing |
| `check_weights.py` | Utility to verify weight shapes |
| `quick_test.py` | Quick model verification |

### Analysis
| File | Description |
|------|-------------|
| `confusion_matrix_13signs.png` | Confusion matrix visualization |

## Lessons Learned

1. **More data improves generalization** - 450 videos/sign vs 20 videos/sign showed clear improvement in model stability

2. **Diverse signers are crucial** - 47 Deaf signers provided varied signing styles that improve real-world performance

3. **Test accuracy ≠ Real-world accuracy** - 72.80% test accuracy translated to only ~54% real-world accuracy

4. **Sign distinctiveness matters** - Visually distinct signs (arm, hello) work reliably; similar signs (all/any, after/arm) cause confusion

5. **Architecture matters** - Our regular LSTM achieved 72.80% vs the paper's Bidirectional LSTM at 84.2%

6. **Feature selection matters** - Using 225 features (hands + pose) vs paper's 63 features (hands only) may add noise

## Limitations

1. **Only 13 signs** - Small vocabulary limits practical use
2. **Regular LSTM** - Not matching paper's Bidirectional LSTM architecture
3. **No left-hand flip** - Left-handed signs not normalized to right-hand
4. **Extra features** - Pose landmarks may add noise rather than help
5. **Random train/test split** - Not signer-independent like the paper

## Next Steps (V4)

To match the PopSign paper's 84.2% accuracy:

- [ ] Implement Bidirectional LSTM
- [ ] Use hands-only features (63 instead of 225)
- [ ] Implement left-hand flip normalization
- [ ] Train on all 250 signs
- [ ] Use signer-independent train/test splits

## References

1. **PopSign Dataset Paper**: Thad Starner et al., "PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones", NeurIPS 2023 Datasets and Benchmarks Track

2. **Dataset Website**: https://signdata.cc.gatech.edu

3. **Kaggle Competition**: https://www.kaggle.com/competitions/asl-signs (Google - Isolated Sign Language Recognition)

4. **1st Place Solution**: https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution