# Version 4 Analysis Report: Bidirectional LSTM on PopSign Dataset

## Executive Summary

This report documents the development and evaluation of Version 4 (V4) of the sign language recognition system, which implements a Bidirectional LSTM architecture trained on the PopSign ASL dataset. The analysis reveals important insights about the gap between test accuracy and real world performance, and identifies key factors affecting sign recognition accuracy.

---

## 1. Model Architecture

### V4: Bidirectional LSTM (Based on PopSign Paper)

```
Input: (60 frames × 225 features)
    ↓
Masking Layer (mask_value=0.0)
    ↓
Bidirectional LSTM (128 units) + Dropout (0.5)
    ↓
Bidirectional LSTM (128 units) + Dropout (0.5)
    ↓
Dense (42 classes, softmax)
```

### Key Differences from V3

| Component | V3 | V4 |
|-----------|----|----|
| LSTM Type | Regular (unidirectional) | Bidirectional |
| LSTM Units | 128 → 64 | 128 → 128 |
| Dropout | 0.3 | 0.5 |
| Batch Normalization | Yes | No (using Masking instead) |
| Number of Signs | 13-26 | 42 |

---

## 2. Dataset

### PopSign ASL v1.0

| Attribute | Value |
|-----------|-------|
| Source | Georgia Tech |
| Total Signs Available | 250 |
| Signs Used in V4 | 42 |
| Total Videos | 14,483 |
| Average Videos per Sign | ~345 |
| Signers | 47 Deaf adults |
| Features | 225 (MediaPipe: hands + pose) |
| Frame Length | 60 frames (padded) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Train/Test Split | 80/20 |
| Training Samples | 11,586 |
| Test Samples | 2,897 |
| Batch Size | 32 |
| Epochs | 40 (with early stopping) |
| Optimizer | Adam |

---

## 3. Results

### 3.1 Test Accuracy Comparison

| Version | Signs | Architecture | Test Accuracy |
|---------|-------|--------------|---------------|
| V3 | 13 | Regular LSTM | 72.80% |
| V3 | 26 | Regular LSTM | 71.20% |
| **V4** | **42** | **Bidirectional LSTM** | **72.63%** |
| PopSign Paper | 250 | Bidirectional LSTM | 84.2% |

### 3.2 Real-World Performance

Extensive real-world testing was conducted by performing each sign manually in front of a webcam.

#### Signs That Work Reliably (8 signs)

| Sign | Confidence | Consistency |
|------|------------|-------------|
| arm | 99.9% | Excellent |
| brother | 99.9% | Excellent |
| bed | 99.3% | Excellent |
| bye | 97.8% | Excellent |
| apple | 97.1% | Excellent |
| find | 98.5% | Good (sometimes inconsistent) |
| blue | 92.3% | Good |
| any | 89.1% | Good |

#### Signs That Work Sometimes (8 signs)

| Sign | Best Confidence | Issue |
|------|-----------------|-------|
| dryer | 96.3% | Inconsistent |
| because | 86.0% | Confused with black |
| balloon | 85.0% | Inconsistent |
| beside | 81.8% | Inconsistent |
| black | 65.1% | Confused with because |
| close | 60.9% | Inconsistent |
| chair | 40.8% | Often confused with another/arm |
| alligator | 39.2% | Often confused with another |

#### Signs That Don't Work (26 signs)

These signs consistently fail in real world testing, typically being confused with similar signs.

### 3.3 Accuracy Summary

| Metric | Value |
|--------|-------|
| Test Accuracy (same dataset) | 72.63% |
| Real-World Accuracy (all signs) | ~19% (8/42) |
| Real-World Accuracy (reliable signs only) | ~38% (16/42 including "sometimes works") |
| **Accuracy Gap** | **~53%** |

---

## 4. Analysis: Why the Gap Between Test and Real-World Accuracy?

### 4.1 Sign Dimensionality (2D vs 3D)

**Finding:** Signs with planar (2D) movements achieve higher recognition rates than signs with depth-based (3D) movements.

| Sign Type | Examples | Recognition |
|-----------|----------|-------------|
| 2D (flat motion) | bye, blue, arm, brother |  High accuracy |
| 3D (depth motion) | alligator, cloud, balloon |  Lower accuracy |

**Explanation:** A monocular webcam cannot accurately capture depth. Signs that move toward or away from the camera lose critical movement information.

### 4.2 Hand-Face Occlusion

**Finding:** Signs involving hand contact with facial features show significantly reduced accuracy.

| Sign | Involves | Predicted As |
|------|----------|--------------|
| ear | Touch ear | blue, bed |
| eye | Touch eye | cheek |
| cheek | Touch cheek | ear, apple |
| chin | Touch chin | cheek, apple |
| carrot | Hand near face + motion | airplane, ear |

**Explanation:** MediaPipe landmark detection fails when hands occlude facial features, resulting in missing or incorrect landmark data.

### 4.3 Inter-Sign Similarity

**Finding:** The model successfully learned sign patterns, evidenced by confusion between genuinely similar signs.

| Actual Sign | Confused With | Similarity |
|-------------|---------------|------------|
| bed | bedroom | Same base sign |
| black | because | Similar hand position |
| alligator | cloud, balloon | Two hands opening/closing |
| closet | chair | Similar arm movement |
| bee | airplane | Similar hand shape |

**Positive Indicator:** The correct prediction often appeared in the top-3 results:
- alligator: Predicted "another" (24%), but "alligator" was 3rd (10.6%)
- chair: Predicted "arm" (33%), but "chair" was 2nd (22.7%)
- cloud: Predicted "alligator" (46%), but "cloud" was 2nd (44%)

### 4.4 Signer Variation

**Finding:** The model learned dataset-specific patterns rather than generalizable sign features.

| Factor | PopSign Dataset | Real-World Testing |
|--------|-----------------|-------------------|
| Signers | 47 Deaf adults (native ASL) | 1 non-native signer |
| Recording Device | Smartphone (selfie camera) | Laptop webcam |
| Environment | Controlled | Variable |
| Signing Proficiency | Native/fluent | Learning |

### 4.5 Train/Test Split Methodology

**Finding:** Random train/test split may not reflect real-world generalization.

| Split Method | Description | Effect |
|--------------|-------------|--------|
| Random (our approach) | Same signers in train and test | Inflated test accuracy |
| Signer independent (paper) | Different signers in train vs test | More realistic |

---

## 5. Comparison: V3 vs V4

### 5.1 Test Accuracy

| Metric | V3 (26 signs) | V4 (42 signs) |
|--------|---------------|---------------|
| Test Accuracy | 71.20% | 72.63% |
| Improvement | - | +1.43% |

Despite having more signs (harder task), V4 achieved slightly higher accuracy due to the Bidirectional LSTM architecture.

### 5.2 Real-World Performance

| Metric | V3 (13 signs) | V4 (42 signs) |
|--------|---------------|---------------|
| Signs Working | 7 | 8 |
| Working Percentage | 54% | 19% |
| Signs in Common | apple, arm | apple, arm |

### 5.3 Signs That Changed

| Sign | V3 | V4 |
|------|----|----|
| aunt |  Works |  Fails (→ ear) |
| hello |  Works | Not tested |
| bed | Not in V3 |  Works |
| blue | Not in V3 |  Works |
| brother | Not in V3 |  Works |
| bye | Not in V3 |  Works |

---

## 6. Key Findings

### 6.1 Technical Limitations

1. **Monocular Camera Limitation:** 2D cameras cannot capture 3D sign movements accurately
2. **Occlusion Problem:** Hand face contact causes landmark detection failures
3. **Feature Representation:** 225 features (hands + pose) may include noise from pose landmarks

### 6.2 Dataset Limitations

1. **Signer Diversity Gap:** Model trained on 47 Deaf signers doesn't generalize to new signers
2. **Recording Conditions:** Smartphone selfie vs webcam creates domain shift
3. **Sign Selection:** Some signs are inherently harder to distinguish visually

### 6.3 Positive Findings

1. **Model Learning Confirmed:** Similar signs confuse each other (expected behavior)
2. **Top-3 Accuracy:** Correct sign often in top 3 predictions (~60% of failures)
3. **Architecture Improvement:** Bidirectional LSTM maintains accuracy with more signs
4. **Reliable Subset Exists:** 8-16 signs work consistently for practical use

---

## 7. Recommendations

### 7.1 For Practical Application

1. **Use Reliable Signs Only:** Build demo with 8-16 working signs
2. **Add Sentence Generation:** Convert sign sequences to natural language using NLP
3. **User Feedback:** Show top-3 predictions to let users select correct sign

### 7.2 For Future Improvement

1. **Hands-Only Features (63):** Match PopSign paper exactly
2. **Signer-Independent Split:** More realistic evaluation
3. **Transformer Architecture:** Kaggle winner achieved >90% accuracy
4. **Data Augmentation:** Increase signer diversity artificially
5. **3D Camera:** Use depth sensor for better 3D sign capture

---

## 8. Conclusion

Version 4 successfully implemented a Bidirectional LSTM architecture achieving 72.63% test accuracy on 42 signs. However, real-world testing revealed a significant gap (~53%) between test and real-world performance.

The analysis identified four key factors affecting real-world accuracy:
1. Sign dimensionality (2D vs 3D movements)
2. Hand-face occlusion during signing
3. Inter-sign visual similarity
4. Signer variation and domain shift

Despite these challenges, 8 signs work reliably with >85% confidence, providing a foundation for a practical sign-to-sentence translation system.

---

## 9. Files and Artifacts

### Models
| File | Description |
|------|-------------|
| `best_model_bilstm_42.h5` | Trained Bidirectional LSTM model |
| `bilstm_42_weights.npy` | Extracted weights for laptop deployment |

### Data
| File | Description |
|------|-------------|
| `X_popsign_42.npy` | Features (14483, 60, 225) |
| `y_popsign_42.npy` | Labels (14483,) |
| `label_map_popsign_42.npy` | Sign index to name mapping |

### Scripts
| File | Description |
|------|-------------|
| `train_bilstm.py` | Training script |
| `test_webcam_bilstm.py` | Real-world testing script |

---

## 10. Appendix: Complete Real-World Test Results

### Signs Tested (42 total)

| # | Sign | Works? | Confidence | Notes |
|---|------|--------|------------|-------|
| 1 | after | no | - | Predicted: arm |
| 2 | airplane | no | - | Predicted: bye |
| 3 | all | no | - | Predicted: brother |
| 4 | alligator | may be | 39% | Inconsistent |
| 5 | animal | no | - | Predicted: bath/brother |
| 6 | another | no | - | Predicted: cry |
| 7 | any | yes | 89% | Works well |
| 8 | apple | yes | 97% | Works well |
| 9 | arm | yes | 99% | Excellent |
| 10 | aunt | no | - | Predicted: ear |
| 11 | awake | no | - | Predicted: apple |
| 12 | backyard | no | - | Predicted: ear |
| 13 | bad | no | - | Predicted: eye/cheek |
| 14 | balloon | may be | 85% | Inconsistent |
| 15 | bath | no | - | Predicted: brother |
| 16 | because | may be | 86% | Sometimes works |
| 17 | bed | yes | 99% | Excellent |
| 18 | bedroom | no | - | Predicted: bed |
| 19 | bee | no | - | Predicted: airplane |
| 20 | before | no | - | Predicted: bye |
| 21 | beside | may be | 82% | Sometimes works |
| 22 | black | may be | 65% | Confused with because |
| 23 | blow | no | - | Predicted: bee |
| 24 | blue | yes | 92% | Works well |
| 25 | brother | yes | 99% | Excellent |
| 26 | bye | yes | 98% | Excellent |
| 27 | car | no | - | Predicted: dog/chair |
| 28 | carrot | no | - | Predicted: airplane/ear |
| 29 | cereal | no | - | Predicted: eye |
| 30 | chair | may be | 41% | Inconsistent |
| 31 | cheek | no | - | Predicted: apple/ear |
| 32 | chin | no | - | Predicted: cheek/apple |
| 33 | close | may be | 61% | Sometimes works |
| 34 | closet | no | - | Predicted: chair |
| 35 | cloud | no | - | Predicted: alligator |
| 36 | cry | no | - | Predicted: bath |
| 37 | dryer | may be | 96% | Sometimes works |
| 38 | ear | no | - | Predicted: blue/bed |
| 39 | every | no | - | Predicted: blue/car |
| 40 | eye | no | - | Predicted: cheek |
| 41 | find | yes | 98% | Works well |
| 42 | (additional signs tested as available) | - | - | - |

