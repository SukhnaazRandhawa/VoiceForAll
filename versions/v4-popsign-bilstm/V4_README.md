# Version 4: Bidirectional LSTM on PopSign Dataset

[← Back to Main](../../README.md) | [← Previous: V3 PopSign LSTM](../v3-popsign-lstm/V3_README.md) | [Next: V5 PopSign Transformer →](../v5-popsign-transformer/V5_README.md)

---

## Overview

V4 implements the **Bidirectional LSTM architecture** described in the PopSign paper, scaling up to 42 signs and replacing the regular LSTM from V3. It also introduces two new architectural elements : `Masking` and `Bidirectional` wrapping, and removes BatchNormalization in favour of higher dropout.

The headline result is that despite the task being significantly harder (42 signs vs 13), test accuracy stayed roughly the same as V3 (72.63% vs 72.80%), which confirms that BiLSTM is a stronger architecture. However, real-world webcam accuracy dropped sharply, from ~54% in V3 to ~19% in V4, because more signs means more opportunities for confusion, and many of the new signs share similar visual patterns.

V4 produced the most thorough real-world analysis of the project so far, identifying four concrete reasons why test accuracy and webcam accuracy diverge so dramatically. These findings directly shaped the feature selection strategy in V5 and V6.

---

## What Changed From V3

| Component | V3 | V4 |
|---|---|---|
| LSTM type | Regular (unidirectional) | Bidirectional |
| LSTM units | 128 → 64 | 128 → 128 (both layers) |
| Dropout | 0.3 | 0.5 |
| BatchNormalization | Yes | No |
| Masking layer | No | Yes |
| Number of signs | 13–26 | 42 |
| Total training videos | 5,861 | 14,483 |

---

## Architecture

```
Input(60, 225)
→ Masking(mask_value=0.0)
→ Bidirectional(LSTM(128, return_sequences=True))
→ Dropout(0.5)
→ Bidirectional(LSTM(128, return_sequences=False))
→ Dropout(0.5)
→ Dense(42, softmax)
```

### What is Bidirectional LSTM?

A regular LSTM reads the sequence in one direction: frame 1 → frame 2 → ... → frame 40. By the time it reaches frame 40, it has built up context from everything before it, but it has no knowledge of what comes after any given frame.

A Bidirectional LSTM runs **two LSTMs simultaneously** on the same sequence:
- One reads **forward**: frame 1 → frame 2 → ... → frame 60
- One reads **backward**: frame 60 → frame 59 → ... → frame 1

Their outputs are concatenated at each time step, so every frame's representation has context from both the past **and** the future.

```
Forward LSTM:  frame1 → frame2 → frame3 → ... → frame60
                                                      ↓
                                               concatenate → richer representation
                                                      ↑
Backward LSTM: frame1 ← frame2 ← frame3 ← ... ← frame60
```

For sign language this is particularly valuable. Consider a sign where the hand shape at frame 30 is ambiguous, it could belong to two different signs. The forward pass knows what happened before frame 30, and the backward pass knows what happens after. Together they can resolve the ambiguity that neither direction alone could.

In Keras, wrapping any LSTM with `Bidirectional()` handles all of this automatically, it creates both LSTMs and concatenates their outputs.

### What is the Masking Layer?

```python
Masking(mask_value=0.0)
```

Recall that shorter videos are zero-padded to reach 60 frames. This means some frames at the end of a sequence are entirely zeros, they don't represent real signing, just padding.

Without masking, the LSTM treats these zero frames as real data and tries to learn from them, which adds noise. The `Masking` layer tells the model: **"any frame that is entirely zeros is padding, ignore it during computation."**

The model then only processes the real frames of each video, which is both more accurate and more efficient. This is especially important in V4 because with 42 signs from 47 signers, video lengths vary considerably.

### Why Higher Dropout (0.5 vs 0.3)?

With more signs (42 vs 13), the model has more parameters and a higher risk of overfitting, memorising training examples rather than learning general patterns. Increasing dropout from 0.3 to 0.5 means 50% of neurons are randomly switched off during each training step, forcing the remaining neurons to learn more robust, distributed representations.

### Why Remove BatchNormalization?

V3 used BatchNormalization after each LSTM layer. V4 removes it. The reasoning: Masking creates variable-length sequences within each batch, and BatchNormalization computes statistics across the batch at each time step. When some sequences are masked at different points, these statistics become inconsistent and can actually hurt training. The Masking + BatchNorm combination is problematic, so BatchNorm was dropped in favour of higher Dropout for regularisation.

---

## Dataset

- **Dataset:** PopSign ASL v1.0 (same as V3)
- **Signs used:** 42 (expanded from 13 in V3)
- **Total videos:** 14,483
- **Average videos per sign:** ~345
- **Signers:** 47 Deaf adults
- **Features:** 225 per frame (left hand + right hand + pose)
- **Sequence length:** 60 frames (zero-padded)

### Training Configuration

| Parameter | Value |
|---|---|
| Train/test split | 80/20 stratified |
| Training samples | 11,586 |
| Test samples | 2,897 |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping patience | 10 |
| Optimizer | Adam |

---

## Results

### Test Set Performance

| Version | Signs | Architecture | Test Accuracy |
|---|---|---|---|
| V3 | 13 | Regular LSTM | 72.80% |
| V3 | 26 | Regular LSTM | 71.20% |
| **V4** | **42** | **Bidirectional LSTM** | **72.63%** |
| PopSign Paper | 250 | Bidirectional LSTM | 84.2% |

Maintaining 72.63% accuracy on 42 signs (vs 72.80% on 13 signs) is a meaningful result, the task tripled in difficulty but accuracy barely changed, confirming that BiLSTM is a genuine architectural improvement over regular LSTM.

### Real-World Webcam Performance

#### Signs That Work Reliably

| Sign | Confidence | Notes |
|---|---|---|
| arm | 99.9% | Excellent |
| brother | 99.9% | Excellent |
| bed | 99.3% | Excellent |
| bye | 97.8% | Excellent |
| find | 98.5% | Good |
| apple | 97.1% | Good |
| blue | 92.3% | Good |
| any | 89.1% | Good |

#### Signs That Work Sometimes

| Sign | Best Confidence | Issue |
|---|---|---|
| dryer | 96.3% | Inconsistent |
| because | 86.0% | Confused with black |
| balloon | 85.0% | Inconsistent |
| beside | 81.8% | Inconsistent |
| black | 65.1% | Confused with because |
| close | 60.9% | Inconsistent |
| chair | 40.8% | Often confused with arm |
| alligator | 39.2% | Often confused with another |

#### Summary

| Metric | Value |
|---|---|
| Test set accuracy | 72.63% |
| Real-world accuracy (reliable only) | ~19% (8/42) |
| Real-world accuracy (including partial) | ~38% (16/42) |
| Gap between test and real-world | ~53% |

---

## Why Such a Large Test vs Real-World Gap?

V4's real-world testing was the most systematic in the project so far. Four distinct causes were identified:

### 1. 2D Camera Cannot Capture 3D Movement

A laptop webcam only captures two dimensions (x and y). Signs that move **toward or away from the camera** depth movement, lose their most distinctive feature entirely.

| Sign type | Examples | Real-world result |
|---|---|---|
| Flat/planar movement | bye, blue, arm, brother |  High accuracy |
| Depth-based movement | alligator, cloud, balloon |  Low accuracy |

Signs like `alligator` involve opening and closing two hands in 3D space, on a 2D webcam this just looks like hands moving slightly, which resembles several other signs.

### 2. Hand-Face Occlusion

When the hand touches or covers part of the face during a sign, MediaPipe loses track of the hand landmarks, they become occluded by the face. The result is missing or wildly incorrect landmark values feeding into the model.

| Sign | Involves | Predicted as |
|---|---|---|
| ear | Touch ear | blue, bed |
| eye | Touch eye | cheek |
| cheek | Touch cheek | ear, apple |
| chin | Touch chin | cheek, apple |

### 3. Inter-Sign Visual Similarity

Many signs share similar hand shapes or movements. The model correctly learned that these are similar, which is actually evidence the model is working — but it cannot always distinguish them:

| Actual sign | Confused with | Why |
|---|---|---|
| bed | bedroom | Same base hand shape |
| black | because | Similar hand position near face |
| alligator | cloud, balloon | Both involve two hands opening/closing |
| bee | airplane | Similar single-hand shape |

Importantly, the **correct sign often appeared in the top-3 predictions** even when it wasn't the top prediction, alligator was 3rd (10.6%) when predicted as "another", chair was 2nd (22.7%) when predicted as "arm". This means the model has learned something real, it's just not confident enough to rank the right answer first.

### 4. Signer and Domain Variation

The model was trained on 47 native Deaf signers recorded on a Pixel 4A smartphone in controlled conditions. Real-world testing used a laptop webcam with a non-native signer in variable lighting. This **domain shift** : the difference between training conditions and deployment conditions, accounts for a significant portion of the accuracy gap.

This is also why the random train/test split produces optimistic results: the same 47 signers appear in both training and test sets, so the model has seen each signer's style before. In the real world, the model encounters a completely new signer it has never trained on.

---

## Complete Real-World Test Results (42 Signs)

| # | Sign | Result | Confidence | Notes |
|---|---|---|---|---|
| 1 | after | no | — | Predicted: arm |
| 2 | airplane | no | — | Predicted: bye |
| 3 | all | no | — | Predicted: brother |
| 4 | alligator | partial | 39% | Inconsistent |
| 5 | animal | no | — | Predicted: bath/brother |
| 6 | another | no | — | Predicted: cry |
| 7 | any | yes | 89% | Works well |
| 8 | apple | yes | 97% | Works well |
| 9 | arm | yes | 99% | Excellent |
| 10 | aunt | no | — | Predicted: ear |
| 11 | awake | no | — | Predicted: apple |
| 12 | backyard | no | — | Predicted: ear |
| 13 | bad | no | — | Predicted: eye/cheek |
| 14 | balloon | partial | 85% | Inconsistent |
| 15 | bath | no | — | Predicted: brother |
| 16 | because | partial | 86% | Sometimes works |
| 17 | bed | yes | 99% | Excellent |
| 18 | bedroom | no | — | Predicted: bed |
| 19 | bee | no | — | Predicted: airplane |
| 20 | before | no | — | Predicted: bye |
| 21 | beside | partial | 82% | Sometimes works |
| 22 | black | partial | 65% | Confused with because |
| 23 | blow | no | — | Predicted: bee |
| 24 | blue | yes | 92% | Works well |
| 25 | brother | yes | 99% | Excellent |
| 26 | bye | yes | 98% | Excellent |
| 27 | car | no | — | Predicted: dog/chair |
| 28 | carrot | no | — | Predicted: airplane/ear |
| 29 | cereal | no | — | Predicted: eye |
| 30 | chair | partial | 41% | Inconsistent |
| 31 | cheek | no | — | Predicted: apple/ear |
| 32 | chin | no | — | Predicted: cheek/apple |
| 33 | close | partial | 61% | Sometimes works |
| 34 | closet | no | — | Predicted: chair |
| 35 | cloud | no | — | Predicted: alligator |
| 36 | cry | no | — | Predicted: bath |
| 37 | dryer | partial | 96% | Sometimes works |
| 38 | ear | no | — | Predicted: blue/bed |
| 39 | every | no | — | Predicted: blue/car |
| 40 | eye | no | — | Predicted: cheek |
| 41 | find | yes | 98% | Works well |
| 42 | (additional) | — | — | — |

---

## Files

### Scripts

| File | Description |
|---|---|
| `scripts/train_bilstm.py` | Trains the Bidirectional LSTM on 42 signs; includes masking, higher dropout, and weight saving |
| `scripts/test_webcam_bilstm.py` | Webcam app — rebuilds architecture from code, loads weights from `.npy`, manual record-and-predict interaction |

### Data

| File | Shape | Description |
|---|---|---|
| `data/X_popsign_42.npy` | (14483, 60, 225) | Feature arrays for 42 signs |
| `data/y_popsign_42.npy` | (14483,) | Integer class labels |
| `data/label_map_popsign_42.npy` | dict | `{0: 'after', 1: 'airplane', ...}` |

### Models & Saved Artefacts

| File | Description |
|---|---|
| `models/bilstm_42_weights.npy` | Trained BiLSTM weights saved as numpy array for cross-platform compatibility |

> Note: The `.h5` model file (`best_model_bilstm_42.h5`) is saved during training but not committed, weights are loaded via the `.npy` file in the webcam app.

---

## Key Learnings

1. **Bidirectional context genuinely helps** : maintaining 72.63% on 42 signs (vs 72.80% on 13) confirms BiLSTM extracts more signal from the same features than regular LSTM.
2. **More signs ≠ lower accuracy, if architecture improves** : the task got 3× harder but accuracy held steady, which would not have happened with a regular LSTM.
3. **The 53% test-to-real-world gap has identifiable causes** : it's not random noise; it's 2D/3D limitations, occlusion, visual similarity, and domain shift. Each cause points to a specific fix.
4. **Sign distinctiveness remains the dominant factor** : the 8 reliable signs (arm, brother, bed, bye, find, apple, blue, any) all have visually unique, primarily planar movements with no face contact.
5. **225 features is increasingly suspected as the problem** : pose landmarks add noise without adding useful information for distinguishing signs. The PopSign paper used hands-only features (63). This became the primary hypothesis to test in V5 and V6.
6. **Top-3 accuracy is much better than top-1** : the correct sign appears in the top 3 predictions far more often than it ranks first. This suggested that better features, not a fundamentally broken model, was the path to improvement.

---

## What Changed in V5

The analysis here directly motivated V5's design:

- **Transformer architecture** : testing whether attention mechanisms further improve on BiLSTM
- **Focus on sign selection** : prioritising visually distinctive signs over broad vocabulary coverage

[Continue to Version 5 →](../v5-popsign-transformer/V5_README.md)