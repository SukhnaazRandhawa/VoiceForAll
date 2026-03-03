# Version 5: Enhanced Transformer for PopSign ASL Recognition

[← Back to Main](../../README.md) | [← Previous: V4 BiLSTM](../v4-popsign-bilstm/V4_README.md) | [Next: V6 PopSign Hands-Only →](../v6-popsign-hands-only/V6_README.md)

---

## Overview

V5 is the most architecturally ambitious version of the project. It replaces the BiLSTM from V4 with an **Enhanced Transformer**: a significantly more complex model that combines a multi-scale CNN backbone, two Transformer blocks with 8-head attention, Gaussian noise regularisation, and a combined pooling strategy.

The motivation came directly from V4's analysis: the model was learning real patterns (evidenced by meaningful top-3 predictions) but wasn't generalising well to new signers and webcam conditions. V5 attacks this with aggressive regularisation techniques, label smoothing, Gaussian noise, and higher dropout, specifically designed to reduce the gap between test accuracy and real-world performance.

The result is counterintuitive but important: **test accuracy dropped slightly** (70.80% vs V4's 72.63%), but **real-world accuracy improved** (10/42 signs vs V4's 8/42). This trade-off, slightly worse on paper, better in practice, is one of the key research insights of the project.

The training script was run on the university GPU or Google Colab; only the trained weights and the webcam testing script are present in this version's directory.

---

## What Changed From V4

| Component | V4 | V5 |
|---|---|---|
| Architecture | Bidirectional LSTM | Enhanced Transformer |
| Feature extraction | 225 (hands + pose) | 225 (hands + pose) |
| CNN backbone | None | Multi-scale Conv1D (k=3 and k=5) |
| Attention heads | N/A | 8 heads per block |
| Transformer blocks | N/A | 2 |
| Pooling | None (LSTM hidden state) | GlobalAvgPool + GlobalMaxPool combined |
| Regularisation | Dropout(0.5) | GaussianNoise(0.1) + Dropout(0.1/0.3) |
| Loss function | Categorical cross-entropy | Categorical cross-entropy + label smoothing (0.1) |
| Learning rate | Adam default | Adam(0.0001) — explicit lower rate |
| Batch size | 32 | 64 |
| Max epochs | 40 | 100 |
| Test accuracy | 72.63% | 70.80% |
| Real-world working signs | 8/42 (19%) | 10/42 (24%) |

---

## Architecture

The model is built using the Functional API rather than Sequential, because the architecture has branching paths (the multi-scale CNN) that Sequential cannot express.

```
Input(60, 225)
→ GaussianNoise(0.1)                         ← active during training only
→ ┌─ Conv1D(64, k=3) → Conv1D(64, k=3) ─┐
  └─ Conv1D(64, k=5) → Conv1D(64, k=5) ─┘
→ Concatenate()                              ← 128 features per frame
→ + Positional Embedding(60, 128)
→ [Transformer Block 1]:
     MultiHeadAttention(8 heads, key_dim=16)
     → Dropout(0.1) → Add & LayerNorm
     → Dense(256, relu) → Dropout(0.1) → Dense(128)
     → Dropout(0.1) → Add & LayerNorm
→ [Transformer Block 2]:
     (identical structure to Block 1)
→ GlobalAveragePooling1D()  ─┐
→ GlobalMaxPooling1D()      ─┴─ Concatenate() → 256 features
→ Dense(128, relu) → Dropout(0.3)
→ Dense(64, relu)  → Dropout(0.3)
→ Dense(42, softmax)
```

---

## New Architectural Concepts

### 1. Multi-Scale CNN Backbone

Before the Transformer sees the sequence, it first passes through two parallel sets of Conv1D layers:

```python
conv_short = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
conv_short = Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv_short)

conv_long  = Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
conv_long  = Conv1D(64, kernel_size=5, padding='same', activation='relu')(conv_long)

x = Concatenate()([conv_short, conv_long])
```

**What is Conv1D?**

Conv1D (1D Convolution) is a sliding window operation applied along the time axis. A kernel of size 3 looks at 3 consecutive frames at a time and learns local patterns across them. A kernel of size 5 looks at 5 consecutive frames.

Think of it like reading a sentence, a small window picks up individual words and short phrases, while a larger window picks up longer phrases and context. For sign language:
- **Kernel size 3** captures fast, sharp movements, a quick flick of the wrist, a sudden hand shape change
- **Kernel size 5** captures slower, smoother movements, the arc of an arm, a gradual hand opening

By running both in parallel and concatenating their outputs, every frame's representation contains information about both fast and slow local temporal patterns before the Transformer's global attention even begins. This gives the attention mechanism richer features to work with.

### 2. Gaussian Noise

```python
x = GaussianNoise(0.1)(inputs)
```

This layer adds random noise drawn from a Gaussian (normal) distribution with standard deviation 0.1 to the input during training. During inference (webcam testing), it is automatically switched off.

**Why does this help?**

The training data (PopSign) was recorded on Pixel 4A smartphones in controlled conditions. Real webcam data is noisier- lighting varies, the camera is lower resolution, MediaPipe occasionally produces slightly wrong landmark positions. By training with artificial noise, the model learns to be robust to these small imperfections rather than relying on perfectly clean input.

This is similar to how humans learn to recognise handwriting, if you only ever read perfectly printed text, you'll struggle with messy handwriting. But if you practise with slightly distorted text, you generalise better.

### 3. Label Smoothing

Label smoothing modifies the training targets. Normally:
- Correct class gets probability 1.0
- All other classes get probability 0.0

With label smoothing of 0.1:
- Correct class gets probability 0.9
- Remaining 0.1 is distributed equally across all other classes

```python
# Normal target for class 3 out of 42:
[0, 0, 0, 1.0, 0, 0, ..., 0]

# With label smoothing=0.1:
[0.0024, 0.0024, 0.0024, 0.9, 0.0024, ..., 0.0024]
```

**Why does this help?**

Without label smoothing, the model is pushed to be 100% confident in every training prediction. This leads to **overconfidence** : the model assigns extreme probabilities (99.9% to one class) even for ambiguous inputs it has never seen before.

Label smoothing tells the model: "even when you're right, don't be completely certain." This produces more calibrated probabilities and better generalisation to new signers whose signing style differs slightly from the training data.

This directly explains why V5's test accuracy is slightly lower than V4 (the model is less aggressively optimised for the test set) but its real-world performance is better (the model is less brittle when it encounters slightly different input).

### 4. Combined Pooling

V2's Transformer used GlobalAveragePooling to collapse 40 frame vectors into one. V5 uses both pooling strategies and concatenates them:

```python
avg_pool = GlobalAveragePooling1D()(x)   # shape: (128,)
max_pool = GlobalMaxPooling1D()(x)       # shape: (128,)
x = Concatenate()([avg_pool, max_pool])  # shape: (256,)
```

- **GlobalAveragePooling** takes the mean across all 60 frames, it captures the overall, sustained character of the sign
- **GlobalMaxPooling** takes the maximum value at each position across all frames, it captures the most distinctive, peak moments of the sign

For sign language, this distinction matters. Some signs are defined by their average shape (like `blue`- a consistent B handshape throughout), while others are defined by a peak moment (like `bye`- the distinctive wave at its apex). Using both preserves information that either alone would lose.

### 5. More Attention Heads (8 vs 4)

V2 used 4 attention heads. V5 uses 8, but with a smaller key dimension (key_dim=16 vs key_dim=64):

```python
MultiHeadAttention(num_heads=8, key_dim=16)
```

More heads means the model runs 8 parallel attention operations simultaneously, each potentially specialising in a different aspect of the signing sequence. The smaller key dimension keeps the total parameter count manageable — 8 heads × 16 dimensions is similar total capacity to 4 heads × 32 dimensions, but with more diversity in what each head can learn to focus on.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | PopSign ASL v1.0 |
| Signs | 42 classes |
| Total samples | 14,483 |
| Train/test split | 80/20 stratified |
| Input shape | (60, 225) |
| Batch size | 64 |
| Optimizer | Adam(lr=0.0001) |
| Loss | Categorical cross-entropy + label smoothing (0.1) |
| Max epochs | 100 |
| Early stopping patience | 25 |

The lower learning rate (0.0001 vs Adam's default 0.001) combined with more epochs (100 vs 40) means the model trains more carefully and slowly, less likely to overshoot a good solution. The longer early stopping patience (25 vs 10) gives the model more time to find improvements before stopping.

---

## Results

### Test Set Performance

| Version | Architecture | Test Accuracy | Real-world working signs |
|---|---|---|---|
| V3 | Regular LSTM | 72.80% | 7/13 (54%) |
| V4 | Bidirectional LSTM | 72.63% | 8/42 (19%) |
| **V5** | **Enhanced Transformer** | **70.80%** | **10/42 (24%)** |
| PopSign Paper | BiLSTM | 84.20% | N/A |

### Real-World Webcam Performance

#### Signs That Work Reliably

| Sign | Confidence | Notes |
|---|---|---|
| arm | 95% | Excellent |
| bed | 96% | Excellent |
| blue | 87% | Very good |
| apple | 83% | Very good |
| bath | 80% | Good |
| bye | 77% | Good |
| cheek | 74% | Good |
| brother | 72% | Good |
| cloud | 64% | Moderate |
| awake | 53% | Lower confidence but consistent |

#### Signs That Work Sometimes

| Sign | Confidence | Issue |
|---|---|---|
| balloon | 33% | Inconsistent |
| because | 26% | Inconsistent |
| chair | 17% | Low confidence |
| closet | 43% | Inconsistent |
| alligator | variable | Depth movement issue |

#### New Signs Working in V5 (Not in V4)

Four signs that failed completely in V4 now work in V5: `awake`, `bath`, `cheek`, and `cloud`. This improvement is attributed to the Gaussian noise training making the model more tolerant of the occlusion and landmark noise that caused these signs to fail before.

---

## The Key Insight: Lower Test Accuracy, Better Real-World Performance

V5's test accuracy (70.80%) is slightly *lower* than V4's (72.63%), yet it works on more signs in practice (10 vs 8). This seems contradictory but makes complete sense once you understand what each regularisation technique does:

| Technique | Effect on test accuracy | Effect on real-world |
|---|---|---|
| Label smoothing | Slightly lower (less overfit to test distribution) | Better (more calibrated, less brittle) |
| Gaussian noise | Slightly lower (harder training task) | Better (robust to webcam noise) |
| Lower learning rate | Similar (more careful optimisation) | Better (avoids sharp minima that don't generalise) |

The test set comes from the same PopSign distribution as the training data, same signers, same recording conditions. A model that perfectly memorises dataset-specific features will score higher on this test set but fail on a new signer with a webcam. V5's regularisation deliberately prevents this memorisation.

This trade-off is a core research finding of the project and directly relevant to the dissertation discussion of real-world AI system deployment.

---

## Files

### Scripts

| File | Description |
|---|---|
| `scripts/test_webcam_transformer.py` | Rebuilds the Enhanced Transformer architecture from code, loads weights from `.npy`, runs manual record-and-predict webcam app |

> Note: The training script was run on the university GPU / Google Colab. Only the trained weights are stored locally as `transformer_enhanced_42_weights.npy`.

### Data

| File | Description |
|---|---|
| `data/label_map_popsign_42.npy` | Dictionary mapping class indices to sign names — shared with V4 (same 42 signs) |

### Models & Saved Artefacts

| File | Description |
|---|---|
| `models/transformer_enhanced_42_weights.npy` | Trained Enhanced Transformer weights saved as numpy array for cross-platform compatibility |

---

## Key Learnings

1. **Regularisation can improve real-world performance at the cost of test accuracy** : this is a genuine and important trade-off, not a failure. A model that generalises better to new conditions is more valuable than one that scores higher on a held-out set from the same distribution.
2. **Multi-scale temporal features help** : combining fast (k=3) and slow (k=5) CNN features before attention gives the Transformer richer local patterns to work with, capturing both sharp hand movements and sustained hand shapes.
3. **Combined pooling preserves more information** : using both average and max pooling captures both the sustained character and the peak moments of a sign, which pure average pooling misses.
4. **Gaussian noise is a cheap but effective augmentation** : adding noise during training costs nothing at inference time but meaningfully improves robustness to the imperfect landmark data that real webcams produce.
5. **225 features remained a limitation** : despite all architectural improvements, pose landmarks continued to add noise. The hypothesis going into V6 was that switching to hands-only features (63) would be more impactful than any further architectural change.
6. **The gap between test and real-world accuracy persists** : even with aggressive regularisation, 70.80% test accuracy translated to only ~24% real-world accuracy. The remaining gap pointed to a more fundamental issue: feature quality, not model complexity.

---

## What Changed in V6

V5 established that architecture and regularisation improvements alone cannot close the test-to-real-world gap. V6 took a fundamentally different approach, replicating the **exact methodology from the PopSign paper**, including hands-only features, left-hand mirroring, and the paper's specific BiLSTM configuration, to determine whether feature selection was the missing piece.

[Continue to Version 6 →](../v6-popsign-hands-only/V6_README.md)

---

## References

1. Thad Starner et al., *"PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones"*, NeurIPS 2023
2. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
3. Google ASL Signs Kaggle Competition: https://www.kaggle.com/competitions/asl-signs