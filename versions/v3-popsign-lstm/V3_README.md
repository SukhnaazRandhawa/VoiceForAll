# Version 3: PopSign LSTM Baseline

[← Back to Main](../../README.md) | [← Previous: V2 Transformer](../v2-transformer/V2_README.md) | [Next: V4 PopSign BiLSTM →](../v4-popsign-bilstm/V4_README.md)

---

## Overview

V3 makes the most significant dataset change in the project, switching from WLASL to the **PopSign ASL v1.0 dataset**, collected by Georgia Tech and published at NeurIPS 2023. The motivation was straightforward: V1 and V2 both revealed a persistent gap between test accuracy and real-world webcam performance, and the root cause was traced not to model architecture but to **insufficient and low-diversity training data**.

WLASL provided roughly 20 videos per sign from unknown YouTube sources. PopSign provides roughly 450 videos per sign from 47 verified Deaf adult signers recorded under standardised conditions. That difference in data quality and quantity is the defining change in V3.

The version also introduced a new interaction model for the webcam app, instead of continuous automatic prediction, the user **manually controls recording** by pressing SPACE to start and stop a sign, after which the model predicts. This proved more reliable in practice than the rolling-buffer approach used in V1 and V2.

---

## Why Switch to PopSign?

| Problem with WLASL | PopSign Solution |
|---|---|
| ~20 videos per sign | ~450 videos per sign (22× more) |
| Scraped from YouTube | Collected from 47 verified Deaf signers |
| Unknown signer diversity | Controlled, diverse signing styles |
| Inconsistent video quality | Standardised smartphone recording |
| ~54% real-world accuracy (V1/V2) | Improved real-world generalisation |

---

## Dataset Details

- **Dataset:** PopSign ASL v1.0
- **Institution:** Georgia Tech
- **Published:** NeurIPS 2023 Datasets and Benchmarks Track
- **License:** CC BY 4.0 (free for research use)
- **Website:** https://signdata.cc.gatech.edu

### Statistics

| Metric | Value |
|---|---|
| Signs used | 13 (subset of 250 available) |
| Total videos | 5,861 |
| Videos per sign | 255–557 (avg ~450) |
| Signers | 47 Deaf adults |
| Recording device | Pixel 4A smartphone (selfie camera) |
| Resolution | 1944 × 2592 |
| Frame rate | 120 fps |

### Signs Included

```
TV, after, airplane, all, alligator, animal,
another, any, apple, arm, aunt, awake, hello
```

### Videos Per Sign

| Sign | Videos | Sign | Videos |
|---|---|---|---|
| TV | 509 | any | 489 |
| after | 395 | apple | 459 |
| airplane | 255 | arm | 300 |
| all | 291 | aunt | 487 |
| alligator | 518 | awake | 507 |
| animal | 542 | hello | 552 |
| another | 557 | | |

---

## Feature Extraction

Same MediaPipe Holistic pipeline as V1 and V2- 225 features per frame:

| Body Part | Landmarks | Features |
|---|---|---|
| Left hand | 21 | 63 (x, y, z per landmark) |
| Right hand | 21 | 63 (x, y, z per landmark) |
| Pose (body) | 33 | 99 (x, y, z per landmark) |
| **Total** | **75** | **225** |

### Sequence Handling

- Each video is sampled to exactly **60 frames** (up from 40 in V1/V2)
- Videos shorter than 60 frames are **zero-padded** at the end
- Videos longer than 60 frames are **truncated**
- No normalisation applied in this version

The move from 40 to 60 frames gives the model more temporal context per sign, which matters for longer signs in the PopSign vocabulary.

---

## Model Architecture

V3 uses the same LSTM family as V1, but adds **BatchNormalization** layers, a new addition not present in earlier versions:

```
Input(60, 225)
→ LSTM(128, return_sequences=True)
→ Dropout(0.3)
→ BatchNormalization()
→ LSTM(64, return_sequences=False)
→ Dropout(0.3)
→ BatchNormalization()
→ Dense(64, relu)
→ Dropout(0.3)
→ Dense(13, softmax)
```

### What is BatchNormalization?

BatchNormalization is a layer that normalises the outputs of the previous layer during training. After each LSTM layer, the values flowing into the next layer can be on very different scales, some large, some small. BatchNormalization rescales them to have a consistent mean and variance.

This has two benefits: it makes training faster and more stable, and it acts as a mild regulariser (helping prevent overfitting). Think of it as tidying up the numbers between layers so the next layer always receives input in a well-behaved range.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs |
| Train/test split | 80/20 stratified |

---

## The New Webcam Interaction Model

V1 and V2 used a **continuous rolling-buffer** approach : the app predicted automatically every time 40 frames were collected, using stability filters to reduce noise. V3 switches to a **manual record-and-predict** approach:

1. User presses **SPACE** to start recording
2. User performs the sign
3. User presses **SPACE** again to stop recording
4. Model predicts on the recorded frames
5. Result displayed on screen with confidence colour coding

```python
if key == ord(' '):
    if not recording:
        recording = True
        recorded_frames = []
    else:
        recording = False
        if len(recorded_frames) >= MIN_FRAMES:
            sign, conf, probs = predict_sign(recorded_frames)
```

This approach is more reliable for evaluation because the model always gets a complete, intentional sign rather than a random 40-frame window that might capture mid-sign or between-sign frames.

The confidence is colour-coded on screen:
- **Green** : confidence > 70%
- **Yellow** : confidence 40–70%
- **Red** : confidence < 40%

### Predict Function : Padding and Truncation

```python
def predict_sign(frames):
    frames = np.array(frames)
    if len(frames) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(frames), 225))
        frames = np.vstack([frames, padding])
    frames = frames[:MAX_FRAMES]
    X = frames.reshape(1, MAX_FRAMES, 225)
    predictions = model.predict(X, verbose=0)[0]
```

Whatever the user recorded (between MIN_FRAMES=10 and MAX_FRAMES=60 frames), this function always produces a (1, 60, 225) array for the model. Short recordings are padded with zeros; long recordings are truncated.

---

## Weight Serialisation : A New Approach

V3 introduces saving model weights as `.npy` files separately from the model architecture, in addition to saving the full `.keras` model. This was done to solve a practical problem - `.keras` model files sometimes have compatibility issues between different machines, TensorFlow versions, or operating systems (important when moving between university GPU systems and a MacBook).

The solution: save weights as a numpy array separately, then rebuild the architecture from scratch and load the weights in.

**Saving weights:**
```python
# Implicitly done via ModelCheckpoint saving .keras,
# then separately: np.save('popsign_weights.npy', model.get_weights())
```

**Loading weights (in test scripts):**
```python
model = Sequential([...])          # rebuild architecture
weights = np.load('popsign_weights.npy', allow_pickle=True)
model.set_weights(list(weights))   # inject weights
```

`check_weights.py` and `test2_for_7_Sign.py` exist specifically to verify this process worked correctly, checking that the number of weight arrays and their shapes match what the model architecture expects.

---

## The 7-Sign Experiment

Midway through V3, a smaller 7-sign subset was trained and tested (via `test_webcam_popsign.py` and the `label_map_7signs.npy` / `final_7_sign_weights.npy` files). This was an exploratory experiment to test whether reducing the number of signs, and therefore reducing inter-class confusion, would improve real-world accuracy.

The 7-sign model uses the identical architecture to the 13-sign model, just with `Dense(7, softmax)` as the output layer instead of `Dense(13, softmax)`.

---

## Results

### Test Set Performance

| Metric | Value |
|---|---|
| Training accuracy | 81.23% |
| Best validation accuracy | 77.75% |
| **Test accuracy** | **72.80%** |

### Real-World Webcam Performance

| Sign | Works? | Confidence | Notes |
|---|---|---|---|
| arm |  Yes | 99.2% | Very reliable |
| hello |  Yes | 97.1% | Very reliable |
| apple |  Yes | 89.0% | Occasionally confused with aunt |
| animal |  Partial | 57.6% | Lower confidence |
| aunt |  Partial | 58.0% | Lower confidence |
| awake |  Partial | 41.0% | Confused with hello |
| TV |  Partial | 27.1% | Low confidence |
| after |  No | — | Predicted as arm (97.8%) |
| airplane |  No | — | Predicted as aunt/awake |
| all |  No | — | Predicted as any (59.6%) |
| alligator |  No | — | Predicted as arm (96.3%) |
| another |  No | — | Predicted as TV (30.7%) |
| any |  No | — | Predicted as animal/aunt |

**Real-world accuracy: 7/13 signs = ~54%**

### The Test vs Real-World Gap

| Environment | Accuracy |
|---|---|
| Test set (PopSign data) | 72.80% |
| Real-world webcam | ~54% |
| Gap | ~19% |

This gap tells us the model learned some patterns that are specific to how PopSign videos were recorded (lighting, camera angle, signing style of the 47 signers) rather than fully generalised sign recognition.

---

## Key Findings

### 1. Sign Distinctiveness Matters More Than Data Volume

The most striking finding in V3 is that the signs with the *least* training data performed best, while signs with the *most* training data failed completely:

| Sign | Videos | Real-world result |
|---|---|---|
| arm | 300 (lowest) |  99.2% confidence |
| another | 557 (highest) |  Complete failure |
| alligator | 518 |  Complete failure |
| hello | 552 |  97.1% confidence |

`arm` and `hello` are **visually distinctive** : their hand shapes and movements are unlike any other sign in the vocabulary. `another`, `alligator`, and `after` all involve similar arm movements and get systematically confused with each other.

This finding was a turning point in the project, it shifted focus from "get more data" to "choose more distinctive signs and better features."

### 2. Systematic Confusion Patterns

| Actual sign | Confused with | Reason |
|---|---|---|
| after | arm | Similar arm extension movement |
| airplane | aunt, awake | Hand near face in both |
| all | any | Very similar hand shapes |
| alligator | arm | Similar arm motion |
| any | animal, aunt | Similar hand configurations |

### 3. The Paper's Methodology Was Better

The PopSign paper achieved 84.2% accuracy using:
- **Bidirectional LSTM** (not regular LSTM)
- **Hands-only features** (63 features, not 225)
- **Left-hand mirroring** (normalising left-handed signs to look like right-handed)
- **Signer-independent splits** (test signers not seen during training)

V3 used none of these. The gap between 72.80% and 84.2% pointed directly at what to implement in V4.

---

## Files

### Scripts

| File | Description |
|---|---|
| `scripts/train_lstm_popsign.py` | Main training script : loads PopSign data, trains 13-class LSTM, saves model and history |
| `scripts/test_webcam_popsign.py` | Webcam app for 13-sign model : manual record-and-predict interaction |
| `scripts/test_webcam_fixed.py` | Webcam app for 7-sign model : loads weights from `.npy` and rebuilds architecture |
| `scripts/check_weights.py` | Utility : prints number and shape of each weight array in a `.npy` weights file |
| `scripts/quick_test.py` | Sanity check : loads 13-sign model, runs a dummy prediction, verifies probabilities sum to 1 |
| `scripts/test1_for_7_Sign.py` | Prints both 13-sign and 7-sign label maps side by side for comparison |
| `scripts/test2_for_7_Sign.py` | Verifies the 7-sign weight file has the correct output layer shapes `(64, 7)` and `(7,)` |

### Data

| File | Shape | Description |
|---|---|---|
| `data/X_popsign.npy` | (5861, 60, 225) | Feature arrays for all 13 signs |
| `data/Y_popsign.npy` | (5861,) | Integer class labels |
| `data/label_map_popsign.npy` | dict | `{0: 'TV', 1: 'after', ...}` — 13-sign map |
| `data/label_map_popsign_26.npy` | dict | Label map for 26-sign experiment |
| `data/label_map_7signs.npy` | dict | Label map for 7-sign experiment |
| `data/label_map_popsign_7.npy` | dict | Alternative 7-sign label map |

### Models & Saved Artefacts

| File | Description |
|---|---|
| `models/best_model_popsign.keras` | Best 13-sign model (72.80% test accuracy) |
| `models/best_model_popsign_26.keras` | Model from 26-sign experiment |
| `models/popsign_weights.npy` | 13-sign model weights as numpy array |
| `models/popsign_13_weights_fresh.npy` | Re-saved 13-sign weights for cross-platform use |
| `models/popsign_7_weights.npy` | 7-sign model weights |
| `models/final_7_sign_weights.npy` | Final 7-sign weights used by the webcam app |
| `models/final_7_sign.keras` | Full saved 7-sign model |

### Analysis

| File | Description |
|---|---|
| `Analysis/confusion_matrix_13signs.png` | Confusion matrix heatmap for the 13-sign LSTM model |

---

## Comparison: V1 / V2 / V3

| Metric | V1 (WLASL) | V2 (WLASL) | V3 (PopSign) |
|---|---|---|---|
| Dataset | WLASL | WLASL | PopSign |
| Videos per sign | ~20 | ~20 | ~450 |
| Total videos | ~400 | ~400 | 5,861 |
| Sequence length | 40 frames | 40 frames | 60 frames |
| Features | 225 | 225 | 225 |
| Architecture | LSTM | LSTM + Transformer | LSTM + BatchNorm |
| Test accuracy | ~96%* | 57% / 81% | 72.80% |
| Real-world accuracy | 40–47% | inconsistent | ~54% |
| App interaction | Continuous auto | Continuous auto | Manual record |

*V1 had data leakage

---

## Key Learnings

1. **Data quality beats data quantity** : 450 diverse, verified videos per sign outperformed 20 scraped YouTube videos, but even this wasn't enough to overcome visually similar signs.
2. **Sign distinctiveness is the dominant factor** : a sign with 300 training examples but a unique visual form (arm) outperformed signs with 550 examples but ambiguous motions (another, alligator).
3. **The paper's methodology matters** : the PopSign paper's specific choices (BiLSTM, hand-only features, left-hand mirroring) weren't arbitrary; each one addresses a real generalisation problem.
4. **Manual recording outperforms rolling buffer** : giving the model a clean, complete sign to evaluate is more reliable than hoping the automatic trigger captures the right 40-frame window.
5. **225 features may be too many** : pose landmarks add noise for signs that primarily differ in hand shape and movement; hands-only features (63) became the hypothesis to test in V4.

---

## What Changed in V4

The findings here directly shaped V4's design:

- **Bidirectional LSTM** : to match the paper's architecture
- **Hands-only features (63)** : removing pose landmarks to reduce noise
- **Left-hand mirroring** : normalising left-handed signs to right-hand perspective
- **Expanded vocabulary** : moving toward more signs with better feature selection

[Continue to Version 4 →](../v4-popsign-bilstm/V4_README.md)

---

## References

1. Thad Starner et al., *"PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones"*, NeurIPS 2023 Datasets and Benchmarks Track
2. Dataset website: https://signdata.cc.gatech.edu
3. Kaggle competition: https://www.kaggle.com/competitions/asl-signs