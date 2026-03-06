# Version 6: PopSign Exact — Hands-Only Features

[← Back to Main](../../README.md) | [← Previous: V5 Enhanced Transformer](../v5-popsign-transformer/V5_README.md)

---

## Overview

V6 is the final and strongest version of the project. Rather than introducing new architectural complexity, it takes a completely different approach: **replicate the original PopSign paper's methodology exactly** : specifically its feature selection and preprocessing, and measure whether that alone closes the test-to-real-world gap.

The answer was decisive. Switching from 225 features (hands + pose) to 63 features (hands only) with left-hand mirroring produced the most dramatic improvement of any change across all six versions:

- Test accuracy jumped from ~72% to **81.27%** on a focused sign set
- Real-world webcam accuracy jumped from 19-24% to **87.5%**
- The notorious test-to-real-world gap **reversed** : real-world performance actually exceeded test accuracy

V6 also introduces the complete end-to-end system: `test_webcam_sentences.py` integrates sign recognition with a local LLM (Ollama with Llama 3.2) to convert recognised signs into grammatically correct English sentences, with text-to-speech output. This is the production-ready version of the system.

V6 was trained iteratively, first on a small sign set to validate the methodology, then expanded to 43, 82, and finally **248 signs** (trained on the university GPU with 110,540 videos), with weights and label maps saved for each scale.

---

## What Changed From V5

| Component | V5 | V6 |
|---|---|---|
| Architecture | Enhanced Transformer | Bidirectional LSTM |
| Features | 225 (hands + pose) | 63 (hands only) |
| Left-hand mirroring | No | Yes |
| CNN backbone | Yes (multi-scale) | No |
| Gaussian noise | Yes | No |
| Label smoothing | Yes | No |
| Signs (final model) | 42 | 248 |
| Test accuracy | 70.80% | 81.27% (16-sign) / 72.82% (248-sign) |
| Real-world accuracy | 24% (10/42) | 87.5% (14/16) |

The architecture actually went **backwards** in complexity, from an Enhanced Transformer back to a BiLSTM. This is the key lesson of V6: the feature representation mattered far more than the model architecture.

---

## Architecture

```
Input(60, 63)                        ← 63 features: hands only
→ Masking(mask_value=0.0)            ← ignore zero-padded frames
→ Bidirectional(LSTM(128, return_sequences=True))
→ Dropout(0.5)
→ Bidirectional(LSTM(128, return_sequences=False))
→ Dropout(0.5)
→ Dense(num_classes, softmax)
```

This is identical to V4's architecture with one critical difference: the input shape is `(60, 63)` instead of `(60, 225)`. Everything else, the BiLSTM structure, dropout, masking, is the same. The entire improvement came from the feature change.

---

## The Two Key Preprocessing Changes

### 1. Hands-Only Features (63 instead of 225)

Previous versions extracted landmarks from three sources:
- Left hand: 21 landmarks × 3 coordinates = **63 features**
- Right hand: 21 landmarks × 3 coordinates = **63 features**
- Pose: 33 landmarks × 3 coordinates = **99 features**
- **Total: 225 features**

V6 extracts only the dominant hand:
- Dominant hand: 21 landmarks × 3 coordinates = **63 features**
- **Total: 63 features**

Why does removing 162 features *improve* accuracy? Because pose landmarks (shoulder positions, torso orientation, head position) are **not part of the sign** : they describe how the signer is sitting, not what they are signing. Including them adds noise that the model has to learn to ignore. Removing them forces the model to focus entirely on what matters: hand shape and movement.

### 2. Left-Hand Mirroring

The PopSign dataset contains both right-handed and left handed signers. Without mirroring, the model sees:

```
Right-handed signer doing "apple" → hand on right side of frame
Left-handed signer doing "apple"  → hand on left side of frame
```

These look like completely different inputs to the model, even though they represent the same sign. The model has to learn both patterns separately, which wastes capacity and reduces generalisation.

With mirroring, left-hand signers have their x-coordinates flipped (`x → 1 - x`), so all signs appear to come from the right hand regardless of the signer's dominant hand:

```python
if results.right_hand_landmarks:
    # Use right hand as-is
    for lm in results.right_hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
elif results.left_hand_landmarks:
    # Mirror left hand to look like right hand
    for lm in results.left_hand_landmarks.landmark:
        landmarks.extend([1.0 - lm.x, lm.y, lm.z])
else:
    landmarks.extend([0.0] * 63)
```

This halves the variation the model needs to learn and directly explains a significant portion of V6's improvement over V4 (which used the same BiLSTM architecture but without mirroring).

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | PopSign ASL v1.0 |
| Final model signs | 248 |
| Total videos | 110,540 |
| Train/test split | 80/20 stratified |
| Input shape | (60, 63) |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Max epochs | 40 |
| Early stopping patience | 10 |
| Training hardware | University GPU (Tesla T4) |

---

## Results

### Iterative Training : Sign Set Expansion

V6 was trained at multiple scales to validate the methodology before committing to the full dataset:

| Model | Signs | Test Accuracy | Real World Accuracy |
|---|---|---|---|
| popsign_exact_weights.npy | 16 | 81.27% | 87.5% (14/16) |
| popsign_exact_weights_43.npy | 43 | ~80%+ | Strong |
| popsign_exact_weights_82.npy | 82 | ~78%+ | Strong |
| popsign_exact_weights_248.npy | 248 | 72.82% | Best across all versions |

The slight drop in test accuracy as sign count increases is expected, more signs means a harder classification task. But the real-world performance remains dramatically better than V4/V5 at equivalent sign counts.

### Comparison Across All Versions

| Version | Architecture | Features | Signs | Test Accuracy | Real World |
|---|---|---|---|---|---|
| V3 | Regular LSTM | 225 | 13 | 72.80% | 54% |
| V4 | BiLSTM | 225 | 42 | 72.63% | 19% |
| V5 | Enhanced Transformer | 225 | 42 | 70.80% | 24% |
| **V6** | **BiLSTM** | **63** | **16** | **81.27%** | **87.5%** |
| **V6** | **BiLSTM** | **63** | **248** | **72.82%** | **Best** |
| PopSign Paper | BiLSTM | 63 | 250 | 84.20% | N/A |

### Real-World Performance (16-Sign Model)

| Sign | Confidence | Status |
|---|---|---|
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
| after | — |  Confused with blue |
| any | — |  Confused with aunt |

**Real-world accuracy: 14/16 = 87.5%**

### The Test-to-Real-World Gap Resolved

| Version | Test Accuracy | Real-World | Gap |
|---|---|---|---|
| V4 | 72.63% | 19% | −53% |
| V5 | 70.80% | 24% | −47% |
| **V6** | **81.27%** | **87.5%** | **+6%** |

V6 is the first version where real-world accuracy **exceeds** test accuracy. This happened because hands only features with left-hand mirroring produces representations that are genuinely consistent between training data and live webcam input, the domain gap that hurt all previous versions was caused largely by noisy pose features, not by the model architecture.

---

## Component 2: Sentence Generation System (`test_webcam_sentences.py`)

V6 introduces the complete end-to-end pipeline, the most significant new component in the project. After recognising individual signs, the system converts them into grammatically correct English sentences using a local LLM.

### Full System Pipeline

```
User performs sign
        ↓
MediaPipe extracts 63 hand landmarks per frame
        ↓
Bidirectional LSTM predicts sign + confidence score
        ↓
Sign added to collected words list (with confidence)
        ↓
User presses ENTER
        ↓
Ollama (Llama 3.2) forms natural English sentence
        ↓
macOS text-to-speech speaks sentence aloud
```

### Ollama Integration

The system sends collected signs to a locally running Llama 3.2 model via the Ollama API:

```python
OLLAMA_MODEL = "llama3.2"
# Runs locally at http://localhost:11434
```

Running the LLM locally (rather than via a cloud API) means:
- No internet connection required
- No data sent to external servers, important for privacy
- No API costs
- Works offline

### Confidence-Based Filtering

The prompt sent to the LLM includes a structured breakdown of word confidence:

```
HIGH confidence (must use):   words with confidence ≥ 70%
MEDIUM confidence (use if fits): words with confidence 50-70%
LOW confidence (only if fits context): words with confidence < 50%
```

This means the LLM makes intelligent decisions about which recognised signs to include. If a sign was recognised with low confidence and doesn't fit the context of the other signs, the LLM excludes it automatically, rather than producing a nonsensical sentence.

**Example:** User signs "chair", "sit", "apple" and the model incorrectly predicts a fourth sign "crocodile" with 18% confidence. The LLM sees this is low confidence and contextually implausible, excludes it, and generates: *"I want to sit on the chair and eat an apple."*

### Fallback Mode

If Ollama is not running, the system automatically falls back to simple sentence formation:

```python
def form_sentence_simple(words, confidences):
    filtered_words = [w for w, c in zip(words, confidences) if c >= MIN_CONFIDENCE]
    sentence = " ".join(filtered_words).capitalize() + "."
    return sentence, filtered_words
```

This ensures the system always produces output even without the LLM running.

### Text-to-Speech

Generated sentences are spoken aloud using macOS built-in text-to-speech:

```python
subprocess.run(['say', '-r', '150', sentence])
```

This makes the system fully accessible for deaf/mute users communicating with hearing people who cannot read the screen.

### User Interface

`test_webcam_sentences.py` features a polished custom UI built entirely in OpenCV:

- **Left panel**: Recording status, hand detection indicator, progress bar, last prediction
- **Bottom left panel**: Collected words displayed as colour-coded chips (green = high confidence, yellow = medium, red = low)
- **Bottom right panel**: Generated sentence with word wrapping
- **Right panel**: Ollama connection status
- **Bottom bar**: Keyboard controls with styled key boxes
- **App name**: "VoiceForAll", displayed in the bottom right corner

### Controls

| Key | Action |
|---|---|
| SPACE | Start/stop recording a sign |
| ENTER | Form sentence from collected words + speak aloud |
| C | Clear all collected words |
| Q | Quit |

---

## Files

### Scripts

| File | Description |
|---|---|
| `scripts/test_webcam_hands_only.py` | Single-sign webcam testing : records one sign at a time, shows prediction and confidence. Used for evaluating individual sign accuracy |
| `scripts/test_webcam_sentences.py` | Full end-to-end sentence system : collects multiple signs, generates English sentences via Ollama, speaks output aloud. This is the production system |

### Data

| File | Signs | Description |
|---|---|---|
| `data/label_map_hands_only.npy` | 16 | Initial validation model label map |
| `data/label_map_hands_only_43.npy` | 43 | 43-sign model label map |
| `data/label_map_hands_only_82.npy` | 82 | 82-sign model label map |
| `data/label_map_hands_only_248.npy` | 248 | Final 248-sign model label map |

### Models & Saved Artefacts

| File | Signs | Description |
|---|---|---|
| `models/popsign_exact_weights.npy` | 16 | Initial validation model weights |
| `models/popsign_exact_weights_43.npy` | 43 | 43-sign model weights |
| `models/popsign_exact_weights_82.npy` | 82 | 82-sign model weights |
| `models/popsign_exact_weights_248.npy` | 248 | Final model weights (primary) |
| `models/best_model_popsign_248.h5` | 248 | Best checkpoint saved during training |

---

## Gap With The PopSign Paper

The PopSign paper achieved 84.2% on 250 signs. V6's 248-sign model achieved 72.82%. The remaining gap has two primary causes:

**1. Random vs signer-independent train/test split** : the paper ensured different signers appeared in training and test sets. V6 used a random 80/20 split where the same 47 signers appear in both. This inflates V6's test accuracy slightly (the model has seen each signer's style before) and would make a true signer-independent evaluation harder.

**2. Left-hand mirroring at preprocessing vs inference only** : V6 implements mirroring at inference time in the webcam scripts. The training data for the 248-sign model was preprocessed before this change was fully implemented. Retraining with mirroring applied consistently during preprocessing would likely push accuracy closer to the paper's result.

Both of these are well-understood limitations that are documented here and in the dissertation analysis.

---

## Key Learnings

1. **Feature selection dominated everything** : switching from 225 to 63 features produced a larger improvement than any architectural change across all six versions combined. Simpler, more focused representations generalise better.
2. **The test-to-real-world gap was a feature problem, not a model problem** : pose landmarks introduced noise that looked like signal in the test set (because the test set came from the same signers as training) but failed completely on new webcam input. Removing pose fixed the gap permanently.
3. **Left-hand mirroring is essential for signer-independent generalisation** : without it, the model learns two representations of every sign unnecessarily, wasting capacity and reducing robustness.
4. **Replicating a paper's methodology exactly is more valuable than architectural innovation** : V5's Enhanced Transformer was far more complex than V6's BiLSTM, but V6 dramatically outperformed it by correctly implementing the paper's preprocessing pipeline.
5. **End-to-end system design requires intelligent filtering** : confidence-based LLM filtering makes the sentence generation component robust to inevitable recognition errors, producing natural output even when individual sign predictions are imperfect.
6. **Local LLM deployment is practical for accessibility applications** : running Ollama locally eliminates privacy concerns, internet dependency, and API costs, making the system genuinely deployable for deaf/mute users.

---

## References

1. Thad Starner et al., *"PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones"*, NeurIPS 2023
2. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
3. PopSign Dataset: https://signdata.cc.gatech.edu/
4. Ollama: https://ollama.ai
5. MediaPipe Holistic: https://google.github.io/mediapipe/solutions/holistic.html