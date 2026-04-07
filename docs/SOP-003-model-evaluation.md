# SOP-003: Model Evaluation

**Project:** VoiceForAll — ASL Sign Language to Speech Translation  
**Version:** All Versions (V1–V6)  
**Last Updated:** April 2026  
**Author:** Sukhnaaz Kaur

---

## Purpose

This SOP defines the standard procedure for evaluating VoiceForAll models. It covers two types of evaluation used throughout the project: test set evaluation (quantitative) and real-world webcam evaluation (qualitative). Both are required for a complete assessment of model performance.

---

## Two Types of Evaluation

| Type | Method | What it measures |
|---|---|---|
| Test set evaluation | Run model on held-out data | Statistical accuracy on PopSign distribution |
| Real-world evaluation | Live webcam testing | Practical accuracy on new signer, new conditions |

Both are necessary. Test accuracy alone was shown to be misleading across V1–V5, where test accuracy of ~72% consistently translated to real-world accuracy of 19–54%.

---

## Type 1: Test Set Evaluation

### Step 1 — Load model and test data
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking, Input

# Load label map
label_map = np.load('data/label_map_hands_only_248.npy', allow_pickle=True).item()
NUM_CLASSES = len(label_map)

# Rebuild model architecture
model = Sequential([
    Input(shape=(60, 63)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Load trained weights
weights = np.load('models/popsign_exact_weights_248.npy', allow_pickle=True)
model.set_weights(list(weights))
```

### Step 2 — Run evaluation
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
print(f"Test accuracy: {accuracy * 100:.2f}%")
```

### Expected results

| Model | Signs | Expected test accuracy |
|---|---|---|
| popsign_exact_weights.npy | 16 | ~81.27% |
| popsign_exact_weights_248.npy | 248 | ~72.82% |

---

## Type 2: Real-World Webcam Evaluation

### Step 1 — Start Ollama (optional)
```bash
ollama serve
```

### Step 2 — Run the single-sign evaluation script
```bash
cd versions/v6-popsign-hands-only/scripts
python test_webcam_hands_only.py
```

### Step 3 — Test each sign systematically

For each sign being evaluated:

1. Press `SPACE` to start recording
2. Perform the sign clearly, facing the camera
3. Press `SPACE` to stop recording
4. Record the predicted sign and confidence score
5. Mark as pass or fail

### Scoring criteria

| Result | Condition |
|---|---|
| Pass | Correct sign predicted at any confidence |
| Partial | Correct sign in top-3 predictions |
| Fail | Correct sign not predicted |

### Real-world accuracy formula
---

## Benchmark Results Across All Versions

| Version | Architecture | Features | Signs | Test Accuracy | Real-World |
|---|---|---|---|---|---|
| V1 | LSTM | 225 | 20 | ~96%* | ~40–47% |
| V2 | Transformer | 225 | 20 | 81% | Inconsistent |
| V3 | LSTM | 225 | 13 | 72.80% | 54% |
| V4 | BiLSTM | 225 | 42 | 72.63% | 19% |
| V5 | Enhanced Transformer | 225 | 42 | 70.80% | 24% |
| **V6** | **BiLSTM** | **63** | **248** | **72.82%** | **Best** |

*V1 had data leakage — result is not comparable

---

## Interpreting Results

### The test-to-real-world gap

A large gap between test accuracy and real-world accuracy indicates one or more of the following:

| Cause | Indicator | Fix applied in |
|---|---|---|
| Noisy features | Gap persists across architectures | V6 (hands-only) |
| Data leakage | Test accuracy unusually high | V2 (split fixed) |
| Domain shift | Works on dataset, fails on webcam | V6 (mirroring) |
| Visual similarity | Specific signs confused consistently | Documented in failure analysis |

### Confidence thresholds

| Confidence | Interpretation | Action |
|---|---|---|
| ≥ 70% | High confidence | Always include in sentence |
| 50–70% | Medium confidence | Include if contextually appropriate |
| 30–50% | Low confidence | LLM decides based on context |
| < 30% | Below floor | Discard |

---

## Verification Checklist

Before reporting evaluation results, confirm:

- [ ] Test data was not used during training
- [ ] Train/test split was stratified by class
- [ ] Model weights loaded without error
- [ ] Real-world testing used a different signer from training data where possible
- [ ] Both test accuracy and real-world accuracy are reported
- [ ] Failure cases are documented with predicted vs actual sign

