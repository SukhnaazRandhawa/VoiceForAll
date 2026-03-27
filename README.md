# VoiceForAll : Bidirectional AI Sign Language Translation System

> *Giving voice to people who cannot speak*

A final year BSc dissertation project at the **University of Birmingham**, Department of Artificial Intelligence and Computer Science.

---

## What Is This?

VoiceForAll is a real time, bidirectional sign language translation system designed to help deaf and mute individuals communicate more easily with hearing people. It uses a laptop camera to detect hand movements, recognises American Sign Language (ASL) signs using a deep learning model, converts them into grammatically correct English sentences using a local large language model, and speaks them aloud, all in real time, with no internet connection required.

The system was built iteratively across **six versions**, each testing a specific architectural or methodological hypothesis. The journey from V1 to V6 tells the story of how the right feature representation matters more than model complexity.

---

## System Components

The complete system integrates three core components:

```
SIGN RECOGNITION          SENTENCE FORMATION         SPEECH OUTPUT
─────────────────         ──────────────────         ─────────────
Laptop camera             Ollama (Llama 3.2)         macOS say
     ↓                    running locally                  ↑
MediaPipe Holistic   →    Confidence-based      →    English sentence
     ↓                    NLP filtering              spoken aloud
63 hand landmarks
     ↓
Bidirectional LSTM
     ↓
Sign prediction +
confidence score
```

---

## Project Structure

```
FinalProject/
├── README.md                          ← You are here
├── versions/
│   ├── v1-lstm-baseline/              ← WLASL dataset, regular LSTM, 20 signs
│   │   └── V1_README.md
│   ├── v2-transformer/                ← Transformer architecture comparison
│   │   └── V2_README.md
│   ├── v3-popsign-lstm/               ← PopSign dataset, 13 signs
│   │   └── V3_README.md
│   ├── v4-popsign-bilstm/             ← Bidirectional LSTM, 42 signs
│   │   └── V4_README.md
│   ├── v5-popsign-transformer/        ← Enhanced Transformer, 42 signs
│   │   └── V5_README.md
│   └── v6-popsign-hands-only/         ← Final system, 248 signs 
│       └── V6_README.md
└── Weekly_Report/                     ← Supervisor progress reports
```

---

## Version History at a Glance

| Version | Architecture | Dataset | Features | Signs | Test Accuracy | Real-World |
|---|---|---|---|---|---|---|
| [V1](versions/v1-lstm-baseline/V1_README.md) | Regular LSTM | WLASL-2000 | 225 | 20 | ~96% | Limited |
| [V2](versions/v2-transformer/V2_README.md) | Transformer | WLASL-2000 | 225 | 20 | ~81% | Limited |
| [V3](versions/v3-popsign-lstm/V3_README.md) | Regular LSTM | PopSign ASL v1.0 | 225 | 13 | 72.80% | 54% |
| [V4](versions/v4-popsign-bilstm/V4_README.md) | Bidirectional LSTM | PopSign ASL v1.0 | 225 | 42 | 72.63% | 19% |
| [V5](versions/v5-popsign-transformer/V5_README.md) | Enhanced Transformer | PopSign ASL v1.0 | 225 | 42 | 70.80% | 24% |
| [**V6**](versions/v6-popsign-hands-only/V6_README.md) | **Bidirectional LSTM** | **PopSign ASL v1.0** | **63** | **248** | **81.27% / 72.82%** | **87.5%**  |

---

## The Central Finding

**The feature representation mattered more than the model architecture.**

Across V1 to V5, every version used 225 features, hand landmarks, pose landmarks, and body position data combined. Despite architectural changes (regular LSTM → Transformer → BiLSTM → Enhanced Transformer), the test-to-real-world gap persisted at 47-53%.

V6 changed one thing: it dropped 162 features and kept only the 63 hand landmarks. Test accuracy jumped from ~72% to 81.27% on a focused sign set, and real world accuracy jumped from 19-24% to **87.5%**. The gap did not just close, it reversed. Real world accuracy exceeded test accuracy for the first time.

The explanation: pose landmarks (shoulder position, torso orientation, head position) describe *how the signer is sitting*, not *what they are signing*. They added noise that looked like signal in the test set but failed completely on new webcam input.

---

## Key Learnings Across All Versions

**V1 → V2:** Transformers outperformed LSTMs on WLASL (81% vs 57% honest baseline), a genuine architectural improvement. However, strong test accuracy still did not translate to reliable real-world webcam performance, suggesting the bottleneck was the input representation, not the architecture.

**V2 → V3:** Switching from WLASL to the PopSign dataset (NeurIPS 2023) dramatically improved data quality and real-world relevance. The dataset was recorded on smartphones by 47 deaf signers, much closer to real deployment conditions.

**V3 → V4:** Bidirectional context genuinely helped. Reading sign sequences both forward and backward lets the model use future context to disambiguate ambiguous mid-sign frames. However, the test-to-real-world gap worsened as sign count increased from 13 to 42.

**V4 → V5:** Aggressive regularisation (label smoothing, Gaussian noise, multi-scale CNN backbone) improved real-world performance slightly despite lower test accuracy. The gap between test and real-world performance was a feature problem, not a regularisation problem.

**V5 → V6:** Removing pose landmarks and adding left-hand mirroring solved the test-to-real-world gap entirely. Architectural complexity was irrelevant, the BiLSTM from V4 outperformed the Enhanced Transformer from V5 once given the right features.

---

## Final System (V6) : Quick Start

### Requirements

```bash
# Python packages
pip install tensorflow mediapipe opencv-python numpy scikit-learn requests

# For sentence generation (optional but recommended)
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```

### Run Sign Recognition Only

```bash
cd versions/v6-popsign-hands-only/scripts
python test_webcam_hands_only.py
```

**Controls:** `SPACE` to record a sign, `Q` to quit.

### Run Full System (Signs → Sentences → Speech)

```bash
# First, start Ollama in a separate terminal
ollama serve

# Then run the full system
cd versions/v6-popsign-hands-only/scripts
python test_webcam_sentences.py
```

**Controls:** `SPACE` to record a sign, `ENTER` to generate sentence and speak, `C` to clear, `Q` to quit.

---

## Dataset

**PopSign ASL v1.0** (NeurIPS 2023, Georgia Tech)

- 250 ASL signs recorded via smartphones
- 47 deaf signers across diverse demographics
- ~110,540 videos total; 248 signs used in V6
- Available at: https://signdata.cc.gatech.edu/

This dataset was chosen because it was recorded in naturalistic conditions by actual deaf signers using smartphones  conditions that closely match real-world deployment on a laptop webcam.

---

## Model Architecture (V6 Final)

```
Input(60 frames × 63 features)
→ Masking(mask_value=0.0)
→ Bidirectional(LSTM(128, return_sequences=True))
→ Dropout(0.3)
→ Bidirectional(LSTM(128, return_sequences=False))
→ Dropout(0.3)
→ Dense(248, softmax)
```

| Parameter | Value |
|---|---|
| Input features | 63 (right hand, or left hand mirrored) |
| Sequence length | 60 frames |
| LSTM units | 128 per direction (256 total per layer) |
| Training samples | 88,432 |
| Test samples | 22,108 |
| Optimizer | Adam |
| Batch size | 32 |
| Training hardware | University GPU (Tesla T4) |

---

## Results Summary

### Test Accuracy : All Versions

| Version | Signs | Test Accuracy |
|---|---|---|
| V3 | 13 | 72.80% |
| V4 | 42 | 72.63% |
| V5 | 42 | 70.80% |
| V6 (focused) | 16 | **81.27%** |
| V6 (full) | 248 | **72.82%** |
| PopSign Paper | 250 | 84.20% |

### Real-World Webcam Accuracy, V6 (16-sign model)

14 out of 16 signs working reliably = **87.5% real-world accuracy**

### Gap Closed

| Version | Test | Real-World | Gap |
|---|---|---|---|
| V4 | 72.63% | 19% | −53% |
| V5 | 70.80% | 24% | −47% |
| **V6** | **81.27%** | **87.5%** | **+6%**  |

---

## Why Not 84%? (Comparison with PopSign Paper)

The original PopSign paper achieved 84.2% on 250 signs. The V6 248 sign model achieved 72.82%. Two specific methodological differences explain this gap:

**1. Train/test split:** The paper used a signer-independent split, different signers in training and test sets. V6 used a random 80/20 split where the same 47 signers appear in both. This means V6's model has seen each signer's style during training, slightly inflating test accuracy while making it harder to generalise to truly new signers.

**2. Preprocessing consistency:** Left-hand mirroring is fully implemented at inference time in the V6 webcam scripts. Applying it consistently during training data preprocessing would likely push accuracy closer to the paper's result.

Both of these are well understood, documented limitations not implementation errors.

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| MediaPipe Holistic | Real-time hand landmark extraction |
| TensorFlow / Keras | Model training and architecture |
| Ollama (Llama 3.2) | Local LLM for sentence formation |
| macOS `say` | Text-to-speech output |
| PopSign ASL v1.0 | Primary training dataset |
| WLASL-2000 | Dataset used in V1 and V2 |
| University GPU (Tesla T4) | Training V6 248-sign model |
| Google Colab (T4) | Training earlier versions |

---

## Version READMEs

Each version has its own detailed README explaining the hypothesis, architecture, results, and key learnings:

- [V1 README](versions/v1-lstm-baseline/V1_README.md) — LSTM baseline on WLASL, data leakage discovery
- [V2 README](versions/v2-transformer/V2_README.md) — Transformer vs LSTM comparison, fixed data leakage
- [V3 README](versions/v3-popsign-lstm/V3_README.md) — PopSign dataset introduction, webcam testing
- [V4 README](versions/v4-popsign-bilstm/V4_README.md) — Bidirectional LSTM, 42 signs, gap analysis
- [V5 README](versions/v5-popsign-transformer/V5_README.md) — Enhanced Transformer, regularisation study
- [V6 README](versions/v6-popsign-hands-only/V6_README.md) — Final system, 248 signs, full pipeline 

---

## References

1. Starner et al., *"PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones"*, NeurIPS 2023
2. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
3. Hochreiter & Schmidhuber, *"Long Short-Term Memory"*, Neural Computation 1997
4. Camgoz et al., *"Sign Language Transformers"*, CVPR 2020
5. PopSign Dataset: https://signdata.cc.gatech.edu/
6. Ollama: https://ollama.ai
7. MediaPipe: https://google.github.io/mediapipe/

---

## Author

**Sukhnaaz**

BSc Artificial Intelligence and Computer Science

University of Birmingham