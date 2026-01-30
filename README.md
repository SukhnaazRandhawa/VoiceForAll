# Bidirectional AI-Powered Sign Language Translation System

**Author:** Sukhnaaz Kaur

**Supervisor:** Martin Escardo

**University of Birmingham — BSc Artificial Intelligence and Computer Science**

---

## Project Overview

This project develops a real-time sign language recognition system aimed at bridging communication between deaf/mute individuals and hearing people. The system uses computer vision and deep learning to recognize American Sign Language (ASL) gestures from webcam input.

The project evolved through multiple iterations, each improving upon the previous through architectural changes and methodological fixes.

---

## Version History

| Version | Description | Test Accuracy | Key Learning |
|---------|-------------|---------------|--------------|
| [V1 - LSTM Baseline](versions/v1-lstm-baseline/README.md) | Initial LSTM implementation | 40-47% | Established baseline, discovered data leakage issue |
| [V2 - Transformer](versions/v2-transformer/README.md) | Proper evaluation + Transformer architecture | 57% (LSTM) → 81% (Transformer) | Attention mechanism significantly improves recognition |

---

## Repository Structure
```
FinalProject/
├── versions/
│   ├── v1-lstm-baseline/       # Initial LSTM approach
│   └── v2-transformer/         # Transformer with proper evaluation
├── shared-data/                # Large dataset files (WLASL-2000)
├── data/                       # Raw WLASL data and metadata
├── literature_review/          # Research papers and notes
├── Weekly_Report/              # Progress documentation
└── README.md                   # This file
```

---

## Dataset

This project uses the **WLASL-2000** dataset:
- 21,083 videos covering 2,000 ASL words
- Features extracted using MediaPipe (225 landmarks per frame)
- Current experiments focus on top 20 words with sufficient samples

The processed dataset files are located in `shared-data/`.

---

## Quick Start

Each version folder contains its own README with specific setup instructions. Start with:

1. **[Version 1](versions/v1-lstm-baseline/README.md)** — To understand the baseline approach
2. **[Version 2](versions/v2-transformer/README.md)** — For the current best-performing model and real-time demo

---

## Technologies

- **MediaPipe** — Hand and pose landmark extraction
- **TensorFlow/Keras** — Model training (LSTM, Transformer)
- **OpenCV** — Real-time video capture
- **Python 3.11**

---

## Report

For detailed methodology, evaluation, and analysis, please refer to the dissertation report.
