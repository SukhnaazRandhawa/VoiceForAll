# V5: Enhanced Transformer for PopSign ASL Recognition

## Overview

V5 implements an Enhanced Transformer architecture for American Sign Language (ASL) recognition using the PopSign ASL v1.0 dataset. This version incorporates advanced techniques including label smoothing, Gaussian noise augmentation, multi scale CNN feature extraction, and combined pooling strategies.

## Model Architecture

### Enhanced Transformer

```
Input (60, 225)
    │
    ▼
┌─────────────────────────────────────┐
│       Gaussian Noise (0.1)          │  ← Real-world stability
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Multi-Scale CNN Backbone        │
│  ┌─────────────┬─────────────┐      │
│  │ Conv1D k=3  │ Conv1D k=5  │      │  ← Fast + slow movements
│  │ Conv1D k=3  │ Conv1D k=5  │      │
│  └──────┬──────┴──────┬──────┘      │
│         │  Concatenate │             │
│         └──────┬───────┘             │
└────────────────┼────────────────────┘
                 │ (128 features)
    ▼
┌─────────────────────────────────────┐
│    Positional Embedding (128)       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Transformer Block 1            │
│  • MultiHeadAttention (8 heads)     │
│  • Feed-Forward (256 → 128)         │
│  • LayerNorm + Residual             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Transformer Block 2            │
│  • MultiHeadAttention (8 heads)     │
│  • Feed-Forward (256 → 128)         │
│  • LayerNorm + Residual             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       Combined Pooling              │
│  GlobalAvgPool + GlobalMaxPool      │  ← Capture average + peak
│         (256 features)              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Classification Head            │
│  Dense(128) → Dropout(0.3)          │
│  Dense(64)  → Dropout(0.3)          │
│  Dense(42, softmax)                 │
└─────────────────────────────────────┘
```

## Key Innovations

### 1. Label Smoothing (0.1)
- Prevents model overconfidence
- Distributes 10% probability across all classes
- Improves generalization to new signers

### 2. Gaussian Noise Augmentation
- Adds noise during training (σ = 0.1)
- Simulates real-world webcam jitter
- Makes model robust to imperfect landmark detection

### 3. Multi-Scale CNN Backbone
- **Kernel size 3**: Captures fast, short-term movements
- **Kernel size 5**: Captures slow, long-term movements
- Concatenated features provide richer temporal representation

### 4. Combined Pooling Strategy
- **Global Average Pooling**: Captures overall sign pattern
- **Global Max Pooling**: Captures peak moments of sign
- Concatenation preserves both average and distinctive features

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | PopSign ASL v1.0 |
| Signs | 42 classes |
| Total samples | 14,483 videos |
| Train/Test split | 80/20 (stratified) |
| Input shape | (60 frames, 225 features) |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Categorical Crossentropy (label_smoothing=0.1) |
| Epochs | 100 (early stopping patience=25) |

## Results

### Test Accuracy
| Metric | Value |
|--------|-------|
| Test Accuracy | 70.80% |
| Training Accuracy | 71.0% |
| Train-Val Gap | ~0.2% (minimal overfitting) |

### Comparison with Previous Versions

| Version | Architecture | Test Accuracy | Real-World Working Signs |
|---------|--------------|---------------|--------------------------|
| V3 | Regular LSTM | 71.20% | ~7/13 (54%) |
| V4 | Bidirectional LSTM | 72.63% | 8/42 (19%) |
| **V5** | **Enhanced Transformer** | **70.80%** | **10/42 (24%)** |

### Real-World Performance

#### Signs That Work Reliably (10 signs)
| Sign | Confidence | Notes |
|------|------------|-------|
| arm | 95% | Excellent |
| bed | 96% | Excellent |
| blue | 87% | Very good |
| apple | 83% | Very good |
| bath | 80% | Good |
| bye | 77% | Good |
| cheek | 74% | Good |
| brother | 72% | Good |
| cloud | 64% | Moderate |
| awake | 53% | Lower confidence but works |

#### Signs That Work Sometimes (5 signs)
- balloon (33%)
- because (26%)
- chair (17%)
- closet (43%)
- alligator (variable)

#### New Signs Working in V5 (Not in V4)
-  awake
-  bath
-  cheek
-  cloud

## Feature Extraction

Features are extracted using MediaPipe Holistic:
- **Left hand**: 21 landmarks × 3 coordinates = 63 features
- **Right hand**: 21 landmarks × 3 coordinates = 63 features
- **Pose**: 33 landmarks × 3 coordinates = 99 features
- **Total**: 225 features per frame

## File Structure

```
v5-popsign-transformer/
├── data/
│   └── label_map_popsign_42.npy      # Sign name mappings
├── models/
│   └── transformer_enhanced_42_weights.npy  # Trained weights
├── scripts/
│   └── test_webcam_transformer.py    # Real-time testing
└── V5_README.md                      # This file
```

## Usage

### Real-Time Testing
```bash
cd v5-popsign-transformer/scripts
python test_webcam_transformer.py
```

### Controls
- **SPACE**: Start/Stop recording
- **Q**: Quit

## Key Findings

### Why Lower Test Accuracy = Better Real-World Performance

1. **No Overfitting**: Train (71.0%) ≈ Val (70.8%) gap is minimal
2. **Label Smoothing**: Model makes "soft" predictions, less brittle
3. **Noise Training**: Model learned through "blurry" data, handles real webcam better
4. **Generalization**: Learned sign patterns, not dataset-specific features

### Technical Limitations Identified

| Limitation | Affected Signs | Reason |
|------------|----------------|--------|
| 2D camera | cloud, alligator, blow | Cannot capture depth movement |
| Hand-face occlusion | ear, eye, carrot | MediaPipe loses landmarks |
| Similar signs | bed/bedroom, black/because | Fine distinctions difficult |
| Signer variation | All signs | Model trained on 47 Deaf signers |

### Confusion Patterns

Signs that consistently confuse each other:
- **Face-touching**: ear ↔ cheek ↔ eye ↔ apple
- **Similar motion**: black ↔ because, bed ↔ bedroom
- **Depth-based**: cloud ↔ alligator ↔ balloon

## Dissertation Insights

### Research Contributions

1. **Generalization vs Accuracy Trade-off**: Lower test accuracy with regularization techniques (label smoothing, noise) can yield better real-world performance.

2. **Multi-Scale Temporal Processing**: Combining short-term (kernel=3) and long-term (kernel=5) CNN features improves sign recognition for signs with varying speeds.

3. **Real-World Gap Analysis**: 70.80% test accuracy → 24% real-world accuracy highlights the domain shift between dataset and live webcam conditions.

4. **Technical Limitations Documentation**: Identified specific failure modes (2D depth, occlusion, signer variation) that inform future system design.

## Future Improvements

1. **Fine-tune on personal recordings**: Adapt model to specific signer
2. **Ensemble V4 + V5**: Combine BiLSTM and Transformer predictions
3. **Focus on 2D-friendly signs**: Build vocabulary avoiding depth-dependent signs
4. **Data augmentation**: Add more variation during training

## Dependencies

```
tensorflow >= 2.10
mediapipe >= 0.9
opencv-python >= 4.5
numpy >= 1.21
```

## References

- PopSign ASL Dataset: NeurIPS 2023
- Google ASL Signs Kaggle Competition
- MediaPipe Holistic: Google AI



