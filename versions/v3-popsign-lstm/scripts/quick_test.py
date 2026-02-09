import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

# Build model
model = Sequential([
    Input(shape=(60, 225)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(13, activation='softmax')
])

# Load weights
weights = np.load('../models/popsign_13_weights_fresh.npy', allow_pickle=True)
model.set_weights(list(weights))
print(" Weights loaded successfully!")

# Load label map
label_map = np.load('../data/label_map_popsign.npy', allow_pickle=True).item()
print(f" Label map loaded: {len(label_map)} signs")
print(f"   Signs: {list(label_map.values())}")

# Test with random input
test_input = np.random.randn(1, 60, 225)
prediction = model.predict(test_input, verbose=0)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class]

print(f"\n Test prediction works!")
print(f"   Predicted class: {predicted_class} ({label_map[predicted_class]})")
print(f"   Confidence: {confidence*100:.1f}%")
print(f"   All probabilities sum to: {prediction.sum():.4f}")