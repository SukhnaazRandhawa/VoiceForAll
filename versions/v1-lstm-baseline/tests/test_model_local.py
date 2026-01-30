# tests/test_model_local.py

import numpy as np
from tensorflow import keras

print("=" * 60)
print("TESTING MODEL ON MACBOOK")
print("=" * 60)

# 1. Load the model (new .keras format)
print("\n Loading model...")
model = keras.models.load_model('models/sign_model_20.keras')
print(" Model loaded successfully!")

# 2. Load word list
print("\n Loading word list...")
words = np.load('models/top_20_words.npy', allow_pickle=True)
print(f" Loaded {len(words)} words:")
print(f"   {list(words)}")

# 3. Load normalization parameters
print("\n Loading normalization parameters...")
norm_mean = np.load('models/norm_mean.npy')
norm_std = np.load('models/norm_std.npy')
print(f" Mean: {norm_mean:.4f}, Std: {norm_std:.4f}")

# 4. Test with dummy data
print("\n Testing with dummy data...")
dummy_input = np.random.rand(1, 40, 225)
dummy_normalized = (dummy_input - norm_mean) / norm_std

prediction = model.predict(dummy_normalized, verbose=0)
predicted_idx = np.argmax(prediction)
confidence = prediction[0][predicted_idx] * 100

print(f" Prediction works!")
print(f"   Predicted word: '{words[predicted_idx]}'")
print(f"   Confidence: {confidence:.1f}%")

print("\n" + "=" * 60)
print(" ALL TESTS PASSED - Ready to build website!")
print("=" * 60)