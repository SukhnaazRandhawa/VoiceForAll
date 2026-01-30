# Retrain the model locally on MacBook

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter

print("=" * 60)
print("RETRAINING MODEL LOCALLY")
print("=" * 60)

# 1. Load full dataset
print("\n Loading data...")
X = np.load('models/X_wlasl2000.npy')
y = np.load('models/y_wlasl2000.npy')
labels = np.load('models/label_map.npy', allow_pickle=True)
print(f" Loaded {X.shape[0]} videos, {len(labels)} words")

# 2. Get top 20 words
print("\n Filtering to top 20 words...")
word_counts = Counter(y)
sorted_counts = word_counts.most_common()
top_20_indices = [word_idx for word_idx, count in sorted_counts[:20]]
top_20_words = [labels[idx] for idx in top_20_indices]

old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(top_20_indices)}

X_20 = []
y_20 = []
for i in range(len(X)):
    if y[i] in top_20_indices:
        X_20.append(X[i])
        y_20.append(old_to_new[y[i]])

X_20 = np.array(X_20)
y_20 = np.array(y_20)
print(f" Filtered: {X_20.shape[0]} videos, 20 words")

# 3. Normalize
print("\n Normalizing...")
norm_mean = X_20.mean()
norm_std = X_20.std()
X_20_norm = (X_20 - norm_mean) / norm_std
print(f" Mean: {norm_mean:.4f}, Std: {norm_std:.4f}")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_20_norm, y_20, test_size=0.2, random_state=42, stratify=y_20
)
y_train_cat = to_categorical(y_train, num_classes=20)
y_test_cat = to_categorical(y_test, num_classes=20)
print(f" Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 5. Build model
print("\n🏗️ Building model...")
model = Sequential([
    Input(shape=(40, 225)),
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(20, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(" Model built")

# 6. Train
print("\n Training (this takes 3-5 minutes on CPU)...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    verbose=1
)

# 7. Evaluate
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n Final Test Accuracy: {accuracy * 100:.2f}%")

# 8. Save everything
print("\n Saving model and parameters...")
model.save('models/sign_model_20.keras')  # New Keras format
np.save('models/top_20_words.npy', np.array(top_20_words))
np.save('models/norm_mean.npy', norm_mean)
np.save('models/norm_std.npy', norm_std)

print("\n Saved:")
print("   - models/sign_model_20.keras")
print("   - models/top_20_words.npy")
print("   - models/norm_mean.npy")
print("   - models/norm_std.npy")

print("\n" + "=" * 60)
print(" RETRAINING COMPLETE!")
print("=" * 60)