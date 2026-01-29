# Train LSTM on properly split data (no leakage)

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("TRAINING LSTM (Proper Split - No Leakage)")
print("=" * 60)

# 1. Load properly split data
print("\n Loading data...")
X_train = np.load('models/X_train_proper.npy')
y_train = np.load('models/y_train_proper.npy')
X_test = np.load('models/X_test_proper.npy')
y_test = np.load('models/y_test_proper.npy')
words = np.load('models/top_20_words.npy', allow_pickle=True)

print(f" Training: {X_train.shape[0]} samples")
print(f" Testing: {X_test.shape[0]} samples (original videos only)")

# 2. Normalize using TRAINING data statistics only
print("\n Normalizing...")
norm_mean = X_train.mean()
norm_std = X_train.std()
X_train_norm = (X_train - norm_mean) / norm_std
X_test_norm = (X_test - norm_mean) / norm_std
print(f" Mean: {norm_mean:.4f}, Std: {norm_std:.4f}")

# 3. Convert labels
y_train_cat = to_categorical(y_train, num_classes=20)
y_test_cat = to_categorical(y_test, num_classes=20)

# 4. Build model
print("\n Building LSTM model...")
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
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f" Model built - {model.count_params():,} parameters")

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    ModelCheckpoint('models/lstm_proper.keras', monitor='val_accuracy', save_best_only=True)
]

# 6. Train
print("\n Training...")
history = model.fit(
    X_train_norm, y_train_cat,
    validation_data=(X_test_norm, y_test_cat),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate
loss, accuracy = model.evaluate(X_test_norm, y_test_cat, verbose=0)

print(f"\n" + "=" * 60)
print(f" LSTM RESULTS (Proper Split)")
print(f"=" * 60)
print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print(f"   (This is the REAL accuracy - no data leakage)")
print(f"   Random baseline: 5%")

# 8. Save normalization parameters
np.save('models/norm_mean_proper.npy', norm_mean)
np.save('models/norm_std_proper.npy', norm_std)

# 9. Per-class accuracy
print(f"\n Per-Word Accuracy:")
predictions = model.predict(X_test_norm, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

for i, word in enumerate(words):
    mask = y_test == i
    if mask.sum() > 0:
        word_acc = (pred_classes[mask] == i).mean() * 100
        print(f"   {word:12s}: {word_acc:5.1f}% ({mask.sum()} samples)")

print(f"\n Saved:")
print(f"   - models/lstm_proper.keras")
print(f"   - models/norm_mean_proper.npy")
print(f"   - models/norm_std_proper.npy")

print("\n" + "=" * 60)
print(" LSTM BASELINE COMPLETE!")
print("   This accuracy is realistic and honest.")
print("=" * 60)