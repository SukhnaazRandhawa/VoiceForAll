# Train LSTM on augmented data to establish baseline

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TRAINING LSTM ON AUGMENTED DATA")
print("=" * 60)

# 1. Load augmented data
print("\n Loading augmented data...")
X = np.load('models/X_augmented_20.npy')
y = np.load('models/y_augmented_20.npy')
words = np.load('models/top_20_words.npy', allow_pickle=True)
print(f" Loaded {X.shape[0]} samples, {len(words)} words")

# 2. Normalize
print("\n Normalizing...")
norm_mean = X.mean()
norm_std = X.std()
X_norm = (X - norm_mean) / norm_std
print(f" Mean: {norm_mean:.4f}, Std: {norm_std:.4f}")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
y_train_cat = to_categorical(y_train, num_classes=20)
y_test_cat = to_categorical(y_test, num_classes=20)
print(f" Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 4. Build LSTM model (same architecture as before)
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
print(" Model built")
print(f"   Parameters: {model.count_params():,}")

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ModelCheckpoint('models/lstm_augmented.keras', monitor='val_accuracy', save_best_only=True)
]

# 6. Train
print("\n Training LSTM on augmented data...")
print("   (This may take 5-10 minutes)")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n" + "=" * 60)
print(f" LSTM RESULTS (Augmented Data)")
print(f"=" * 60)
print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print(f"   Previous (no augmentation): 40%")
print(f"   Improvement: {(accuracy * 100) - 40:.1f}%")

# 8. Save normalization parameters
np.save('models/norm_mean_aug.npy', norm_mean)
np.save('models/norm_std_aug.npy', norm_std)

print(f"\n Saved:")
print(f"   - models/lstm_augmented.keras")
print(f"   - models/norm_mean_aug.npy")
print(f"   - models/norm_std_aug.npy")

# 9. Save history for comparison later
np.save('models/lstm_history.npy', {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

print("\n" + "=" * 60)
print(" LSTM BASELINE COMPLETE!")
print("   Next: Build Transformer and compare")
print("=" * 60)