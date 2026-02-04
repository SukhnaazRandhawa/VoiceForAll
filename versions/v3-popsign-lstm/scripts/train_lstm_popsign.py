"""
PopSign LSTM Training Script
Version 3: PopSign dataset with LSTM architecture

This script trains an LSTM model on the PopSign ASL dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os


# Configuration

DATA_DIR = "../data"
MODEL_DIR = "../models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32


# Load Data

print("=" * 50)
print("Loading Data")
print("=" * 50)

X = np.load(os.path.join(DATA_DIR, "X_popsign.npy"))
y = np.load(os.path.join(DATA_DIR, "Y_popsign.npy"))
label_map = np.load(os.path.join(DATA_DIR, "label_map_popsign.npy"), allow_pickle=True).item()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of classes: {len(label_map)}")
print(f"\nSigns: {list(label_map.values())}")


# Train/Test Split

print("\n" + "=" * 50)
print("Splitting Data")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

# One-hot encode labels
num_classes = len(label_map)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


# Build Model

print("\n" + "=" * 50)
print("Building Model")
print("=" * 50)

model = Sequential([
    # First LSTM layer
    LSTM(128, return_sequences=True, input_shape=(60, 225)),
    Dropout(0.3),
    BatchNormalization(),
    
    # Second LSTM layer
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),
    
    # Dense layers
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    # Output layer
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# Callbacks

os.makedirs(MODEL_DIR, exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model_popsign.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# Train Model

print("\n" + "=" * 50)
print("Training")
print("=" * 50)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# Final Evaluation

print("\n" + "=" * 50)
print("Final Results")
print("=" * 50)

loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save training history
history_dict = {
    'train_accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
}
np.save(os.path.join(MODEL_DIR, 'training_history.npy'), history_dict)
print(f"\nTraining history saved to {MODEL_DIR}/training_history.npy")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)

