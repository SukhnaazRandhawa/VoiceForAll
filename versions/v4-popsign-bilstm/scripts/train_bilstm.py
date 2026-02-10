import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Load data
X = np.load('X_popsign_42.npy')
y = np.load('y_popsign_42.npy')
label_map = np.load('label_map_popsign_42.npy', allow_pickle=True).item()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of classes: {len(label_map)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# One-hot encode
num_classes = len(label_map)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build Bidirectional LSTM (matching PopSign paper)
model = Sequential([
    Input(shape=(60, 225)),
    Masking(mask_value=0.0),
    
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model_bilstm_42.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
print("\n" + "="*50)
print("Training Bidirectional LSTM on 42 signs...")
print("="*50)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=40,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Final evaluation
print("\n" + "="*50)
print("Final Results")
print("="*50)
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")