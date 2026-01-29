# Train Transformer model for comparison with LSTM

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("TRAINING TRANSFORMER MODEL")
print("=" * 60)

# 1. Load data
print("\n Loading data...")
X_train = np.load('models/X_train_proper.npy')
y_train = np.load('models/y_train_proper.npy')
X_test = np.load('models/X_test_proper.npy')
y_test = np.load('models/y_test_proper.npy')
words = np.load('models/top_20_words.npy', allow_pickle=True)

print(f" Training: {X_train.shape[0]} samples")
print(f" Testing: {X_test.shape[0]} samples")

# 2. Normalize
print("\n Normalizing...")
norm_mean = X_train.mean()
norm_std = X_train.std()
X_train_norm = (X_train - norm_mean) / norm_std
X_test_norm = (X_test - norm_mean) / norm_std

# 3. Convert labels
y_train_cat = to_categorical(y_train, num_classes=20)
y_test_cat = to_categorical(y_test, num_classes=20)

# 4. Build Transformer model
print("\n Building Transformer model...")

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Single transformer encoder block"""
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(x1)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    x2 = LayerNormalization(epsilon=1e-6)(x1 + ff_output)
    
    return x2

# Model parameters
sequence_length = 40
n_features = 225
head_size = 64
num_heads = 4
ff_dim = 128
num_transformer_blocks = 2
dropout = 0.3

# Input
inputs = Input(shape=(sequence_length, n_features))

# Project input to smaller dimension for efficiency
x = Dense(128)(inputs)

# Add positional information (learnable)
positions = tf.range(start=0, limit=sequence_length, delta=1)
position_embedding = keras.layers.Embedding(
    input_dim=sequence_length, 
    output_dim=128
)(positions)
x = x + position_embedding

# Transformer blocks
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

# Global pooling (aggregate all time steps)
x = GlobalAveragePooling1D()(x)

# Classification head
x = Dense(64, activation='relu')(x)
x = Dropout(dropout)(x)
outputs = Dense(20, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f" Model built - {model.count_params():,} parameters")
model.summary()

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
    ModelCheckpoint('models/transformer.keras', monitor='val_accuracy', save_best_only=True)
]

# 6. Train
print("\n Training Transformer...")
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
print(f" TRANSFORMER RESULTS")
print(f"=" * 60)
print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print(f"   LSTM Accuracy: 57.00%")
print(f"   Difference: {(accuracy * 100) - 57:.1f}%")

# 8. Per-class accuracy
print(f"\n Per-Word Accuracy:")
predictions = model.predict(X_test_norm, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

for i, word in enumerate(words):
    mask = y_test == i
    if mask.sum() > 0:
        word_acc = (pred_classes[mask] == i).mean() * 100
        print(f"   {word:12s}: {word_acc:5.1f}% ({mask.sum()} samples)")

# 9. Save normalization parameters
np.save('models/norm_mean_transformer.npy', norm_mean)
np.save('models/norm_std_transformer.npy', norm_std)

# 10. Save history for comparison
np.save('models/transformer_history.npy', {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

# 11. Check confusion on previously failing signs
print(f"\n Focus on Previously Failing Signs:")
problem_signs = ['no', 'thin', 'deaf', 'before']
for sign in problem_signs:
    i = list(words).index(sign)
    mask = y_test == i
    if mask.sum() > 0:
        correct = (pred_classes[mask] == i).sum()
        total = mask.sum()
        lstm_acc = {'no': 0, 'thin': 0, 'deaf': 20, 'before': 20}[sign]
        print(f"   {sign:8s}: {correct}/{total} correct ({correct/total*100:.0f}%) - LSTM was {lstm_acc}%")

print(f"\n Saved:")
print(f"   - models/transformer.keras")
print(f"   - models/norm_mean_transformer.npy")
print(f"   - models/norm_std_transformer.npy")

print("\n" + "=" * 60)
print(" TRANSFORMER TRAINING COMPLETE!")
print("=" * 60)