# Proper augmentation: split first, then augment only training data

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

print("=" * 60)
print("PROPER DATA AUGMENTATION (No Leakage)")
print("=" * 60)

# 1. Load original data
print("\n Loading data...")
X = np.load('models/X_wlasl2000.npy')
y = np.load('models/y_wlasl2000.npy')
labels = np.load('models/label_map.npy', allow_pickle=True)
print(f" Loaded {X.shape[0]} videos")

# 2. Filter to top 20 words
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
print(f" Original data: {X_20.shape[0]} videos, 20 words")

# 3. SPLIT FIRST - before any augmentation!
print("\n Splitting BEFORE augmentation (this prevents data leakage)...")
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_20, y_20, test_size=0.2, random_state=42, stratify=y_20
)
print(f"   Original train: {X_train_orig.shape[0]} videos")
print(f"   Original test: {X_test.shape[0]} videos (NO augmentation applied here)")

# 4. Augmentation functions
def horizontal_flip(sequence):
    flipped = sequence.copy()
    left_hand = flipped[:, 0:63].copy()
    right_hand = flipped[:, 63:126].copy()
    flipped[:, 0:63] = right_hand
    flipped[:, 63:126] = left_hand
    for i in range(0, 225, 3):
        flipped[:, i] = 1 - flipped[:, i]
    return flipped

def add_noise(sequence, noise_level=0.02):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_landmarks(sequence, scale_factor):
    scaled = sequence.copy()
    center = 0.5
    scaled = center + (scaled - center) * scale_factor
    return np.clip(scaled, 0, 1)

def rotate_landmarks_2d(sequence, angle_degrees):
    rotated = sequence.copy()
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    center = 0.5
    for i in range(0, 225, 3):
        x = rotated[:, i] - center
        y = rotated[:, i+1] - center
        rotated[:, i] = x * cos_a - y * sin_a + center
        rotated[:, i+1] = x * sin_a + y * cos_a + center
    return np.clip(rotated, 0, 1)

def time_warp(sequence, factor):
    n_frames = sequence.shape[0]
    n_features = sequence.shape[1]
    new_n_frames = int(n_frames * factor)
    indices = np.linspace(0, n_frames - 1, new_n_frames)
    warped = np.zeros((new_n_frames, n_features))
    for i in range(n_features):
        warped[:, i] = np.interp(indices, np.arange(n_frames), sequence[:, i])
    final_indices = np.linspace(0, new_n_frames - 1, n_frames)
    result = np.zeros_like(sequence)
    for i in range(n_features):
        result[:, i] = np.interp(final_indices, np.arange(new_n_frames), warped[:, i])
    return result

# 5. Augment ONLY training data
print("\n Augmenting ONLY training data...")

X_train_aug = [X_train_orig]
y_train_aug = [y_train_orig]

# Horizontal flip
print("   - Horizontal flip...")
X_flipped = np.array([horizontal_flip(seq) for seq in X_train_orig])
X_train_aug.append(X_flipped)
y_train_aug.append(y_train_orig.copy())

# Noise (2 variations)
print("   - Adding noise...")
for noise_level in [0.01, 0.02]:
    X_noisy = np.array([add_noise(seq, noise_level) for seq in X_train_orig])
    X_train_aug.append(X_noisy)
    y_train_aug.append(y_train_orig.copy())

# Scale (2 variations)
print("   - Scaling...")
for scale in [0.9, 1.1]:
    X_scaled = np.array([scale_landmarks(seq, scale) for seq in X_train_orig])
    X_train_aug.append(X_scaled)
    y_train_aug.append(y_train_orig.copy())

# Rotation (2 variations)
print("   - Rotating...")
for angle in [-10, 10]:
    X_rotated = np.array([rotate_landmarks_2d(seq, angle) for seq in X_train_orig])
    X_train_aug.append(X_rotated)
    y_train_aug.append(y_train_orig.copy())

# Time warp (2 variations)
print("   - Time warping...")
for factor in [0.9, 1.1]:
    X_warped = np.array([time_warp(seq, factor) for seq in X_train_orig])
    X_train_aug.append(X_warped)
    y_train_aug.append(y_train_orig.copy())

# Combine augmented training data
X_train = np.concatenate(X_train_aug, axis=0)
y_train = np.concatenate(y_train_aug, axis=0)

# Shuffle training data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"\n Final Dataset:")
print(f"   Training: {X_train.shape[0]} samples (augmented)")
print(f"   Testing: {X_test.shape[0]} samples (original only - no augmentation!)")
print(f"   Augmentation factor: {X_train.shape[0] / X_train_orig.shape[0]:.1f}x")

# 6. Save
print("\n Saving...")
np.save('models/X_train_proper.npy', X_train)
np.save('models/y_train_proper.npy', y_train)
np.save('models/X_test_proper.npy', X_test)
np.save('models/y_test_proper.npy', y_test)
np.save('models/top_20_words.npy', np.array(top_20_words))

print(f"\n Saved:")
print(f"   - models/X_train_proper.npy: {X_train.shape}")
print(f"   - models/y_train_proper.npy: {y_train.shape}")
print(f"   - models/X_test_proper.npy: {X_test.shape}")
print(f"   - models/y_test_proper.npy: {y_test.shape}")

print("\n" + "=" * 60)
print(" PROPER AUGMENTATION COMPLETE!")
print("   Test set contains ONLY original videos (no leakage)")
print("=" * 60)