# Generate augmented training data from existing landmarks

import numpy as np
from collections import Counter

print("=" * 60)
print("DATA AUGMENTATION FOR SIGN LANGUAGE")
print("=" * 60)

# 1. Load original data
print("\n Loading data...")
X = np.load('models/X_wlasl2000.npy')
y = np.load('models/y_wlasl2000.npy')
labels = np.load('models/label_map.npy', allow_pickle=True)
print(f" Loaded {X.shape[0]} videos, shape: {X.shape}")

# 2. Filter to top 20 words (same as before)
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

# 3. Augmentation functions
def horizontal_flip(sequence):
    """
    Flip left and right hands.
    Features: [left_hand(0-62), right_hand(63-125), pose(126-224)]
    For x coordinates (every 3rd starting at 0), flip: x -> 1-x
    """
    flipped = sequence.copy()
    
    # Swap left hand (0-62) and right hand (63-125)
    left_hand = flipped[:, 0:63].copy()
    right_hand = flipped[:, 63:126].copy()
    flipped[:, 0:63] = right_hand
    flipped[:, 63:126] = left_hand
    
    # Flip x coordinates (every 3rd value starting at 0)
    for i in range(0, 225, 3):  # x coordinates
        flipped[:, i] = 1 - flipped[:, i]
    
    return flipped

def add_noise(sequence, noise_level=0.02):
    """Add small random noise to landmarks"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_landmarks(sequence, scale_factor):
    """Scale landmarks around center"""
    scaled = sequence.copy()
    center = 0.5  # Assuming normalized coordinates
    scaled = center + (scaled - center) * scale_factor
    return np.clip(scaled, 0, 1)

def rotate_landmarks_2d(sequence, angle_degrees):
    """Rotate x,y coordinates around center"""
    rotated = sequence.copy()
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    center = 0.5
    
    # Rotate x,y pairs (every 3 values: x, y, z)
    for i in range(0, 225, 3):
        x = rotated[:, i] - center
        y = rotated[:, i+1] - center
        rotated[:, i] = x * cos_a - y * sin_a + center
        rotated[:, i+1] = x * sin_a + y * cos_a + center
    
    return np.clip(rotated, 0, 1)

def time_warp(sequence, factor):
    """Speed up or slow down by interpolating frames"""
    n_frames = sequence.shape[0]
    n_features = sequence.shape[1]
    
    # New number of frames
    new_n_frames = int(n_frames * factor)
    
    # Interpolate to target length, then resize back to original
    indices = np.linspace(0, n_frames - 1, new_n_frames)
    warped = np.zeros((new_n_frames, n_features))
    
    for i in range(n_features):
        warped[:, i] = np.interp(indices, np.arange(n_frames), sequence[:, i])
    
    # Resize back to original frame count
    final_indices = np.linspace(0, new_n_frames - 1, n_frames)
    result = np.zeros_like(sequence)
    for i in range(n_features):
        result[:, i] = np.interp(final_indices, np.arange(new_n_frames), warped[:, i])
    
    return result

# 4. Generate augmented data
print("\n Generating augmented data...")

X_augmented = [X_20]  # Start with original
y_augmented = [y_20]

# Augmentation 1: Horizontal flip
print("   - Horizontal flip...")
X_flipped = np.array([horizontal_flip(seq) for seq in X_20])
X_augmented.append(X_flipped)
y_augmented.append(y_20.copy())

# Augmentation 2: Add noise (2 variations)
print("   - Adding noise...")
for noise_level in [0.01, 0.02]:
    X_noisy = np.array([add_noise(seq, noise_level) for seq in X_20])
    X_augmented.append(X_noisy)
    y_augmented.append(y_20.copy())

# Augmentation 3: Scale (2 variations)
print("   - Scaling...")
for scale in [0.9, 1.1]:
    X_scaled = np.array([scale_landmarks(seq, scale) for seq in X_20])
    X_augmented.append(X_scaled)
    y_augmented.append(y_20.copy())

# Augmentation 4: Rotation (2 variations)
print("   - Rotating...")
for angle in [-10, 10]:
    X_rotated = np.array([rotate_landmarks_2d(seq, angle) for seq in X_20])
    X_augmented.append(X_rotated)
    y_augmented.append(y_20.copy())

# Augmentation 5: Time warp (2 variations)
print("   - Time warping...")
for factor in [0.9, 1.1]:
    X_warped = np.array([time_warp(seq, factor) for seq in X_20])
    X_augmented.append(X_warped)
    y_augmented.append(y_20.copy())

# 5. Combine all augmented data
X_aug_combined = np.concatenate(X_augmented, axis=0)
y_aug_combined = np.concatenate(y_augmented, axis=0)

print(f"\n Augmentation Results:")
print(f"   Original: {X_20.shape[0]} samples")
print(f"   Augmented: {X_aug_combined.shape[0]} samples")
print(f"   Increase: {X_aug_combined.shape[0] / X_20.shape[0]:.1f}x")

# 6. Shuffle the data
print("\n Shuffling data...")
shuffle_idx = np.random.permutation(len(X_aug_combined))
X_aug_combined = X_aug_combined[shuffle_idx]
y_aug_combined = y_aug_combined[shuffle_idx]

# 7. Save augmented data
print("\n Saving augmented data...")
np.save('models/X_augmented_20.npy', X_aug_combined)
np.save('models/y_augmented_20.npy', y_aug_combined)
np.save('models/top_20_words.npy', np.array(top_20_words))

print(f"\n Saved:")
print(f"   - models/X_augmented_20.npy: {X_aug_combined.shape}")
print(f"   - models/y_augmented_20.npy: {y_aug_combined.shape}")
print(f"   - models/top_20_words.npy: {len(top_20_words)} words")

print("\n" + "=" * 60)
print(" DATA AUGMENTATION COMPLETE!")
print(f"   You now have {X_aug_combined.shape[0]} training samples")
print("=" * 60)