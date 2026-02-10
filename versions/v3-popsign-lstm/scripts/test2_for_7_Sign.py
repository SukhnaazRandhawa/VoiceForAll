import numpy as np

weights = np.load('../models/popsign_7_weights.npy', allow_pickle=True)
print(f"Number of weight arrays: {len(weights)}")
print(f"Last layer shape: {weights[-2].shape}, {weights[-1].shape}")
print(f"Should be: (64, 7) and (7,) for 7 classes")