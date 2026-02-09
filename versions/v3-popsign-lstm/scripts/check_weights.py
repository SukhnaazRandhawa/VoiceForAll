import numpy as np

# Check the weights file
weights = np.load('../models/popsign_weights.npy', allow_pickle=True)
print(f"Number of weight arrays: {len(weights)}")
print(f"\nShapes of each weight array:")
for i, w in enumerate(weights):
    print(f"  {i}: {w.shape}")