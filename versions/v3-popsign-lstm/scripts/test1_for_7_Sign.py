import numpy as np

# Old 13-sign label map
lm_13 = np.load('../data/label_map_popsign.npy', allow_pickle=True).item()
print("13-sign label map:")
for idx, sign in sorted(lm_13.items()):
    print(f"  {idx}: {sign}")

print()

# New 7-sign label map
lm_7 = np.load('../data/label_map_popsign_7.npy', allow_pickle=True).item()
print("7-sign label map:")
for idx, sign in sorted(lm_7.items()):
    print(f"  {idx}: {sign}")