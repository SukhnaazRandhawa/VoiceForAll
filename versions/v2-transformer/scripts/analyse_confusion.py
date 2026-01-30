# Analyse which signs get confused with which

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("CONFUSION MATRIX ANALYSIS")
print("=" * 60)

# Load model and data
model = keras.models.load_model('models/lstm_proper.keras')
X_test = np.load('models/X_test_proper.npy')
y_test = np.load('models/y_test_proper.npy')
words = np.load('models/top_20_words.npy', allow_pickle=True)
norm_mean = np.load('models/norm_mean_proper.npy')
norm_std = np.load('models/norm_std_proper.npy')

# Normalize test data
X_test_norm = (X_test - norm_mean) / norm_std

# Get predictions
predictions = model.predict(X_test_norm, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

# Build confusion matrix
n_classes = len(words)
confusion = np.zeros((n_classes, n_classes), dtype=int)

for true, pred in zip(y_test, pred_classes):
    confusion[true][pred] += 1

# Print confusion analysis
print("\n Most Common Confusions:")
print("-" * 50)

for i, word in enumerate(words):
    true_count = (y_test == i).sum()
    correct = confusion[i][i]
    
    if true_count > 0:
        # Find what this word gets confused with
        confusions = []
        for j, other_word in enumerate(words):
            if i != j and confusion[i][j] > 0:
                confusions.append((other_word, confusion[i][j]))
        
        confusions.sort(key=lambda x: -x[1])
        
        print(f"\n{word.upper()} (Accuracy: {correct}/{true_count} = {correct/true_count*100:.0f}%)")
        if confusions:
            print(f"   Confused with:")
            for conf_word, count in confusions[:3]:  # Top 3 confusions
                print(f"      → {conf_word}: {count} times")
        else:
            print(f"    No confusions!")

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=words, yticklabels=words)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('LSTM Confusion Matrix - Which Signs Get Confused?')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print(f"\n Saved confusion matrix to models/confusion_matrix.png")

# Summary statistics
print("\n" + "=" * 60)
print(" SUMMARY")
print("=" * 60)
print(f"Overall Accuracy: {(pred_classes == y_test).mean()*100:.1f}%")
print(f"Perfect signs (100%): {sum(1 for i in range(n_classes) if confusion[i][i] == (y_test==i).sum() and (y_test==i).sum() > 0)}")
print(f"Zero accuracy signs: {sum(1 for i in range(n_classes) if confusion[i][i] == 0 and (y_test==i).sum() > 0)}")