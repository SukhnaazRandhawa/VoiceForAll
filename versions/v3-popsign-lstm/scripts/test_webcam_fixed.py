import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os


# CONFIGURATION - Change these to switch between models

# For 13-sign model:
MODEL_PATH = "../models/best_model_popsign.keras"
LABEL_MAP_PATH = "../data/label_map_popsign.npy"

# For 26-sign model (comment above, uncomment below):
# MODEL_PATH = "../models/best_model_popsign_26.keras"
# LABEL_MAP_PATH = "../data/label_map_popsign_26.npy"

MAX_FRAMES = 60
MIN_FRAMES = 10

# Load Model and Labels

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print(f"Model loaded! Output shape: {model.output_shape}")

print(f"Loading label map from {LABEL_MAP_PATH}...")
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

if isinstance(list(label_map.keys())[0], int):
    idx_to_sign = label_map
else:
    idx_to_sign = {v: k for k, v in label_map.items()}

print(f"\n{len(idx_to_sign)} signs loaded:")
for idx in sorted(idx_to_sign.keys()):
    print(f"  {idx}: {idx_to_sign[idx]}")

# Verify model output matches label map
num_classes = model.output_shape[-1]
if num_classes != len(idx_to_sign):
    print(f"\n WARNING: Model outputs {num_classes} classes but label map has {len(idx_to_sign)} signs!")
    exit(1)
else:
    print(f"\n Model and label map match ({num_classes} classes)")

# Initialize MediaPipe

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Helper Functions

def extract_landmarks(results):
    """Extract landmarks from MediaPipe results."""
    landmarks = []
    
    # Left hand (63 values)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Right hand (63 values)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Pose (99 values)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)
    
    return landmarks


def predict_sign(frames):
    """Predict sign from recorded frames."""
    frames = np.array(frames)
    
    if len(frames) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(frames), 225))
        frames = np.vstack([frames, padding])
    frames = frames[:MAX_FRAMES]
    
    X = frames.reshape(1, MAX_FRAMES, 225)
    predictions = model.predict(X, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    
    return idx_to_sign[predicted_idx], confidence, predictions

# Main Function

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recording = False
    recorded_frames = []
    last_prediction = None
    last_confidence = None
    
    print("\n" + "=" * 50)
    print("WEBCAM TESTING - PopSign V3")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Start/Stop recording")
    print("  Q     - Quit")
    print("=" * 50 + "\n")
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Draw hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            landmarks = extract_landmarks(results)
            
            # Recording UI
            if recording:
                recorded_frames.append(landmarks)
                cv2.putText(frame, f"RECORDING: {len(recorded_frames)} frames", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                progress = min(len(recorded_frames) / MAX_FRAMES, 1.0)
                cv2.rectangle(frame, (10, 45), (10 + int(300 * progress), 60), (0, 0, 255), -1)
                cv2.rectangle(frame, (10, 45), (310, 60), (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to record", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Prediction display
            if last_prediction:
                if last_confidence > 0.7:
                    color = (0, 255, 0)
                elif last_confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                    
                cv2.putText(frame, f"Prediction: {last_prediction}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"Confidence: {last_confidence*100:.1f}%", 
                           (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Available signs list
            cv2.putText(frame, "Available Signs:", (frame.shape[1] - 180, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos = 55
            for idx in sorted(idx_to_sign.keys()):
                sign = idx_to_sign[idx]
                if last_prediction and sign == last_prediction:
                    cv2.putText(frame, f"> {sign}", (frame.shape[1] - 180, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"  {sign}", (frame.shape[1] - 180, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 22
            
            cv2.imshow('PopSign Test - V3', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if not recording:
                    recording = True
                    recorded_frames = []
                    last_prediction = None
                    last_confidence = None
                    print(" Recording...")
                else:
                    recording = False
                    print(f"  Stopped ({len(recorded_frames)} frames)")
                    
                    if len(recorded_frames) >= MIN_FRAMES:
                        sign, conf, probs = predict_sign(recorded_frames)
                        last_prediction = sign
                        last_confidence = conf
                        
                        print(f"\n{'='*40}")
                        print(f"  PREDICTION: {sign}")
                        print(f"  CONFIDENCE: {conf*100:.1f}%")
                        print(f"{'='*40}")
                        
                        top3 = np.argsort(probs)[-3:][::-1]
                        print("  Top 3:")
                        for i in top3:
                            print(f"    {idx_to_sign[i]}: {probs[i]*100:.1f}%")
                        print()
                    else:
                        print(f" Too short. Need {MIN_FRAMES}+ frames.\n")
                    
                    recorded_frames = []
            
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam testing ended.")


if __name__ == "__main__":
    main()