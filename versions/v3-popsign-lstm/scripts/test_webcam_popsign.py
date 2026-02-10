# """
# PopSign Webcam Testing Script
# Version 3: Real-world testing with webcam

# This script allows you to test the trained PopSign model
# by performing signs in front of your webcam.

# Controls:
#     SPACE - Start/Stop recording a sign
#     Q     - Quit the application
# """

# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# import os


# # Configuration

# MODEL_PATH = "../models/best_model_popsign.keras"
# DATA_DIR = "../data"
# MAX_FRAMES = 60
# MIN_FRAMES = 10


# # Load Model and Label Map

# print("Loading model...")
# model = load_model(MODEL_PATH)

# print("Loading label map...")
# label_map = np.load(os.path.join(DATA_DIR, "label_map_popsign.npy"), allow_pickle=True).item()

# # Create index to sign mapping
# # Handle both {idx: sign} and {sign: idx} formats
# if isinstance(list(label_map.keys())[0], int):
#     idx_to_sign = label_map
# else:
#     idx_to_sign = {v: k for k, v in label_map.items()}

# print(f"\nLoaded model with {len(idx_to_sign)} signs:")
# for idx in sorted(idx_to_sign.keys()):
#     print(f"  {idx}: {idx_to_sign[idx]}")


# # Initialize MediaPipe

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles


# # Feature Extraction Function


# def extract_landmarks(results):
#     """Extract landmarks from MediaPipe results."""
#     landmarks = []
    
#     # Left hand (21 landmarks × 3 coordinates = 63)
#     if results.left_hand_landmarks:
#         for lm in results.left_hand_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 63)
    
#     # Right hand (21 landmarks × 3 coordinates = 63)
#     if results.right_hand_landmarks:
#         for lm in results.right_hand_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 63)
    
#     # Pose (33 landmarks × 3 coordinates = 99)
#     if results.pose_landmarks:
#         for lm in results.pose_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 99)
    
#     return landmarks

# def predict_sign(frames):
#     """Predict sign from recorded frames."""
#     frames = np.array(frames)
    
#     # Pad or truncate to MAX_FRAMES
#     if len(frames) < MAX_FRAMES:
#         padding = np.zeros((MAX_FRAMES - len(frames), 225))
#         frames = np.vstack([frames, padding])
#     frames = frames[:MAX_FRAMES]
    
#     # Reshape for model: (1, 60, 225)
#     X = frames.reshape(1, MAX_FRAMES, 225)
    
#     # Predict
#     predictions = model.predict(X, verbose=0)[0]
#     predicted_idx = np.argmax(predictions)
#     confidence = predictions[predicted_idx]
    
#     return idx_to_sign[predicted_idx], confidence, predictions


# # Main Loop

# def main():
#     # Start webcam
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     # Set camera properties
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     recording = False
#     recorded_frames = []
#     last_prediction = None
#     last_confidence = None
    
#     print("\n" + "=" * 50)
#     print("WEBCAM TESTING STARTED")
#     print("=" * 50)
#     print("Controls:")
#     print("  SPACE - Start/Stop recording")
#     print("  Q     - Quit")
#     print("=" * 50)
    
#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Could not read frame")
#                 break
            
#             # Flip for mirror effect
#             frame = cv2.flip(frame, 1)
            
#             # Convert to RGB for MediaPipe
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(rgb_frame)
            
#             # Draw landmarks
#             if results.left_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, 
#                     results.left_hand_landmarks, 
#                     mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
                
#             if results.right_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, 
#                     results.right_hand_landmarks, 
#                     mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
            
#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, 
#                     results.pose_landmarks, 
#                     mp_holistic.POSE_CONNECTIONS,
#                     mp_drawing_styles.get_default_pose_landmarks_style()
#                 )
            
#             # Extract landmarks for recording
#             landmarks = extract_landmarks(results)
            
#             # UI - Recording status
#             if recording:
#                 recorded_frames.append(landmarks)
#                 cv2.putText(
#                     frame, 
#                     f"RECORDING: {len(recorded_frames)} frames", 
#                     (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, 
#                     (0, 0, 255), 
#                     2
#                 )
#                 # Progress bar
#                 progress = min(len(recorded_frames) / MAX_FRAMES, 1.0)
#                 cv2.rectangle(frame, (10, 40), (10 + int(300 * progress), 55), (0, 0, 255), -1)
#                 cv2.rectangle(frame, (10, 40), (310, 55), (255, 255, 255), 2)
#             else:
#                 cv2.putText(
#                     frame, 
#                     "Press SPACE to record", 
#                     (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, 
#                     (0, 255, 0), 
#                     2
#                 )
                
#             # UI - Last prediction
#             if last_prediction:
#                 cv2.putText(
#                     frame, 
#                     f"Prediction: {last_prediction} ({last_confidence*100:.1f}%)", 
#                     (10, 80), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, 
#                     (255, 255, 0), 
#                     2
#                 )
            
#             # UI - Available signs (right side)
#             cv2.putText(frame, "Available Signs:", (frame.shape[1] - 200, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#             y_pos = 55
#             for idx in sorted(idx_to_sign.keys()):
#                 cv2.putText(frame, f"{idx_to_sign[idx]}", (frame.shape[1] - 200, y_pos), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#                 y_pos += 22
            
#             # Show frame
#             cv2.imshow('PopSign Test - V3', frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
            
#             if key == ord(' '):  # SPACE - toggle recording
#                 if not recording:
#                     # Start recording
#                     recording = True
#                     recorded_frames = []
#                     last_prediction = None
#                     last_confidence = None
#                     print("\n Recording started...")
#                 else:
#                     # Stop recording and predict
#                     recording = False
#                     print(f"⏹  Recording stopped. {len(recorded_frames)} frames captured.")
                    
#                     if len(recorded_frames) >= MIN_FRAMES:
#                         sign, confidence, all_probs = predict_sign(recorded_frames)
#                         last_prediction = sign
#                         last_confidence = confidence
                        
#                         print(f"\n{'='*40}")
#                         print(f"PREDICTION: {sign}")
#                         print(f"CONFIDENCE: {confidence*100:.1f}%")
#                         print(f"{'='*40}")
                        
#                         # Show top 3 predictions
#                         top_3_idx = np.argsort(all_probs)[-3:][::-1]
#                         print("Top 3 predictions:")
#                         for idx in top_3_idx:
#                             print(f"  {idx_to_sign[idx]}: {all_probs[idx]*100:.1f}%")
#                     else:
#                         print(f" Too few frames ({len(recorded_frames)}). Need at least {MIN_FRAMES}.")
                    
#                     recorded_frames = []
            
#             elif key == ord('q'):  # Q - quit
#                 print("\nQuitting...")
#                 break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam testing ended.")

# if __name__ == "__main__":
#     main()





"""
PopSign Webcam Testing Script
Version 3: Real-world testing with webcam

This script loads model weights from a numpy file for maximum
compatibility across different Keras/TensorFlow versions.

Controls:
    SPACE - Start/Stop recording a sign
    Q     - Quit the application
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
import os


# Configuration

WEIGHTS_PATH = "../models/popsign_7_weights.npy"
DATA_DIR = "../data"
MAX_FRAMES = 60
MIN_FRAMES = 10
NUM_CLASSES = 7


# Build Model Architecture

print("Building model architecture...")

model = Sequential([
    Input(shape=(60, 225)),
    
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built successfully.")


# Load Weights from Numpy File

print(f"Loading weights from {WEIGHTS_PATH}...")

try:
    weights = np.load(WEIGHTS_PATH, allow_pickle=True)
    model.set_weights(list(weights))
    print(" Weights loaded successfully!")
except Exception as e:
    print(f" Error loading weights: {e}")
    exit(1)


# Load Label Map

print("Loading label map...")
label_map = np.load(os.path.join(DATA_DIR, "label_map_popsign_7.npy"), allow_pickle=True).item()

if isinstance(list(label_map.keys())[0], int):
    idx_to_sign = label_map
else:
    idx_to_sign = {v: k for k, v in label_map.items()}

print(f"\n{len(idx_to_sign)} signs loaded:")
for idx in sorted(idx_to_sign.keys()):
    print(f"  {idx}: {idx_to_sign[idx]}")


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
                # Progress bar
                progress = min(len(recorded_frames) / MAX_FRAMES, 1.0)
                cv2.rectangle(frame, (10, 45), (10 + int(300 * progress), 60), (0, 0, 255), -1)
                cv2.rectangle(frame, (10, 45), (310, 60), (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to record", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Prediction display
            if last_prediction:
                if last_confidence > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif last_confidence > 0.4:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 0, 255)  # Red - low
                    
                cv2.putText(frame, f"Prediction: {last_prediction}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"Confidence: {last_confidence*100:.1f}%", 
                           (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Available signs list (right side)
            cv2.putText(frame, "Available Signs:", (frame.shape[1] - 180, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos = 55
            for idx in sorted(idx_to_sign.keys()):
                sign = idx_to_sign[idx]
                # Highlight last predicted sign
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
                        
                        # Top 3
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