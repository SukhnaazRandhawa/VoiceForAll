# # app.py - Sign Language Recognition Web App

# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow import keras
# from collections import deque
# import time

# print("=" * 60)
# print("SIGN LANGUAGE RECOGNITION APP")
# print("=" * 60)

# # Load model and parameters
# print("\n Loading model...")
# # model = keras.models.load_model('models/sign_model_20.keras')
# model = keras.models.load_model('models/lstm_augmented.keras')
# words = np.load('models/top_20_words.npy', allow_pickle=True)
# # norm_mean = np.load('models/norm_mean.npy')
# norm_mean = np.load('models/norm_mean_aug.npy')
# # norm_std = np.load('models/norm_std.npy')
# norm_std = np.load('models/norm_std_aug.npy')
# print(f" Model loaded - {len(words)} words")

# # Initialize MediaPipe
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# def extract_keypoints(results):
#     """Extract 225 features from MediaPipe results"""
#     if results.left_hand_landmarks:
#         lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
#     else:
#         lh = np.zeros(63)
    
#     if results.right_hand_landmarks:
#         rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
#     else:
#         rh = np.zeros(63)
    
#     if results.pose_landmarks:
#         pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
#     else:
#         pose = np.zeros(99)
    
#     return np.concatenate([lh, rh, pose])

# # Buffer to collect frames
# frame_buffer = deque(maxlen=40)

# print("\n🎥 Starting webcam...")
# print("   - Show signs to the camera")
# print("   - Press 'q' to quit")
# print("   - Press 'c' to clear buffer")
# print("=" * 60)

# cap = cv2.VideoCapture(0)

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Flip for mirror effect
#         frame = cv2.flip(frame, 1)
        
#         # Convert to RGB for MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)
        
#         # Extract keypoints and add to buffer
#         keypoints = extract_keypoints(results)
#         frame_buffer.append(keypoints)
        
#         # Draw landmarks
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         if results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#         if results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
#         # Make prediction if we have enough frames
#         prediction_text = "Collecting frames..."
#         confidence_text = ""
        
#         if len(frame_buffer) == 40:
#             # Prepare input
#             sequence = np.array(list(frame_buffer))
#             sequence = (sequence - norm_mean) / norm_std
#             sequence = np.expand_dims(sequence, axis=0)
            
#             # Predict
#             prediction = model.predict(sequence, verbose=0)
#             predicted_idx = np.argmax(prediction)
#             confidence = prediction[0][predicted_idx]
            
#             if confidence > 0.3:  # Only show if confident enough
#                 prediction_text = f"Sign: {words[predicted_idx].upper()}"
#                 confidence_text = f"Confidence: {confidence * 100:.1f}%"
#             else:
#                 prediction_text = "Keep signing..."
#                 confidence_text = f"Best guess: {words[predicted_idx]} ({confidence * 100:.1f}%)"
        
#         # Display info on frame
#         cv2.rectangle(image, (0, 0), (400, 100), (0, 0, 0), -1)
#         cv2.putText(image, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(image, f"Buffer: {len(frame_buffer)}/40", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Show frame
#         cv2.imshow('Sign Language Recognition - Press Q to quit', image)
        
#         # Handle key presses
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('c'):
#             frame_buffer.clear()
#             print("Buffer cleared!")

# cap.release()
# cv2.destroyAllWindows()
# print("\n App closed!")



# app.py - Improved Sign Language Recognition with Stability Detection

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque
import time

print("=" * 60)
print("SIGN LANGUAGE RECOGNITION APP (v2 - Stability Detection)")
print("=" * 60)

# Load model and parameters
print("\n Loading model...")
#model = keras.models.load_model('models/lstm_augmented.keras')
model = keras.models.load_model('models/lstm_proper.keras')
words = np.load('models/top_20_words.npy', allow_pickle=True)
#norm_mean = np.load('models/norm_mean_aug.npy')
norm_mean = np.load('models/norm_mean_proper.npy')
#norm_std = np.load('models/norm_std_aug.npy')
norm_std = np.load('models/norm_std_proper.npy')
print(f" Model loaded - {len(words)} words")

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract 225 features from MediaPipe results"""
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(99)
    
    return np.concatenate([lh, rh, pose])

def hands_detected(results):
    """Check if at least one hand is detected"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def calculate_movement(buffer, window=10):
    """Calculate how much movement occurred in recent frames"""
    if len(buffer) < window:
        return 0
    
    recent = list(buffer)[-window:]
    # Calculate variance of hand positions (first 126 features = hands)
    recent_hands = np.array([frame[:126] for frame in recent])
    movement = np.std(recent_hands)
    return movement

# Settings
STABILITY_FRAMES = 8          # Must predict same word for this many frames
CONFIDENCE_THRESHOLD = 0.7    # Minimum confidence to consider
MOVEMENT_THRESHOLD = 0.01     # Minimum movement to consider as signing
COOLDOWN_FRAMES = 20          # Frames to wait after showing a prediction

# Buffers and state
frame_buffer = deque(maxlen=40)
prediction_history = deque(maxlen=STABILITY_FRAMES)
cooldown_counter = 0
last_shown_word = ""
last_shown_time = 0

print("\n🎥 Starting webcam...")
print("   - Show signs to the camera")
print("   - Wait for hand detection")
print("   - Sign clearly and hold briefly")
print("   - Press 'q' to quit")
print("   - Press 'c' to clear")
print("=" * 60)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Extract keypoints and add to buffer
        keypoints = extract_keypoints(results)
        frame_buffer.append(keypoints)
        
        # Draw landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Status variables
        status_text = "Waiting for hands..."
        prediction_text = ""
        confidence_text = ""
        color = (200, 200, 200)  # Gray
        
        # Decrement cooldown
        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        # Check if hands are detected
        if hands_detected(results):
            status_text = "Hands detected"
            color = (0, 255, 255)  # Yellow
            
            # Check if enough frames collected
            if len(frame_buffer) == 40:
                # Calculate movement
                movement = calculate_movement(frame_buffer)
                
                if movement > MOVEMENT_THRESHOLD:
                    status_text = "Analyzing sign..."
                    color = (255, 165, 0)  # Orange
                    
                    # Prepare input
                    sequence = np.array(list(frame_buffer))
                    sequence = (sequence - norm_mean) / norm_std
                    sequence = np.expand_dims(sequence, axis=0)
                    
                    # Predict
                    prediction = model.predict(sequence, verbose=0)
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[0][predicted_idx]
                    predicted_word = words[predicted_idx]
                    
                    # Add to prediction history
                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_history.append(predicted_word)
                    
                    # Check for stability (same prediction multiple times)
                    if len(prediction_history) == STABILITY_FRAMES:
                        # Check if all recent predictions are the same
                        if len(set(prediction_history)) == 1 and cooldown_counter == 0:
                            stable_word = prediction_history[0]
                            
                            # Show the prediction!
                            prediction_text = f"SIGN: {stable_word.upper()}"
                            confidence_text = f"Confidence: {confidence * 100:.0f}%"
                            status_text = "Sign detected!"
                            color = (0, 255, 0)  # Green
                            
                            # Set cooldown and clear history
                            last_shown_word = stable_word
                            last_shown_time = time.time()
                            cooldown_counter = COOLDOWN_FRAMES
                            prediction_history.clear()
                            
                            # Print to terminal
                            print(f" Detected: {stable_word.upper()} ({confidence * 100:.0f}%)")
                else:
                    status_text = "Hold still or sign more clearly"
                    prediction_history.clear()
        else:
            # No hands - clear prediction history
            prediction_history.clear()
        
        # Show last prediction for a few seconds
        if last_shown_word and time.time() - last_shown_time < 2.0:
            prediction_text = f"SIGN: {last_shown_word.upper()}"
            color = (0, 255, 0)
        
        # Display info on frame
        # Background box
        cv2.rectangle(image, (0, 0), (450, 120), (0, 0, 0), -1)
        
        # Status
        cv2.putText(image, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Prediction
        if prediction_text:
            cv2.putText(image, prediction_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, confidence_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Buffer info
        cv2.putText(image, f"Buffer: {len(frame_buffer)}/40 | Stable: {len(prediction_history)}/{STABILITY_FRAMES}", 
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Show frame
        cv2.imshow('Sign Language Recognition - Press Q to quit', image)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_buffer.clear()
            prediction_history.clear()
            last_shown_word = ""
            print(" Cleared!")

cap.release()
cv2.destroyAllWindows()
print("\n App closed!")