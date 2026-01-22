# app.py - Sign Language Recognition Web App

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque
import time

print("=" * 60)
print("SIGN LANGUAGE RECOGNITION APP")
print("=" * 60)

# Load model and parameters
print("\n Loading model...")
model = keras.models.load_model('models/sign_model_20.keras')
words = np.load('models/top_20_words.npy', allow_pickle=True)
norm_mean = np.load('models/norm_mean.npy')
norm_std = np.load('models/norm_std.npy')
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

# Buffer to collect frames
frame_buffer = deque(maxlen=40)

print("\n🎥 Starting webcam...")
print("   - Show signs to the camera")
print("   - Press 'q' to quit")
print("   - Press 'c' to clear buffer")
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
        
        # Make prediction if we have enough frames
        prediction_text = "Collecting frames..."
        confidence_text = ""
        
        if len(frame_buffer) == 40:
            # Prepare input
            sequence = np.array(list(frame_buffer))
            sequence = (sequence - norm_mean) / norm_std
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predict
            prediction = model.predict(sequence, verbose=0)
            predicted_idx = np.argmax(prediction)
            confidence = prediction[0][predicted_idx]
            
            if confidence > 0.3:  # Only show if confident enough
                prediction_text = f"Sign: {words[predicted_idx].upper()}"
                confidence_text = f"Confidence: {confidence * 100:.1f}%"
            else:
                prediction_text = "Keep signing..."
                confidence_text = f"Best guess: {words[predicted_idx]} ({confidence * 100:.1f}%)"
        
        # Display info on frame
        cv2.rectangle(image, (0, 0), (400, 100), (0, 0, 0), -1)
        cv2.putText(image, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Buffer: {len(frame_buffer)}/40", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Sign Language Recognition - Press Q to quit', image)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_buffer.clear()
            print("Buffer cleared!")

cap.release()
cv2.destroyAllWindows()
print("\n App closed!")