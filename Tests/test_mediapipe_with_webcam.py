import cv2
import mediapipe as mp
import numpy as np

print("Starting MediaPipe test...")
print("Press 'q' to quit")

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot access webcam!")
            break
        
        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect keypoints
        results = holistic.process(image)
        
        # Convert back for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS
            )
        
        # Show window
        cv2.imshow('MediaPipe Test - Press Q to quit', image)
        
        # Quit on 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Test complete!")