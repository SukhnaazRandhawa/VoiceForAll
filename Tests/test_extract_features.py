import cv2
import mediapipe as mp
import numpy as np

def extract_keypoints(results):
    """Extract 225 features"""
    # Left hand
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    
    # Right hand  
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    
    # Pose
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(99)
    
    return np.concatenate([lh, rh, pose])

print("Testing - wave hands!")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        
        # Print EVERY value change
        print(f"VALUES: {keypoints[0:5]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        cv2.imshow('Test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done!")