import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking, Input
import os

# CONFIGURATION

WEIGHTS_PATH = "../models/popsign_exact_weights_43.npy"
LABEL_MAP_PATH = "../data/label_map_hands_only_43.npy"
MAX_FRAMES = 60
MIN_FRAMES = 10
MIN_CONFIDENCE = 0.40  # Minimum confidence to accept a word

# Load Label Map First

print(f"Loading label map from {LABEL_MAP_PATH}...")
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

if isinstance(list(label_map.keys())[0], int):
    idx_to_sign = label_map
else:
    idx_to_sign = {v: k for k, v in label_map.items()}

NUM_CLASSES = len(idx_to_sign)

print(f"\n {NUM_CLASSES} signs loaded")

# Build Model

print("\nBuilding model...")

model = Sequential([
    Input(shape=(60, 63)),
    Masking(mask_value=0.0),
    
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.5),
    
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load Weights

print(f"Loading weights from {WEIGHTS_PATH}...")

try:
    weights = np.load(WEIGHTS_PATH, allow_pickle=True)
    model.set_weights(list(weights))
    print(" Weights loaded successfully!")
except Exception as e:
    print(f" Error loading weights: {e}")
    exit(1)

# Initialize MediaPipe

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Sentence Formation Function (Simple Version)

def form_sentence_simple(words, confidences):
    """
    Simple sentence formation without LLM.
    Filters low confidence words and joins remaining words.
    """
    # Filter words with low confidence
    filtered_words = []
    for word, conf in zip(words, confidences):
        if conf >= MIN_CONFIDENCE:
            filtered_words.append(word)
        else:
            print(f"  ⚠️ Filtered out '{word}' (confidence: {conf*100:.1f}%)")
    
    if not filtered_words:
        return "No words detected with sufficient confidence."
    
    # Simple sentence formation
    sentence = " ".join(filtered_words)
    
    # Add basic grammar (simple version)
    # Capitalize first letter, add period
    sentence = sentence.capitalize() + "."
    
    return sentence, filtered_words


def form_sentence_with_llm(words, confidences):
    """
    Advanced sentence formation using LLM API.
    Uncomment and add your API key to use.
    """
    # Filter very low confidence words first
    valid_words = []
    valid_conf = []
    for word, conf in zip(words, confidences):
        if conf >= 0.30:  # Keep words above 30%
            valid_words.append(word)
            valid_conf.append(conf)
    
    if not valid_words:
        return "No words detected.", []
    
    # TODO: Add your LLM API call here
    # 
    # import anthropic
    # client = anthropic.Anthropic(api_key="your-api-key")
    # 
    # prompt = f"""
    # Convert these sign language words into a natural English sentence.
    # 
    # Words: {valid_words}
    # Confidence scores: {[f"{c*100:.0f}%" for c in valid_conf]}
    # 
    # Rules:
    # 1. Words with confidence below 50% might be errors - use context to decide
    # 2. If a word doesn't fit the context, ignore it
    # 3. Add grammar words (I, want, to, the, a, etc.)
    # 4. Output ONLY the final sentence
    # """
    # 
    # response = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=100,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # 
    # return response.content[0].text, valid_words
    
    # For now, use simple version
    return form_sentence_simple(words, confidences)


# Helper Functions

def extract_hands_only(results):
    """Extract only hand landmarks (63 features)."""
    landmarks = []
    
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    elif results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([1.0 - lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    return landmarks


def predict_sign(frames):
    """Predict sign from recorded frames."""
    frames = np.array(frames)
    
    if len(frames) < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - len(frames), 63))
        frames = np.vstack([frames, padding])
    frames = frames[:MAX_FRAMES]
    
    X = frames.reshape(1, MAX_FRAMES, 63)
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
    
    # Store collected words and confidences
    collected_words = []
    collected_confidences = []
    
    # Last prediction display
    last_prediction = None
    last_confidence = None
    
    # Generated sentence
    generated_sentence = None
    
    print("\n" + "=" * 60)
    print("V6 SIGN LANGUAGE TO SENTENCE SYSTEM")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Start/Stop recording a sign")
    print("  ENTER - Form sentence from collected words")
    print("  C     - Clear all words and start over")
    print("  Q     - Quit")
    print("=" * 60 + "\n")
    
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
            
            # Hand detection indicator
            if results.right_hand_landmarks:
                hand_status = "RIGHT HAND"
                status_color = (0, 255, 0)
            elif results.left_hand_landmarks:
                hand_status = "LEFT HAND (flipped)"
                status_color = (0, 255, 255)
            else:
                hand_status = "NO HAND"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, hand_status, (frame.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            landmarks = extract_hands_only(results)
            
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
            
            # Last prediction display
            if last_prediction:
                if last_confidence > 0.7:
                    color = (0, 255, 0)
                elif last_confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                    
                cv2.putText(frame, f"Last: {last_prediction} ({last_confidence*100:.0f}%)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display collected words
            y_pos = 130
            cv2.putText(frame, "Collected Words:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
            if collected_words:
                # Show words in a row
                words_display = ""
                for i, (word, conf) in enumerate(zip(collected_words, collected_confidences)):
                    words_display += f"{word}({conf*100:.0f}%) "
                    if len(words_display) > 50:  # Wrap to next line
                        cv2.putText(frame, words_display, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_pos += 25
                        words_display = ""
                
                if words_display:
                    cv2.putText(frame, words_display, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25
            else:
                cv2.putText(frame, "(none)", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                y_pos += 25
            
            # Display generated sentence
            y_pos += 20
            cv2.putText(frame, "Sentence:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
            if generated_sentence:
                # Word wrap the sentence
                words = generated_sentence.split()
                line = ""
                for word in words:
                    if len(line + word) > 40:
                        cv2.putText(frame, line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_pos += 30
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    cv2.putText(frame, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "(Press ENTER to form sentence)", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Controls reminder at bottom
            cv2.putText(frame, "SPACE: Record | ENTER: Form Sentence | C: Clear | Q: Quit", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('V6 Sign to Sentence', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE - Start/Stop recording
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
                        
                        # Add to collected words
                        collected_words.append(sign)
                        collected_confidences.append(conf)
                        
                        print(f"\n{'='*40}")
                        print(f"  PREDICTION: {sign}")
                        print(f"  CONFIDENCE: {conf*100:.1f}%")
                        print(f"  Total words: {len(collected_words)}")
                        print(f"{'='*40}\n")
                        
                        # Clear generated sentence when new word added
                        generated_sentence = None
                    else:
                        print(f" Too short. Need {MIN_FRAMES}+ frames.\n")
                    
                    recorded_frames = []
            
            # ENTER - Form sentence
            elif key == 13:  # Enter key
                if collected_words:
                    print("\n" + "="*50)
                    print("FORMING SENTENCE...")
                    print(f"Words: {collected_words}")
                    print(f"Confidences: {[f'{c*100:.0f}%' for c in collected_confidences]}")
                    
                    generated_sentence, filtered_words = form_sentence_simple(
                        collected_words, collected_confidences
                    )
                    
                    print(f"\n SENTENCE: {generated_sentence}")
                    print(f"   (Used words: {filtered_words})")
                    print("="*50 + "\n")
                else:
                    print(" No words collected. Record some signs first.\n")
            
            # C - Clear all
            elif key == ord('c'):
                collected_words = []
                collected_confidences = []
                generated_sentence = None
                last_prediction = None
                last_confidence = None
                print("🗑️ Cleared all words.\n")
            
            # Q - Quit
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()


