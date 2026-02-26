import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking, Input
import os
import requests

# CONFIGURATION

WEIGHTS_PATH = "../models/popsign_exact_weights_43.npy"
LABEL_MAP_PATH = "../data/label_map_hands_only_43.npy"
MAX_FRAMES = 60
MIN_FRAMES = 10
MIN_CONFIDENCE = 0.30
OLLAMA_MODEL = "llama3.2"

# Load Label Map

print(f"Loading label map from {LABEL_MAP_PATH}...")
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

if isinstance(list(label_map.keys())[0], int):
    idx_to_sign = label_map
else:
    idx_to_sign = {v: k for k, v in label_map.items()}

NUM_CLASSES = len(idx_to_sign)
print(f" {NUM_CLASSES} signs loaded")

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

# Sentence Formation Functions

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False


def form_sentence_simple(words, confidences):
    """Simple fallback sentence formation."""
    filtered_words = [w for w, c in zip(words, confidences) if c >= MIN_CONFIDENCE]
    if not filtered_words:
        return "No words detected.", []
    sentence = " ".join(filtered_words).capitalize() + "."
    return sentence, filtered_words


def form_sentence_with_ollama(words, confidences):
    """
    Form natural sentence using Ollama LLM.
    Smart filtering: removes words that have LOW confidence AND don't fit context.
    Keeps low confidence words if they fit the sentence contextually.
    """
    
    if not words:
        return "No words detected.", []
    
    # Separate words by confidence level
    high_conf_words = []
    medium_conf_words = []
    low_conf_words = []
    
    for word, conf in zip(words, confidences):
        if conf >= 0.70:
            high_conf_words.append(word)
        elif conf >= 0.50:
            medium_conf_words.append(word)
        else:
            low_conf_words.append(word)
    
    # All words for reference
    all_words = list(words)
    
    print(f"    Sending to Ollama: {all_words}")
    
    # Clear, direct prompt
    prompt = f"""Convert these sign language words into a natural English sentence.

ALL WORDS: {all_words}

Confidence breakdown:
- HIGH (must use): {high_conf_words}
- MEDIUM (use if fits): {medium_conf_words}
- LOW (only if fits context): {low_conf_words}

IMPORTANT RULES:
1. You MUST use ALL high confidence words in your sentence
2. Create ONE natural sentence that includes these words
3. Add grammar words (I, want, to, the, a, is, etc.)
4. If a low confidence word doesn't fit, skip it
5. Return ONLY the sentence - no explanations

Words to use: {all_words}
Your sentence:"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.5,
                    'num_predict': 80,
                    'top_p': 0.9
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            sentence = result['response'].strip()
            
            # Clean up - take only first line
            sentence = sentence.split('\n')[0]
            sentence = sentence.replace('"', '').replace("'", "")
            
            # Remove common prefixes
            prefixes_to_remove = [
                "Here's the sentence:",
                "Here is the sentence:",
                "Here's a natural sentence:",
                "The sentence is:",
                "Sentence:",
                "Output:",
                "Your sentence:",
            ]
            for prefix in prefixes_to_remove:
                if sentence.lower().startswith(prefix.lower()):
                    sentence = sentence[len(prefix):].strip()
            
            # Clean up any leaked labels
            sentence = sentence.replace("(confident)", "")
            sentence = sentence.replace("(medium)", "")
            sentence = sentence.replace("(uncertain)", "")
            sentence = sentence.strip()
            
            # Ensure proper ending
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Check if high confidence words are included
            missing = [w for w in high_conf_words if w.lower() not in sentence.lower()]
            if missing:
                print(f"    Warning: Missing words: {missing}")
            
            return sentence, all_words
        else:
            print(f"    Ollama returned status: {response.status_code}")
            return form_sentence_simple(words, confidences)
            
    except Exception as e:
        print(f"    Ollama error: {e}")
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
    # Check Ollama status
    ollama_available = check_ollama_running()
    if ollama_available:
        print(" Ollama is running")
    else:
        print(" Ollama not running. Start with: ollama serve")
        print("   Using simple sentence formation instead.\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recording = False
    recorded_frames = []
    collected_words = []
    collected_confidences = []
    last_prediction = None
    last_confidence = None
    generated_sentence = None
    
    print("\n" + "=" * 60)
    print("V6 SIGN LANGUAGE TO SENTENCE SYSTEM")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Record a sign")
    print("  ENTER - Form sentence (using Ollama AI)")
    print("  C     - Clear all words")
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
            
            # Draw landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Hand status
            if results.right_hand_landmarks:
                hand_status = "RIGHT HAND"
                status_color = (0, 255, 0)
            elif results.left_hand_landmarks:
                hand_status = "LEFT (flipped)"
                status_color = (0, 255, 255)
            else:
                hand_status = "NO HAND"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, hand_status, (frame.shape[1] - 200, 30),
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
            
            # Last prediction
            if last_prediction:
                color = (0, 255, 0) if last_confidence > 0.7 else (0, 255, 255) if last_confidence > 0.4 else (0, 0, 255)
                cv2.putText(frame, f"Last: {last_prediction} ({last_confidence*100:.0f}%)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Collected words
            y_pos = 130
            cv2.putText(frame, f"Words ({len(collected_words)}):", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
            if collected_words:
                words_line = ""
                for word, conf in zip(collected_words, collected_confidences):
                    words_line += f"[{word}] "
                    if len(words_line) > 50:
                        cv2.putText(frame, words_line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_pos += 25
                        words_line = ""
                if words_line:
                    cv2.putText(frame, words_line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25
            
            # Generated sentence
            y_pos += 20
            cv2.putText(frame, "Sentence:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
            
            if generated_sentence:
                # Word wrap
                words = generated_sentence.split()
                line = ""
                for word in words:
                    if len(line + word) > 45:
                        cv2.putText(frame, line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y_pos += 35
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    cv2.putText(frame, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press ENTER to form sentence", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Controls
            cv2.putText(frame, "SPACE: Record | ENTER: Sentence | C: Clear | Q: Quit", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Ollama status
            ollama_status = "AI: ON" if ollama_available else "AI: OFF"
            ollama_color = (0, 255, 0) if ollama_available else (0, 0, 255)
            cv2.putText(frame, ollama_status, (frame.shape[1] - 100, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ollama_color, 2)
            
            cv2.imshow('V6 Sign to Sentence', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE - Record
            if key == ord(' '):
                if not recording:
                    recording = True
                    recorded_frames = []
                    print(" Recording...")
                else:
                    recording = False
                    print(f"  Stopped ({len(recorded_frames)} frames)")
                    
                    if len(recorded_frames) >= MIN_FRAMES:
                        sign, conf, _ = predict_sign(recorded_frames)
                        last_prediction = sign
                        last_confidence = conf
                        
                        collected_words.append(sign)
                        collected_confidences.append(conf)
                        
                        print(f"    Added: {sign} ({conf*100:.0f}%)")
                        print(f"    Total words: {len(collected_words)}\n")
                        
                        generated_sentence = None
                    else:
                        print(f"    Too short\n")
                    
                    recorded_frames = []
            
            # ENTER - Form sentence
            elif key == 13:
                if collected_words:
                    print("\n" + "="*50)
                    print(" FORMING SENTENCE...")
                    print(f"   Words with confidence:")
                    for word, conf in zip(collected_words, collected_confidences):
                        if conf >= 0.70:
                            status = " confident"
                        elif conf >= 0.50:
                            status = " medium"
                        else:
                            status = " uncertain"
                        print(f"      {word}: {conf*100:.0f}% ({status})")
                    
                    if ollama_available:
                        generated_sentence, used_words = form_sentence_with_ollama(
                            collected_words, collected_confidences
                        )
                    else:
                        generated_sentence, used_words = form_sentence_simple(
                            collected_words, collected_confidences
                        )
                    
                    print(f"\n    SENTENCE: {generated_sentence}")
                    print("="*50 + "\n")
                else:
                    print(" No words to form sentence.\n")
            
            # C - Clear
            elif key == ord('c'):
                collected_words = []
                collected_confidences = []
                generated_sentence = None
                last_prediction = None
                last_confidence = None
                print(" Cleared all.\n")
            
            # Q - Quit
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()