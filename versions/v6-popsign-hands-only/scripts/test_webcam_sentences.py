import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking, Input
import os
import requests
import subprocess

# CONFIGURATION

WEIGHTS_PATH = "../models/popsign_exact_weights_43.npy"
LABEL_MAP_PATH = "../data/label_map_hands_only_43.npy"
MAX_FRAMES = 60
MIN_FRAMES = 10
MIN_CONFIDENCE = 0.30
OLLAMA_MODEL = "llama3.2"

# UI COLORS (BGR format)

COLORS = {
    'bg_dark': (40, 40, 40),
    'bg_panel': (50, 50, 50),
    'bg_panel_light': (70, 70, 70),
    'text_white': (255, 255, 255),
    'text_gray': (180, 180, 180),
    'accent_green': (0, 230, 118),
    'accent_yellow': (0, 215, 255),
    'accent_red': (80, 80, 255),
    'accent_blue': (255, 180, 0),
    'recording_red': (60, 60, 220),
    'success_green': (100, 200, 100),
}

# UI DRAWING FUNCTIONS

def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw filled rectangle with rounded corners
    if thickness == -1:
        # Main rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, pt1, pt2, color, thickness)


def draw_panel(frame, x, y, width, height, title=None, alpha=0.7):
    """Draw a semi-transparent panel with optional title."""
    overlay = frame.copy()
    
    # Draw rounded rectangle
    draw_rounded_rect(overlay, (x, y), (x + width, y + height), COLORS['bg_panel'], radius=10)
    
    # Blend with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw title if provided
    if title:
        cv2.putText(frame, title, (x + 15, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_gray'], 1)
        # Draw separator line
        cv2.line(frame, (x + 10, y + 35), (x + width - 10, y + 35), COLORS['bg_panel_light'], 1)
    
    return y + 45 if title else y + 10


def draw_word_chip(frame, word, confidence, x, y):
    """Draw a word as a styled chip/tag."""
    # Determine color based on confidence
    if confidence >= 0.70:
        bg_color = (60, 120, 60)  # Green
        text_color = COLORS['accent_green']
    elif confidence >= 0.50:
        bg_color = (60, 100, 120)  # Yellow/orange
        text_color = COLORS['accent_yellow']
    else:
        bg_color = (60, 60, 100)  # Red
        text_color = COLORS['accent_red']
    
    # Calculate text size
    text = f"{word}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    chip_width = text_width + 20
    chip_height = text_height + 16
    
    # Draw chip background
    draw_rounded_rect(frame, (x, y), (x + chip_width, y + chip_height), bg_color, radius=8)
    
    # Draw text
    cv2.putText(frame, text, (x + 10, y + text_height + 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return chip_width + 10  # Return width for next chip positioning


def draw_progress_bar(frame, x, y, width, height, progress, color):
    """Draw a modern progress bar."""
    # Background
    draw_rounded_rect(frame, (x, y), (x + width, y + height), COLORS['bg_panel_light'], radius=5)
    
    # Progress fill
    if progress > 0:
        fill_width = int(width * min(progress, 1.0))
        if fill_width > 10:
            draw_rounded_rect(frame, (x, y), (x + fill_width, y + height), color, radius=5)


def draw_status_indicator(frame, x, y, text, is_active, active_color=None):
    """Draw a status indicator with dot."""
    if active_color is None:
        active_color = COLORS['accent_green']
    
    dot_color = active_color if is_active else COLORS['text_gray']
    cv2.circle(frame, (x + 8, y + 8), 6, dot_color, -1)
    cv2.putText(frame, text, (x + 22, y + 13),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_white'], 1)


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
mp_drawing_styles = mp.solutions.drawing_styles

# Text-to-Speech

def speak_sentence(sentence):
    """Use macOS built-in text-to-speech."""
    try:
        subprocess.run(['say', '-r', '150', sentence], check=True)
        print(f"   Spoke: {sentence}")
    except Exception as e:
        print(f"   Speech error: {e}")

# Sentence Formation Functions

def check_ollama_running():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False


def form_sentence_simple(words, confidences):
    filtered_words = [w for w, c in zip(words, confidences) if c >= MIN_CONFIDENCE]
    if not filtered_words:
        return "No words detected.", []
    sentence = " ".join(filtered_words).capitalize() + "."
    return sentence, filtered_words


def form_sentence_with_ollama(words, confidences):
    if not words:
        return "No words detected.", []
    
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
    
    all_words = list(words)
    
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
                'options': {'temperature': 0.5, 'num_predict': 80, 'top_p': 0.9}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            sentence = result['response'].strip()
            sentence = sentence.split('\n')[0]
            sentence = sentence.replace('"', '').replace("'", "")
            
            prefixes_to_remove = [
                "Here's the sentence:", "Here is the sentence:",
                "Here's a natural sentence:", "The sentence is:",
                "Sentence:", "Output:", "Your sentence:",
            ]
            for prefix in prefixes_to_remove:
                if sentence.lower().startswith(prefix.lower()):
                    sentence = sentence[len(prefix):].strip()
            
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            return sentence, all_words
        else:
            return form_sentence_simple(words, confidences)
            
    except Exception as e:
        print(f"    Ollama error: {e}")
        return form_sentence_simple(words, confidences)


# Helper Functions

def extract_hands_only(results):
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
    ollama_available = check_ollama_running()
    if ollama_available:
        print(" Ollama is running")
    else:
        print(" Ollama not running. Using simple sentence formation.")
    
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
    print("  ENTER - Form sentence + Speak")
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
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Draw hand landmarks with custom style
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2)
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2)
                )
            
            landmarks = extract_hands_only(results)
            
            # LEFT PANEL - Status & Recording 
            panel_y = draw_panel(frame, 10, 10, 350, 180, "RECORDING STATUS")
            
            # Hand detection status (swapped because frame is mirrored)
            if results.right_hand_landmarks:
                hand_text = "Left Hand Detected"  # Swapped
                hand_color = COLORS['accent_green']
            elif results.left_hand_landmarks:
                hand_text = "Right Hand Detected"  # Swapped
                hand_color = COLORS['accent_yellow']
            else:
                hand_text = "No Hand Detected"
                hand_color = COLORS['accent_red']
            
            draw_status_indicator(frame, 25, panel_y, hand_text, 
                                 results.right_hand_landmarks or results.left_hand_landmarks,
                                 hand_color)
            panel_y += 30
            
            # Recording status
            if recording:
                recorded_frames.append(landmarks)
                draw_status_indicator(frame, 25, panel_y, f"Recording... ({len(recorded_frames)} frames)", 
                                     True, COLORS['recording_red'])
                panel_y += 30
                
                # Progress bar
                progress = len(recorded_frames) / MAX_FRAMES
                draw_progress_bar(frame, 25, panel_y, 310, 15, progress, COLORS['recording_red'])
                panel_y += 25
            else:
                draw_status_indicator(frame, 25, panel_y, "Press SPACE to record", False)
                panel_y += 30
            
            # Last prediction
            if last_prediction:
                panel_y += 10
                cv2.putText(frame, f"Last: {last_prediction} ({last_confidence*100:.0f}%)", 
                           (25, panel_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent_green'], 2)
            
            #  RIGHT PANEL - AI Status 
            draw_panel(frame, frame_width - 200, 10, 190, 80, "AI STATUS")
            
            draw_status_indicator(frame, frame_width - 185, 55, 
                                 "Ollama: Connected" if ollama_available else "Ollama: Offline",
                                 ollama_available)
            
            #  BOTTOM LEFT PANEL - Collected Words 
            words_panel_height = 120
            panel_y = draw_panel(frame, 10, frame_height - words_panel_height - 60, 
                                500, words_panel_height, f"COLLECTED WORDS ({len(collected_words)})")
            
            if collected_words:
                chip_x = 25
                chip_y = panel_y
                for word, conf in zip(collected_words, collected_confidences):
                    chip_width = draw_word_chip(frame, word, conf, chip_x, chip_y)
                    chip_x += chip_width
                    if chip_x > 470:  # Wrap to next line
                        chip_x = 25
                        chip_y += 40
            else:
                cv2.putText(frame, "No words yet - start signing!", (25, panel_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_gray'], 1)
            
            # BOTTOM RIGHT PANEL - Generated Sentence 
            sentence_panel_width = frame_width - 540
            panel_y = draw_panel(frame, 520, frame_height - words_panel_height - 60,
                                sentence_panel_width, words_panel_height, "GENERATED SENTENCE")
            
            if generated_sentence:
                # Word wrap the sentence
                words = generated_sentence.split()
                line = ""
                y_offset = 0
                for word in words:
                    if len(line + word) > 40:
                        cv2.putText(frame, line, (535, panel_y + y_offset + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent_green'], 2)
                        y_offset += 30
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    cv2.putText(frame, line, (535, panel_y + y_offset + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent_green'], 2)
            else:
                cv2.putText(frame, "Press ENTER to generate sentence", (535, panel_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_gray'], 1)
            
            #  BOTTOM BAR - Controls 
            bar_y = frame_height - 50
            
            # Dark background bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, bar_y), (frame_width, frame_height), COLORS['bg_dark'], -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            
            # Top border line
            cv2.line(frame, (0, bar_y), (frame_width, bar_y), COLORS['bg_panel_light'], 1)
            
            controls = [
                ("SPACE", "Record"),
                ("ENTER", "Generate & Speak"),
                ("C", "Clear"),
                ("Q", "Quit")
            ]
            
            # Calculate spacing
            available_width = frame_width - 180  # Leave space for app name
            spacing = available_width // len(controls)
            
            x_pos = 15
            for key, action in controls:
                # Key box with border
                key_box_width = len(key) * 13 + 16
                cv2.rectangle(frame, (x_pos, bar_y + 10), (x_pos + key_box_width, bar_y + 40), 
                             COLORS['bg_panel_light'], -1)
                cv2.rectangle(frame, (x_pos, bar_y + 10), (x_pos + key_box_width, bar_y + 40), 
                             COLORS['text_gray'], 1)
                cv2.putText(frame, key, (x_pos + 8, bar_y + 31),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_white'], 1)
                
                # Action text
                cv2.putText(frame, action, (x_pos + key_box_width + 8, bar_y + 31),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_gray'], 1)
                
                x_pos += spacing
            
            # App title (right side)
            cv2.putText(frame, "SignSpeak AI", (frame_width - 145, bar_y + 33),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent_blue'], 2)

            # ADD THESE TWO LINES:
            cv2.imshow('SignSpeak AI - Sign Language to Speech', frame)
            key = cv2.waitKey(1) & 0xFF


            # SPACE - Record
            if key == ord(' '):
                if not recording:
                    recording = True
                    recorded_frames = []
                    print(" Recording...")
                else:
                    recording = False
                    print(f" Stopped ({len(recorded_frames)} frames)")
                    
                    if len(recorded_frames) >= MIN_FRAMES:
                        sign, conf, _ = predict_sign(recorded_frames)
                        last_prediction = sign
                        last_confidence = conf
                        
                        collected_words.append(sign)
                        collected_confidences.append(conf)
                        
                        print(f"   Added: {sign} ({conf*100:.0f}%)")
                        generated_sentence = None
                    else:
                        print(f"   Too short")
                    
                    recorded_frames = []
            
            # ENTER - Form sentence and speak
            elif key == 13:
                if collected_words:
                    print("\n Forming sentence...")
                    
                    if ollama_available:
                        generated_sentence, _ = form_sentence_with_ollama(
                            collected_words, collected_confidences
                        )
                    else:
                        generated_sentence, _ = form_sentence_simple(
                            collected_words, collected_confidences
                        )
                    
                    print(f" {generated_sentence}")
                    speak_sentence(generated_sentence)
                else:
                    print(" No words collected")
            
            # C - Clear
            elif key == ord('c'):
                collected_words = []
                collected_confidences = []
                generated_sentence = None
                last_prediction = None
                last_confidence = None
                print(" Cleared")
            
            # Q - Quit
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
