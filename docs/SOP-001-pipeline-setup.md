# SOP-001: Pipeline Setup and Deployment

**Project:** VoiceForAll — ASL Sign Language to Speech Translation  
**Version:** V6 (Final)  
**Last Updated:** April 2026  
**Author:** Sukhnaaz Kaur

---

## Purpose

This SOP defines the standard procedure for setting up and running the VoiceForAll V6 inference pipeline from scratch. It covers both local setup and containerised deployment via Docker. Following this procedure exactly will produce a reproducible, working system on any compatible machine.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 | Required for all local setup |
| Docker Desktop | Latest | Required for containerised deployment |
| Ollama | Latest | Required for sentence generation |
| macOS | Any recent | Required for text-to-speech (`say` command) |
| Webcam | Any | Built-in or external |

---

## Option A: Local Setup (macOS)

### Step 1 — Clone the repository
```bash
git clone https://github.com/SukhnaazRandhawa/VoiceForAll.git
cd VoiceForAll
```

### Step 2 — Create and activate a virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install tensorflow-macos mediapipe opencv-python numpy requests keras protobuf
```

### Step 4 — Install and start Ollama
```bash
# Download Ollama from https://ollama.ai
ollama pull llama3.2
ollama serve
```

### Step 5 — Run the inference system
```bash
cd versions/v6-popsign-hands-only/scripts
python test_webcam_sentences.py
```

### Expected output

A webcam window titled "VoiceForAll - Sign Language to Speech" will open.

---

## Option B: Containerised Deployment (Docker)

### Step 1 — Clone the repository
```bash
git clone https://github.com/SukhnaazRandhawa/VoiceForAll.git
cd VoiceForAll
```

### Step 2 — Build and start the containers
```bash
docker compose up
```
This will start two services:
- `voiceforall_ollama` — runs Llama 3.2 locally
- `voiceforall_inference` — runs the full ML inference pipeline

### Step 3 — Expected output

### Notes on macOS Docker limitations
Webcam access and audio output (`say`) are not available inside Docker containers on macOS. The container validates and runs the full inference pipeline successfully — webcam input and speech output are handled by the host machine in live demo scenarios. On a Linux host or cloud VM, full webcam passthrough is supported.

---

## Controls (Both Options)

| Key | Action |
|---|---|
| `SPACE` | Start / stop recording a sign |
| `ENTER` | Generate sentence from collected signs + speak aloud |
| `C` | Clear all collected words |
| `Q` | Quit |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `Error: Could not open webcam` | No webcam detected | Check webcam is connected and not in use |
| `Ollama not running` | Ollama service not started | Run `ollama serve` in a separate terminal |
| `Error loading weights` | Model file missing | Ensure `.npy` files are present in `models/` folder |
| `ModuleNotFoundError` | Missing dependency | Re-run the pip install step |

---

## Verification Checklist

Before considering setup complete, confirm the following:

- [ ] 248 signs load successfully
- [ ] Model weights load without error
- [ ] Webcam window opens
- [ ] Hand landmarks appear on screen when hand is shown
- [ ] Recording works when SPACE is pressed
- [ ] Sentence is generated when ENTER is pressed
- [ ] Speech output works (macOS local only)