# 🦁
- **VoxCPM**: Solid all-around voice cloning with robust detection.
- **Chatterbox**: High-speed, high-fidelity cloning.
- **F5-TTS**: State-of-the-art flow-matching TTS, excellent for German and English.
- **Qwen3-TTS**: Next-gen multimodal TTS with Voice Clone, Voice Design, and Custom Voice modes. (Requires `Qwen3Venv312`).
This setup allows for dynamic voice cloning of any voice (defaulting to Tywin Lannister) via CLI or API.

### Generating Audio via CLI

Use the parameterized `clone.py` script. The system now features **Automatic CPU Fallback**—if your GPU is incompatible (like the RTX 50-series), it will detect the error and switch to CPU automatically.

```bash
# Using the default VoxCPM engine
/home/jj/Antigravity/Voice\ Cloing\ App/VoxCPM/venv/bin/python /home/jj/Antigravity/Voice\ Cloing\ App/clone.py \
  --engine voxcpm \
  --text "Hello world" \
  --reference "Tywin Lannister.wav"

# Force CPU execution (useful for new GPUs)
/home/jj/Antigravity/Voice\ Cloing\ App/VoxCPM/venv/bin/python /home/jj/Antigravity/Voice\ Cloing\ App/clone.py \
  --device cpu \
  --text "Running on CPU"

# Qwen3-TTS Voice Clone (default)
./Qwen3Venv312/bin/python clone.py --engine qwen3-tts --text "Hello world" --reference my_voice.wav --output qwen.wav

# Qwen3-TTS Voice Design
./Qwen3Venv312/bin/python clone.py --engine qwen3-tts --mode design --voice_description "A deep, gravelly male voice with a slight accent" --text "I am the voice you designed."

# Qwen3-TTS Custom Voice (Pre-trained characters)
./Qwen3Venv312/bin/python clone.py --engine qwen3-tts --mode custom --speaker "Gentle Girl" --text "Hello, I am a custom voice."
```
# Using F5-TTS for German (Best Quality)
/home/jj/Antigravity/Voice\ Cloing\ App/VoxCPM/venv/bin/python /home/jj/Antigravity/Voice\ Cloing\ App/clone.py \
  --engine f5-tts \
  --language de \
  --text "Guten Tag, wie geht es Ihnen?" \
  --reference "German_Sample.wav"
```
### 2. Choose Your Engine

| Feature | **VoxCPM** (Default) | **Chatterbox Turbo** | **F5-TTS** (New) |
| :--- | :--- | :--- | :--- |
| **Best For** | **Multilingual** / Cross-lingual Cloning | **English Only** High-Speed / Real-time | **German / Multilingual** High-Fidelity Zero-Shot |
| **Language Support** | 100+ Languages (Auto-detected) | **English Only** | **Multilingual** (German, English, French, Etc.) |
| **Transcription** | Required (Auto with Whisper) | Not Required (Uses Audio Tokens) | Required (Auto with Whisper) |
| **Speed** | Standard | Ultra-Fast (Turbo) | Slower (Diffusion-based) |

**Options:**
- `--engine`: `voxcpm` (default), `chatterbox`, or `f5-tts`.
- `--device`: `cuda`, `cpu`, or `None` (auto).
- `--text`: The text to synthesize.
- `--reference`: Path to the reference audio.
- `--output`: Name of the output file.
- `--language`: (VoxCPM) ASR language.
- `--cfg` / `--steps`: (VoxCPM) generation settings.

---

### API Usage (for n8n)

The API supports engine selection via the `engine` field.

**Endpoint:** `POST /generate`

**Example JSON Payload:**
```json
{
  "engine": "chatterbox",
  "text": "Greetings from the new engine! [chuckle]",
  "prompt_wav": "Tywin Lannister.wav",
  "language": "en"
}
```

---

### Troubleshooting: RTX 50-series / Blackwell Architecture

If you have an **RTX 50-series GPU (sm_120)**, the system is fully compatible and includes **Automatic Configuration**:

1.  **Auto-Native Support**: The script automatically detects the PyTorch nightly libraries and configures `LD_LIBRARY_PATH` for you. You should see `[System] Auto-configured LD_LIBRARY_PATH...` in the logs.
2.  **Manual Fallback**: If for some reason the auto-config fails, you can manually export:
    ```bash
    export LD_LIBRARY_PATH="/home/jj/Antigravity/Voice Cloing App/VoxCPM/venv/lib/python3.13/site-packages/nvidia/cu13/lib/:$LD_LIBRARY_PATH"
    ```
3.  **CPU Safety Net**: If GPU execution still fails, the system automatically falls back to CPU to ensure you always get an output.

---

## 👂 Transcription (ASR)
The system now uses **OpenAI Whisper (medium model)**, which is significantly more accurate for German and other non-English languages than the original tool.

- **Auto-Detection**: It will automatically try to detect the language.
- **Manual Select**: If the detection fails, use `--language de` (CLI) or `"language": "de"` (API).

---

## 🛠️ Troubleshooting

### "Badcase detected" (Error)
VoxCPM checks if the generated audio is too long compared to the text. If you get this message:
- **Increase Threshold**: Use `--threshold 8.0` (CLI) or `"threshold": 8.0` (API).
- **Increase Retries**: Use `--retries 5`.
- **Lower CFG**: Sometimes a very high CFG value (like 3.0+) causes the model to "loop" or hallucinate. Try 2.0 or 1.5.

> [!IMPORTANT]
> The Whisper `medium` model requires about 5GB of VRAM. Your RTX 5060 Ti can easily handle this alongside the VoxCPM model.

---


##  Local Installation


### 1. Prerequisite: Python 3.11
Ensure you have Python 3.11 installed and active.

**Using Pyenv Virtualenv (Recommended):**
```bash
# 1. Install Python 3.11.9
pyenv install 3.11.9

# 2. Create a named virtualenv
pyenv virtualenv 3.11.9 voice-cloning

# 3. Activate it
pyenv activate voice-cloning
```

### 2. Run Setup Script
Once Python 3.11 is active, simply run the setup script. It will create a standard `venv` using your active Python interpreter.

```bash
bash setup.sh
```

### 3. Run the App
```bash
# Activate the environment
source venv/bin/activate

# Run
python clone.py --text "Hello from local install" ...
```
