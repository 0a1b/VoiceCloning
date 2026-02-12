import argparse
import os
import sys
import torch
import soundfile as sf
import numpy as np
import torchaudio as ta

# --- AUTO-CONFIGURATION FOR RTX 50-SERIES / CUDA 13 ---
# The nightly PyTorch build for Blackwell needs help finding libnvrtc-builtins.so.13.0
# We attempt to find it in the current venv and add it to LD_LIBRARY_PATH programmatically.
try:
    # Get the site-packages directory relative to the running python interpreter
    # Expected structure: venv/bin/python -> venv/lib/pythonX.Y/site-packages/nvidia/cu13/lib
    import site
    site_packages = site.getsitepackages()[0] # Usually the first one is the venv site-packages
    cu13_lib = os.path.join(site_packages, "nvidia/cu13/lib")
    
    if os.path.exists(cu13_lib):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cu13_lib not in current_ld:
            # We must fail-safe restart the process to ensure the dynamic linker sees the new var
            print(f"[System] Restarting process with configured LD_LIBRARY_PATH for CUDA 13...")
            new_env = os.environ.copy()
            new_env["LD_LIBRARY_PATH"] = f"{cu13_lib}:{current_ld}"
            try:
                os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
            except Exception as e:
                print(f"[System] Execve failed: {e}. Falling back to os.environ (may fail).")
                os.environ["LD_LIBRARY_PATH"] = new_env["LD_LIBRARY_PATH"]

except Exception as e:
    # Don't crash if this fails, just warn
    print(f"[System] Note: Could not auto-configure CUDA 13 paths: {e}")

# --- TORCHAUDIO NIGHTLY FIX ---
# Replace torchaudio I/O with soundfile to avoid TorchCodec dependency
def custom_load(filepath, **kwargs):
    # Hack: If filepath is already a (tensor, sr) tuple, just return it.
    # This allows injecting pre-loaded audio into libraries that call load() blindly.
    if isinstance(filepath, tuple):
        return filepath
        
    data, sr = sf.read(filepath)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0) # (L) -> (1, L)
    else:
        tensor = tensor.t() # (L, C) -> (C, L)
    return tensor, sr

def custom_save(filepath, src, sample_rate, **kwargs):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    # torchaudio uses (C, L), soundfile uses (L, C). Transpose if C < L.
    if src.ndim == 2 and src.shape[0] < src.shape[1]: 
        src = src.T
    sf.write(filepath, src, sample_rate)

ta.load = custom_load
ta.save = custom_save
print("[System] Patched torchaudio I/O to use soundfile.")

import re

def main():
    parser = argparse.ArgumentParser(description="Multi-Engine Voice Cloning CLI")
    parser.add_argument("--engine", type=str, default="voxcpm", choices=["voxcpm", "chatterbox", "f5-tts", "qwen3-tts"], help="Cloning engine to use")
    parser.add_argument("--text", type=str, required=True, help="The text to synthesize")
    parser.add_argument("--reference", type=str, default="Tywin Lannister.wav", help="Path to the reference audio file")
    parser.add_argument("--output", type=str, default="cloned_output.wav", help="Path to the output wav file")
    parser.add_argument("--cfg", type=float, default=2.5, help="CFG value for guidance")
    parser.add_argument("--steps", type=int, default=20, help="Inference timesteps")
    parser.add_argument("--language", type=str, default="auto", help="Synthesis language (e.g., 'en', 'zh', 'de', 'es', 'it', 'nl')")
    parser.add_argument("--ref_language", type=str, default="auto", help="ASR language for the reference audio (defaults to auto-detect)")
    parser.add_argument("--prompt_text", type=str, default=None, help="Manually provide the transcript of the reference audio (skips ASR)")
    parser.add_argument("--threshold", type=float, default=6.0, help="Ratio threshold for badcase detection (lower is stricter)")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for badcase detection")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu). If None, auto-select.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens for Qwen3-TTS")
    parser.add_argument("--mode", type=str, default="clone", choices=["clone", "design", "custom"], help="Qwen3-TTS mode")
    parser.add_argument("--voice_description", type=str, default=None, help="Voice description for Qwen3-TTS design mode")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker name for Qwen3-TTS custom mode")
    parser.add_argument("--instruct", type=str, default=None, help="Style instruction for Qwen3-TTS custom mode")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["0.6B", "1.7B"], help="Qwen3-TTS model size")
    parser.add_argument("--x_vector_only", action="store_true", help="Use only speaker embedding for Qwen3-TTS cloning (skips ICL/ref_text)")
    parser.add_argument("--temp", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--ref_start", type=float, default=0.0, help="Start time (seconds) in reference audio")
    parser.add_argument("--ref_len", type=float, default=None, help="Duration (seconds) to use from reference audio (overrides default limits)")
    parser.add_argument("--expressive", action="store_true", help="Enable expressive mode (higher variability/intonation)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (1.0 = normal, 0.8 = slower, 1.2 = faster)")
    
    args = parser.parse_args()
    
    # Auto-tune for expressiveness if requested
    if args.expressive:
        if args.temp == 0.9: args.temp = 1.15
        if args.top_p == 1.0: args.top_p = 0.95
        print(f"[System] Expressive Mode ENABLED: temp={args.temp}, top_p={args.top_p}")

    if not os.path.exists(args.reference):
        print(f"Error: Reference audio '{args.reference}' not found.")
        return

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using initial device: {device} | Engine: {args.engine}")

    # 1. Handle Prompt Text (Only for VoxCPM/F5-TTS as they require it for zero-shot)
    prompt_text = args.prompt_text
    
    # Pre-processing: Truncate Reference Audio for F5-TTS
    # F5-TTS and Qwen3-TTS both benefit from shorter, high-quality references.
    # We restrict reference to 15s max to ensure stability.
    if args.engine in ["f5-tts", "qwen3-tts"]:
        try:
            info = sf.info(args.reference)
            limit = args.ref_len if args.ref_len else (25.0 if args.engine == "qwen3-tts" else 15.0)
            
            # If user specified start or if duration > limit, we slice
            if args.ref_start > 0 or info.duration > limit:
                print(f"[System] Selecting reference window: {args.ref_start}s to {args.ref_start + limit}s...")
                data, sr = sf.read(args.reference, start=int(args.ref_start * info.samplerate), frames=int(limit * info.samplerate))
                
                # Save to temp file
                
                # Save to temp file
                import tempfile
                temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                sf.write(temp_ref, data, sr)
                print(f"[System] Truncated reference saved to: {temp_ref}")
                args.reference = temp_ref # SWAP POINTER
        except Exception as e:
            print(f"[Warning] Failed to truncate audio: {e}")

    if args.engine in ["voxcpm", "f5-tts", "qwen3-tts"]:
        if not prompt_text:
            import whisper
            import librosa
            asr_model = whisper.load_model("medium")
            whisper_lang = None if args.ref_language == "auto" else args.ref_language
            print(f"Transcribing reference '{args.reference}' using Whisper (ASR Language: {args.ref_language})...")
            
            try:
                # Load with librosa to 16k directly to bypass ffmpeg
                audio_array, _ = librosa.load(args.reference, sr=16000, mono=True)
                result = asr_model.transcribe(audio_array, language=whisper_lang)
            except Exception as e:
                print(f"[Warning] Manual audio loading failed, attempting standard load provided by Whisper (requires ffmpeg): {e}")
                result = asr_model.transcribe(args.reference, language=whisper_lang)
                
            prompt_text = result["text"].strip()
            print(f"Detected Language: {result['language']}")
            print(f"Recognized Prompt Text: {prompt_text}")

    # 2. Engine Specific Logic with Fallback Handling
    try:
        if args.engine == "voxcpm":
            print(f"Initializing VoxCPM on {device}...")
            import voxcpm
            model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
            # VoxCPM handles device internally or doesn't expose .to()
            
            print(f"Generating voice for: '{args.text[:50]}...'")
            wav = model.generate(
                text=args.text,
                prompt_wav_path=args.reference,
                prompt_text=prompt_text,
                cfg_value=args.cfg,
                inference_timesteps=args.steps,
                normalize=True,
                denoise=True,
                retry_badcase_ratio_threshold=args.threshold,
                retry_badcase_max_times=args.retries,
            )
            sf.write(args.output, wav, model.tts_model.sample_rate)
            
        elif args.engine == "chatterbox":
            print(f"Initializing Chatterbox Turbo on {device}...")
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            
            print(f"Generating voice for: '{args.text[:50]}...'")
            wav = model.generate(
                args.text, 
                audio_prompt_path=args.reference
            )
            # Convert to numpy for soundfile
            if hasattr(wav, 'numpy'):
                wav = wav.squeeze().detach().cpu().numpy()
            sf.write(args.output, wav, model.sr)
            
        elif args.engine == "f5-tts":
            from src.f5_engine import F5TTSWrapper
            # Initialize with device and correct model type
            m_type = "german" if args.language.lower() in ["de", "german", "de-de"] else "english"
            engine = F5TTSWrapper(device=device, model_type=m_type)
            # F5-TTS benefits greatly from the prompt text which we already have (prompt_text)
            wav, sr = engine.generate(
                text=args.text,
                ref_audio=args.reference,
                ref_text=prompt_text
            )
            
            # Post-processing: Normalize to prevent clipping
            if isinstance(wav, np.ndarray):
                 max_val = np.max(np.abs(wav))
                 if max_val > 0.95:
                     wav = wav / max_val * 0.95
            
            # Force PCM_16 for compatibility
            sf.write(args.output, wav, sr, subtype='PCM_16')
            
        elif args.engine == "qwen3-tts":
            from src.qwen3_engine import Qwen3EngineWrapper
            # Determine model type based on mode
            q_type = {"clone": "Base", "design": "VoiceDesign", "custom": "CustomVoice"}.get(args.mode, "Base")
            engine = Qwen3EngineWrapper(device=device, model_size=args.model_size, model_type=q_type)
            
            if args.mode == "design":
                wav, sr = engine.generate_design(
                    text=args.text,
                    voice_description=args.voice_description,
                    language=args.language if args.language != "auto" else "Auto",
                    max_new_tokens=args.max_tokens
                )
            elif args.mode == "custom":
                wav, sr = engine.generate_custom(
                    text=args.text,
                    speaker=args.speaker,
                    instruct=args.instruct,
                    language=args.language if args.language != "auto" else "Auto",
                    max_new_tokens=args.max_tokens
                )
            else: # clone
                wav, sr = engine.generate_clone(
                    text=args.text,
                    ref_audio=args.reference,
                    ref_text=prompt_text,
                    language=args.language if args.language != "auto" else "Auto",
                    max_new_tokens=args.max_tokens,
                    x_vector_only=args.x_vector_only,
                    temperature=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.penalty
                )
            
            
            # Apply speed adjustment if requested
            if args.speed != 1.0:
                print(f"[System] Adjusting speed to {args.speed}x using high-quality Audiostretchy...")
                try:
                    from audiostretchy.stretch import stretch_audio
                    import tempfile
                    
                    # Audiostretchy works on files. We must save first.
                    temp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    temp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    
                    # Save current in-memory wav to temp input
                    sf.write(temp_in, wav, sr)
                    
                    # Stretch (-r ratio: <1 is length ratio? No, ratio is speed. wait. doc check.)
                    # Audiostretchy uses: ratio (float): The stretch ratio. 
                    # If ratio > 1, the audio is stretched (slower). If ratio < 1, the audio is compressed (faster).
                    # Wait, usually "speed 0.8" means "play at 80% speed" -> longer -> stretch ratio > 1.
                    # Speed = 1 / Ratio. Ratio = 1 / Speed.
                    # e.g. Speed 0.5 (half speed) -> Ratio 2.0 (double length).
                    ratio = 1.0 / args.speed
                    
                    stretch_audio(temp_in, temp_out, ratio=ratio)
                    
                    # Load back
                    wav, sr = sf.read(temp_out)
                    
                    # Cleanup speed temps
                    os.remove(temp_in)
                    os.remove(temp_out)
                    
                except Exception as e:
                    print(f"[Warning] Audiostretchy failed ({e}), falling back to librosa (lower quality)...")
                    import librosa
                    wav = librosa.effects.time_stretch(wav, rate=args.speed)
                
            sf.write(args.output, wav, sr)

            # Cleanup temp reference
            if args.reference.startswith("/tmp/"):
                try: os.remove(args.reference); pass
                except: pass
            
    except Exception as e:
        if "CUDA error" in str(e) and device == "cuda":
            print("\n" + "!"*60)
            print("CUDA ERROR DETECTED (Likely Blackwell/sm_120 architecture mismatch).")
            print("Falling back to CPU for this run...")
            print("!"*60 + "\n")
            
            # Re-run on CPU
            if args.engine == "voxcpm":
                import voxcpm
                model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
                wav = model.generate(
                    text=args.text,
                    prompt_wav_path=args.reference,
                    prompt_text=prompt_text,
                    cfg_value=args.cfg,
                    inference_timesteps=args.steps,
                    normalize=True,
                    denoise=True,
                    retry_badcase_ratio_threshold=args.threshold,
                    retry_badcase_max_times=args.retries,
                )
                sf.write(args.output, wav, model.tts_model.sample_rate)
            elif args.engine == "chatterbox":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                model = ChatterboxTurboTTS.from_pretrained(device="cpu")
                wav = model.generate(args.text, audio_prompt_path=args.reference)
                # Convert to numpy for soundfile
                if hasattr(wav, 'numpy'):
                    wav = wav.squeeze().detach().cpu().numpy()
                sf.write(args.output, wav, model.sr)
            elif args.engine == "f5-tts":
                from src.f5_engine import F5TTSWrapper
                m_type = "german" if args.language.lower() in ["de", "german", "de-de"] else "english"
                engine = F5TTSWrapper(device="cpu", model_type=m_type)
                wav, sr = engine.generate(
                    text=args.text,
                    ref_audio=args.reference,
                    ref_text=prompt_text
                )
                sf.write(args.output, wav, sr)
            elif args.engine == "qwen3-tts":
                from src.qwen3_engine import Qwen3EngineWrapper
                q_type = {"clone": "Base", "design": "VoiceDesign", "custom": "CustomVoice"}.get(args.mode, "Base")
                engine = Qwen3EngineWrapper(device="cpu", model_size=args.model_size, model_type=q_type)
                
                if args.mode == "design":
                    wav, sr = engine.generate_design(
                        text=args.text,
                        voice_description=args.voice_description,
                        language=args.language if args.language != "auto" else "Auto",
                        max_new_tokens=args.max_tokens
                    )
                elif args.mode == "custom":
                    wav, sr = engine.generate_custom(
                        text=args.text,
                        speaker=args.speaker,
                        instruct=args.instruct,
                        language=args.language if args.language != "auto" else "Auto",
                        max_new_tokens=args.max_tokens
                    )
                else: # clone
                    wav, sr = engine.generate_clone(
                        text=args.text,
                        ref_audio=args.reference,
                        ref_text=prompt_text,
                        language=args.language if args.language != "auto" else "Auto",
                        max_new_tokens=args.max_tokens,
                        x_vector_only=args.x_vector_only,
                        temperature=args.temp,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.penalty
                    )
                sf.write(args.output, wav, sr)
        else:
            raise e

    print(f"Success! Output saved to: {args.output}")

if __name__ == "__main__":
    main()
