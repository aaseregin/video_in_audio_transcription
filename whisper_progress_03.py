import argparse
import os
import math
import torch
import whisper
import numpy as np
from scipy import signal
import gc # Explicit import for gc.collect()

# Assuming optimize_memory.py is in the same directory or accessible via PYTHONPATH
try:
    from optimize_memory import optimize_gpu_memory, clear_gpu_memory, get_nvidia_gpu, get_gpu_memory_info
except ImportError:
    print("Warning: optimize_memory.py not found. GPU optimization features will be limited.")
    # Define dummy functions if optimize_memory.py is not available
    def optimize_gpu_memory(): print("optimize_gpu_memory (dummy): not available, basic optimizations might not be applied.")
    def clear_gpu_memory(): 
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("clear_gpu_memory (dummy): called basic torch.cuda.empty_cache() and gc.collect().")
    def get_nvidia_gpu():
        if not torch.cuda.is_available(): raise RuntimeError("CUDA not available")
        for i in range(torch.cuda.device_count()):
            if "nvidia" in torch.cuda.get_device_name(i).lower(): return i
        # If only one CUDA device, assume it's the one, even if name check fails.
        if torch.cuda.device_count() == 1: return 0 
        raise RuntimeError("NVIDIA GPU not found by dummy/basic function, or multiple non-NVIDIA CUDA devices present.")
    def get_gpu_memory_info(): 
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            idx = torch.cuda.current_device()
            print(f"\n=== GPU Memory Info (Device {idx} - {torch.cuda.get_device_name(idx)}) ===")
            print(f"Allocated: {torch.cuda.memory_allocated(idx)/1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved(idx)/1024**2:.2f} MB")
        else:
            print("get_gpu_memory_info (dummy): CUDA not available or no devices.")


def format_timestamp(seconds: float) -> str:
    """Format seconds to [HH:MM:SS] or [MM:SS] timestamp (omit hours if 0)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Предварительная обработка аудио для улучшения качества."""
    audio = audio.copy().astype(np.float32) 
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    nyquist = sample_rate / 2
    cutoff = 100  # Hz
    b, a = signal.butter(4, cutoff/nyquist, btype='high')
    audio = signal.filtfilt(b, a, audio)
    
    return np.ascontiguousarray(audio, dtype=np.float32)

def transcribe_with_progress(audio_path: str, model_name: str, output_formats: tuple, device_preference: str = None):
    model_obj = None 
    device_str = "cpu" 

    if device_preference:
        device_str = device_preference
    else:
        if torch.cuda.is_available():
            try:
                print("Attempting to optimize GPU memory and select NVIDIA GPU...")
                optimize_gpu_memory() 
                
                nvidia_gpu_index = get_nvidia_gpu()
                torch.cuda.set_device(nvidia_gpu_index) 
                device_str = f"cuda:{nvidia_gpu_index}"
                print(f"Successfully set device to NVIDIA GPU: {torch.cuda.get_device_name(nvidia_gpu_index)} ({device_str})")
                if 'get_gpu_memory_info' in globals() and callable(get_gpu_memory_info): get_gpu_memory_info()
            except RuntimeError as e:
                print(f"NVIDIA GPU specific setup failed: {e}.")
                if torch.cuda.device_count() > 0:
                    device_str = "cuda" 
                    current_dev_idx = torch.cuda.current_device() 
                    torch.cuda.set_device(current_dev_idx) 
                    print(f"Using default CUDA device: {torch.cuda.get_device_name(current_dev_idx)} ({device_str})")
                else:
                    device_str = "cpu"
                    print("No CUDA devices found despite CUDA being available check. Using CPU.")
            except NameError: 
                 print("optimize_memory module not fully available. Using generic 'cuda' if available.")
                 device_str = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
                 if device_str == "cuda":
                    current_dev_idx = torch.cuda.current_device()
                    torch.cuda.set_device(current_dev_idx)
                    print(f"Using default CUDA device: {torch.cuda.get_device_name(current_dev_idx)} ({device_str})")
        else:
            print("CUDA not available. Using CPU.")
            device_str = "cpu"
    
    print(f"Selected device: {device_str}")

    print(f"Loading Whisper model '{model_name}' onto {device_str}...")
    model_obj = whisper.load_model(model_name, device=device_str)
    
    if device_str != "cpu":
        print("GPU memory after model load:")
        if 'get_gpu_memory_info' in globals() and callable(get_gpu_memory_info): get_gpu_memory_info()


    print(f"Loading audio file: {audio_path}")
    audio_waveform = whisper.load_audio(audio_path) 
    SAMPLE_RATE = whisper.audio.SAMPLE_RATE
    
    print("Preprocessing audio...")
    audio_processed = preprocess_audio(audio_waveform, SAMPLE_RATE)
    
    total_samples = len(audio_processed)
    total_duration_seconds = total_samples / SAMPLE_RATE

    iteration_chunk_duration_seconds = 30 
    iteration_chunk_size_samples = iteration_chunk_duration_seconds * SAMPLE_RATE
    print(f"Iterating through audio in chunks of {iteration_chunk_duration_seconds} seconds for transcription.")

    base_name = os.path.splitext(audio_path)[0]
    txt_file = open(base_name + ".txt", "w", encoding="utf-8") if "txt" in output_formats else None
    srt_file = open(base_name + ".srt", "w", encoding="utf-8") if "srt" in output_formats else None
    
    detected_language = None
    if model_obj.is_multilingual:
        try:
            sample_for_lang_detect = audio_processed[:min(total_samples, whisper.audio.N_SAMPLES)]
            mel_input_sample = whisper.pad_or_trim(sample_for_lang_detect, length=whisper.audio.N_SAMPLES)
            mel = whisper.log_mel_spectrogram(mel_input_sample).to(model_obj.device)
            
            _, probs = model_obj.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            print(f"Detected language: {detected_language}")
        except Exception as e:
            print(f"Language detection failed: {e}. Proceeding without explicit language hint.")
            detected_language = None
    else:
        print(f"Model '{model_name}' is not multilingual.")
        if ".en" in model_name: 
            detected_language = "en"
            print("Assuming English for .en model.")


    print(f"Transcribing audio ({total_duration_seconds:.1f} seconds)...")
    
    transcribe_kwargs = {"language": detected_language, "beam_size": 1}
    if device_str != "cpu" and hasattr(model_obj, 'dtype') and model_obj.dtype == torch.float16:
        transcribe_kwargs["fp16"] = True 
        print("Using FP16 for transcription operations (model is FP16).")
    else:
        transcribe_kwargs["fp16"] = False


    srt_segment_index = 0
    first_segment_written_to_txt = False 

    for offset_samples in range(0, total_samples, iteration_chunk_size_samples):
        chunk_audio = audio_processed[offset_samples : offset_samples + iteration_chunk_size_samples]
        current_chunk_start_time_seconds = offset_samples / SAMPLE_RATE
        
        result = model_obj.transcribe(chunk_audio, verbose=False, **transcribe_kwargs)

        if not result["segments"]:
            percent_done = min(((offset_samples + len(chunk_audio)) / total_samples) * 100, 100.0)
            end_of_chunk_time = current_chunk_start_time_seconds + (len(chunk_audio) / SAMPLE_RATE)
            print(f"[{format_timestamp(current_chunk_start_time_seconds)} → {format_timestamp(end_of_chunk_time)}] (No speech in chunk) {percent_done:.1f}% done")
            continue

        for segment in result["segments"]:
            segment_start_abs = segment["start"] + current_chunk_start_time_seconds
            segment_end_abs   = segment["end"]   + current_chunk_start_time_seconds
            
            percent_done = min((segment_end_abs / total_duration_seconds) * 100, 100.0)
            print(f"[{format_timestamp(segment_start_abs)} → {format_timestamp(segment_end_abs)}] {percent_done:.1f}% done")

            text_to_write = segment["text"]
            if txt_file:
                if not first_segment_written_to_txt:
                    txt_file.write(text_to_write.lstrip())
                    first_segment_written_to_txt = True
                else:
                    txt_file.write(text_to_write) 
                txt_file.flush()
            
            if srt_file:
                srt_segment_index += 1
                srt_file.write(f"{srt_segment_index}\n")
                srt_file.write(f"{format_timestamp_srt(segment_start_abs)} --> {format_timestamp_srt(segment_end_abs)}\n")
                srt_file.write(text_to_write.strip() + "\n\n")
                srt_file.flush()
        
        if not srt_file and not first_segment_written_to_txt and result["segments"]:
            first_segment_written_to_txt = True # Ensure flag is set even if only TXT output
        
        if device_str != "cpu":
            torch.cuda.empty_cache() # Clear cache after processing each chunk

    if txt_file: txt_file.close()
    if srt_file: srt_file.close()
    print("Transcription complete.")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with progress using OpenAI Whisper.")
    parser.add_argument("--audio_file", "-a", required=True, help="Path to the audio file to transcribe.")
    parser.add_argument("--model", "-m",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3"],
                        required=False,
                        help="Whisper model name to use. If not provided, will be prompted. CRITICAL: If you get OutOfMemoryError, try a smaller model.")
    args = parser.parse_args()

    formats_tuple = ("txt", "srt") 
    
    user_selected_model = args.model
    user_selected_device = None

    # Логика выбора устройства и модели
    ALL_MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3"]

    # 1. Выбор устройства (CPU/GPU)
    while user_selected_device not in ["cpu", "gpu"]:
        choice = input("Использовать CPU или GPU? (cpu/gpu): ").strip().lower()
        if choice in ["cpu", "gpu"]:
            user_selected_device = choice
        else:
            print("Неверный ввод. Пожалуйста, введите 'cpu' или 'gpu'.")

    # 2. Выбор модели
    if not user_selected_model: # Если модель не указана через аргумент --model
        print("\n--- Выбор модели Whisper ---")
        if user_selected_device == "cpu":
            print("Вы выбрали CPU. Выбор модели влияет на скорость и точность:")
            print("  - Меньшие модели: быстрее, но менее точные.")
            print("  - Большие модели: медленнее, но более точные.")
            print("Доступные модели для CPU:")
            for i, model_name in enumerate(ALL_MODELS):
                print(f"  {i+1}. {model_name}")
            
            while True:
                try:
                    model_choice_idx = int(input(f"Введите номер модели (1-{len(ALL_MODELS)}): ")) - 1
                    if 0 <= model_choice_idx < len(ALL_MODELS):
                        user_selected_model = ALL_MODELS[model_choice_idx]
                        break
                    else:
                        print(f"Неверный номер. Введите число от 1 до {len(ALL_MODELS)}.")
                except ValueError:
                    print("Неверный ввод. Пожалуйста, введите число.")
        
        elif user_selected_device == "gpu":
            # Логика выбора модели для GPU будет добавлена здесь
            print("Логика выбора модели для GPU будет реализована позже.")
            # Пока что, если GPU выбран, но модель не задана, используем small по умолчанию или выходим
            # Для простоты пока установим small, потом доработаем
            print("По умолчанию для GPU будет использована модель 'small'. Это будет доработано.")
            user_selected_model = "small" # Временное решение
            if not torch.cuda.is_available():
                print("ВНИМАНИЕ: CUDA недоступна! Выбран GPU, но будет использован CPU.")
                user_selected_device = "cpu" # Переключаемся на CPU, если CUDA нет
                # Повторно предлагаем выбор модели для CPU, так как GPU недоступен
                print("Пожалуйста, выберите модель для CPU:")
                for i, model_name in enumerate(ALL_MODELS):
                    print(f"  {i+1}. {model_name}")
                while True:
                    try:
                        model_choice_idx = int(input(f"Введите номер модели (1-{len(ALL_MODELS)}): ")) - 1
                        if 0 <= model_choice_idx < len(ALL_MODELS):
                            user_selected_model = ALL_MODELS[model_choice_idx]
                            break
                        else:
                            print(f"Неверный номер. Введите число от 1 до {len(ALL_MODELS)}.")
                    except ValueError:
                        print("Неверный ввод. Пожалуйста, введите число.")
    
    # Если модель все еще не выбрана (например, пользователь пропустил ввод для GPU)
    if not user_selected_model:
        print("Модель не была выбрана. Используется модель 'small' по умолчанию.")
        user_selected_model = "small"

    print(f"--- Конфигурация транскрибации ---")
    print(f"Аудиофайл: {args.audio_file}")
    print(f"Выбранное устройство: {user_selected_device}")
    print(f"Выбранная модель: {user_selected_model}")
    print(f"Форматы вывода: {formats_tuple}")
    print("---------------------------------")

    # Адаптируем user_selected_device для передачи в transcribe_with_progress
    # transcribe_with_progress ожидает 'cuda' или 'cpu'
    effective_device_for_function = "cuda" if user_selected_device == "gpu" and torch.cuda.is_available() else "cpu"

    try:
        transcribe_with_progress(args.audio_file, user_selected_model, formats_tuple, effective_device_for_function)
    except torch.cuda.OutOfMemoryError:
        print("\n####################################################################")
        print("CRITICAL ERROR: CUDA Out of Memory!")
        print("Your GPU does not have enough VRAM for the selected Whisper model.")
        print(f"Selected model: '{user_selected_model}' (This model is likely too large for your GPU).")
        print("\nRECOMMENDATION: Use a SMALLER model.")
        print("   Example: python whisper_progress.py --audio_file your_audio.mp3 --model small")
        print("   Available smaller models: 'small', 'base', 'tiny' (and their .en variants).")
        print("\nOther things to check:")
        print(" - Ensure no other GPU-intensive applications are running.")
        print(" - If `optimize_memory.py` was reported as not found, ensure it's in the same directory.")
        print("####################################################################")
    except RuntimeError as e:
        print(f"\n####################################################################")
        print(f"A runtime error occurred: {e}")
        print("This might be related to your CUDA setup, GPU drivers, or PyTorch installation.")
        print("If this is a CUDA/GPU issue, try running `python check_gpu.py` for diagnostics.")
        print("Ensure you have the latest compatible NVIDIA drivers and CUDA toolkit installed for PyTorch.")
        print("####################################################################")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n####################################################################")
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        print("####################################################################")
        import traceback
        traceback.print_exc()