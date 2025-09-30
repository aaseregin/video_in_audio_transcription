# whisper_progress.py

import argparse
import os
import math
import torch
import numpy as np
from scipy import signal
import gc
import librosa

# ПРАВИЛЬНЫЙ ИМПОРТ для faster-whisper
from faster_whisper import WhisperModel

# Утилиты для GPU остаются без изменений
try:
    from optimize_memory import optimize_gpu_memory, clear_gpu_memory, get_nvidia_gpu, get_gpu_memory_info
except ImportError:
    print("Warning: optimize_memory.py not found.")
    def optimize_gpu_memory(): pass
    def clear_gpu_memory(): pass
    def get_nvidia_gpu():
        if not torch.cuda.is_available(): raise RuntimeError("CUDA not available")
        return 0
    def get_gpu_memory_info(): pass

# Константы и утилиты форматирования
SAMPLE_RATE = 16000

def format_timestamp(seconds: float) -> str:
    # ... (код без изменений)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"

def format_timestamp_srt(seconds: float) -> str:
    # ... (код без изменений)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # ... (код без изменений)
    audio = audio.copy().astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0: audio = audio / max_val
    nyquist = sample_rate / 2
    b, a = signal.butter(4, 100/nyquist, btype='high')
    audio = signal.filtfilt(b, a, audio)
    return np.ascontiguousarray(audio, dtype=np.float32)

# ОСНОВНАЯ ФУНКЦИЯ, ИСПРАВЛЕННАЯ ПОД FASTER-WHISPER
def transcribe_with_progress(audio_path: str, model_name: str, output_formats: tuple, device_preference: str = None, task: str = "transcribe", language: str = None):
    device = "cpu"
    compute_type = "int8"

    if device_preference == "cuda" and torch.cuda.is_available():
        try:
            optimize_gpu_memory()
            gpu_index = get_nvidia_gpu()
            device = "cuda"
            compute_type = "float16" # float16 для GPU - максимальная скорость
            print(f"Using GPU device {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
            get_gpu_memory_info()
        except Exception as e:
            print(f"GPU setup failed: {e}. Falling back to CPU.")
    else:
        print("Using CPU.")

    print(f"Loading model '{model_name}' (compute_type: {compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print("Model loaded.")

    total_duration_seconds = librosa.get_duration(filename=audio_path)
    print(f"Transcribing audio ({total_duration_seconds:.1f} seconds)...")

    # model.transcribe возвращает генератор, который мы будем итерировать
    segments, info = model.transcribe(audio_path, beam_size=1, task=task, language=language)

    print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")

    base_name = os.path.splitext(audio_path)[0]
    txt_path = base_name + ".txt"
    srt_path = base_name + ".srt"

    with open(txt_path, "w", encoding="utf-8") if "txt" in output_formats else open(os.devnull, 'w') as txt_file, \
         open(srt_path, "w", encoding="utf-8") if "srt" in output_formats else open(os.devnull, 'w') as srt_file:
        
        srt_segment_index = 1
        for segment in segments:
            # Вывод прогресса в консоль
            percent_done = (segment.end / total_duration_seconds) * 100
            print(f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {segment.text.strip()} ({percent_done:.2f}%)")
            
            # Запись в TXT
            if "txt" in output_formats:
                txt_file.write(segment.text.strip() + " ")
            
            # Запись в SRT
            if "srt" in output_formats:
                srt_file.write(f"{srt_segment_index}\n")
                srt_file.write(f"{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}\n")
                srt_file.write(segment.text.strip() + "\n\n")
                srt_segment_index += 1
    
    print("Transcription complete.")
    clear_gpu_memory()

if __name__ == "__main__":
    # Код парсера аргументов и выбора модели остается без изменений
    parser = argparse.ArgumentParser(description="Transcribe audio with progress using faster-whisper.")
    parser.add_argument("--audio_file", "-a", required=True, help="Path to the audio file to transcribe.")
    parser.add_argument("--model", "-m",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2", "large-v3"],
                        required=False,
                        help="Whisper model name to use.")
    args = parser.parse_args()
    # ... (весь остальной код в __main__ остается таким же)
    formats_tuple = ("txt", "srt")
    user_selected_model = args.model
    user_selected_device = None

    ALL_MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2", "large-v3"]

    while user_selected_device not in ["cpu", "gpu"]:
        choice = input("Использовать CPU или GPU? (cpu/gpu): ").strip().lower()
        if choice in ["cpu", "gpu"]: user_selected_device = choice
        else: print("Неверный ввод.")
            
    if user_selected_device == "gpu" and not torch.cuda.is_available():
        print("ВНИМАНИЕ: CUDA недоступна! Выбран GPU, но будет использован CPU.")
        user_selected_device = "cpu"

    if not user_selected_model:
        print("\n--- Выбор модели Whisper ---")
        for i, model_name in enumerate(ALL_MODELS): print(f"  {i+1}. {model_name}")
        while True:
            try:
                model_choice_idx = int(input(f"Введите номер модели (1-{len(ALL_MODELS)}): ")) - 1
                if 0 <= model_choice_idx < len(ALL_MODELS):
                    user_selected_model = ALL_MODELS[model_choice_idx]
                    break
                else: print(f"Неверный номер.")
            except ValueError: print("Неверный ввод.")
    
    print(f"\n--- Конфигурация ---")
    print(f"Аудиофайл: {args.audio_file}")
    print(f"Устройство: {user_selected_device}")
    print(f"Модель: {user_selected_model}")
    print("-----------------------\n")
    
    effective_device = "cuda" if user_selected_device == "gpu" else "cpu"

    try:
        transcribe_with_progress(args.audio_file, user_selected_model, formats_tuple, effective_device)
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()