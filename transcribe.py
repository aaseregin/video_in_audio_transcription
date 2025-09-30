# transcribe.py

import torch
import time
from faster_whisper import WhisperModel
from optimize_memory import optimize_gpu_memory, clear_gpu_memory, get_gpu_memory_info

def transcribe_audio(audio_path):
    device = "cpu"
    compute_type = "int8"

    if torch.cuda.is_available():
        print("CUDA доступен, используем GPU.")
        device = "cuda"
        compute_type = "float16"
    else:
        print("CUDA недоступен, используем CPU.")
    
    try:
        if device == "cuda":
            optimize_gpu_memory()

        print("Загрузка модели large-v3...")
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        
        if device == "cuda":
            get_gpu_memory_info()
        
        print("\nНачинаем транскрипцию...")
        start_time = time.time()
        
        # model.transcribe возвращает генератор
        segments, info = model.transcribe(audio_path, language="ru")

        # Собираем полный текст из генератора
        full_text = "".join(segment.text for segment in segments)
        
        end_time = time.time()
        print(f"Транскрипция завершена за {end_time - start_time:.2f} секунд.")

        if device == "cuda":
            clear_gpu_memory()
        
        return full_text
        
    except Exception as e:
        print(f"\nОшибка при транскрипции: {e}")
        if device == "cuda":
            print("\nТекущее состояние GPU:")
            get_gpu_memory_info()
        raise

if __name__ == "__main__":
    audio_path = "input.mp3"
    result_text = transcribe_audio(audio_path)
    print("\n--- Результат транскрибации ---")
    print(result_text)