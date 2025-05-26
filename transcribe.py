import whisper
import torch
from optimize_memory import optimize_gpu_memory, clear_gpu_memory, get_gpu_memory_info, get_nvidia_gpu
import os
import time

def verify_gpu_setup():
    """Проверяет и настраивает GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступен")
    
    # Получаем информацию о GPU
    nvidia_gpu_id = get_nvidia_gpu()
    device_name = torch.cuda.get_device_name(nvidia_gpu_id)
    print(f"\nИспользуем GPU: {device_name}")
    print(f"Общая память: {torch.cuda.get_device_properties(nvidia_gpu_id).total_memory / 1024**3:.2f} GB")
    
    # Проверяем, что GPU действительно NVIDIA
    if "nvidia" not in device_name.lower():
        raise RuntimeError(f"Найдена не NVIDIA GPU: {device_name}")
    
    # Устанавливаем активное устройство
    torch.cuda.set_device(nvidia_gpu_id)
    
    # Проверяем, что устройство установлено правильно
    if torch.cuda.current_device() != nvidia_gpu_id:
        raise RuntimeError("Не удалось установить правильное GPU устройство")
    
    return f"cuda:{nvidia_gpu_id}"

def load_model_in_steps(device):
    """Загружает модель пошагово для экономии памяти"""
    print("\nНачинаем пошаговую загрузку модели...")
    
    # Шаг 1: Загружаем базовую модель
    print("Шаг 1: Загрузка базовой модели...")
    model = whisper.load_model("large", device="cpu")
    clear_gpu_memory()
    
    # Шаг 2: Перемещаем модель на GPU по частям
    print("Шаг 2: Перемещение модели на GPU...")
    model = model.to(device)
    clear_gpu_memory()
    
    # Шаг 3: Включаем оптимизации
    print("Шаг 3: Включение оптимизаций...")
    model.eval()  # Переключаем в режим оценки
    with torch.no_grad():  # Отключаем градиенты
        torch.cuda.empty_cache()
    
    print("Модель успешно загружена")
    return model

def transcribe_audio(audio_path):
    try:
        # Проверяем и настраиваем GPU
        device = verify_gpu_setup()
        print(f"Используем устройство: {device}")
        
        # Оптимизируем память перед загрузкой модели
        optimize_gpu_memory()
        
        # Загружаем модель пошагово
        model = load_model_in_steps(device)
        
        # Выводим информацию о памяти
        get_gpu_memory_info()
        
        # Транскрибируем аудио с оптимизированными параметрами
        print("\nНачинаем транскрипцию...")
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                result = model.transcribe(
                    audio_path,
                    batch_size=1,  # Минимальный размер батча
                    fp16=True,     # Используем половинную точность
                    language="ru",  # Указываем язык для оптимизации
                    beam_size=1     # Уменьшаем размер луча для экономии памяти
                )
        
        # Очищаем память после транскрипции
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        print(f"\nОшибка при транскрипции: {str(e)}")
        print("\nТекущее состояние GPU:")
        get_gpu_memory_info()
        raise

if __name__ == "__main__":
    audio_path = "input.mp3"  # Замените на путь к вашему аудиофайлу
    result = transcribe_audio(audio_path)
    print(result["text"]) 