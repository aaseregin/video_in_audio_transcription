import torch
import gc
import os

def get_nvidia_gpu():
    """Находит и возвращает индекс NVIDIA GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступен")
    
    # Получаем информацию о всех GPU
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "nvidia" in device_name.lower():
            print(f"Найдена NVIDIA GPU: {device_name}")
            return i
    
    raise RuntimeError("NVIDIA GPU не найдена")

def optimize_gpu_memory():
    """Оптимизирует использование памяти GPU"""
    # Очищаем кэш CUDA
    torch.cuda.empty_cache()
    
    # Принудительно запускаем сборщик мусора
    gc.collect()
    
    # Устанавливаем более агрессивные настройки памяти
    torch.cuda.set_per_process_memory_fraction(0.6)  # Уменьшаем до 60%
    
    # Настраиваем параметры выделения памяти
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True,garbage_collection_threshold:0.6'
    
    # Отключаем кэширование для экономии памяти
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Синхронизируем CUDA
    torch.cuda.synchronize()

def clear_gpu_memory():
    """Очищает память GPU"""
    # Очищаем кэш CUDA
    torch.cuda.empty_cache()
    
    # Принудительно запускаем сборщик мусора
    gc.collect()
    
    # Синхронизируем CUDA
    torch.cuda.synchronize()
    
    # Проверяем, что память действительно очищена
    if torch.cuda.memory_allocated() > 0:
        print(f"Предупреждение: {torch.cuda.memory_allocated() / 1024**3:.2f} GB памяти все еще выделено")

def get_gpu_memory_info():
    """Выводит информацию о памяти GPU"""
    print("\n=== Информация о памяти GPU ===")
    print(f"Выделено памяти: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Зарезервировано памяти: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Максимально выделено памяти: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Максимально зарезервировано памяти: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB") 