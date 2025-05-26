import torch
import subprocess
import platform
import os

def get_nvidia_smi_info():
    """Получает информацию о GPU через nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Ошибка при выполнении nvidia-smi: {str(e)}"

def get_detailed_gpu_info():
    """Получает подробную информацию о GPU через PyTorch"""
    print("\n=== Информация о GPU через PyTorch ===")
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    print(f"Версия CUDA: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"Имя: {torch.cuda.get_device_name(i)}")
        print(f"Общая память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"Выделено памяти: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Зарезервировано памяти: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"Свободно памяти: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
        
        # Проверяем, является ли GPU активным
        if i == torch.cuda.current_device():
            print("Статус: Активен")
        else:
            print("Статус: Неактивен")

def check_cuda_environment():
    """Проверяет переменные окружения и настройки CUDA"""
    print("\n=== Переменные окружения CUDA ===")
    cuda_vars = [var for var in os.environ if 'CUDA' in var]
    for var in cuda_vars:
        print(f"{var}: {os.environ[var]}")
    
    print("\n=== Настройки PyTorch ===")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"cuDNN включен: {torch.backends.cudnn.enabled}")
    print(f"cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")

def main():
    print("=== Информация о системе ===")
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    print("\n=== Информация от nvidia-smi ===")
    print(get_nvidia_smi_info())
    
    get_detailed_gpu_info()
    check_cuda_environment()

if __name__ == "__main__":
    main() 