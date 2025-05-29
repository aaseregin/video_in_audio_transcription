import os
import subprocess
from pathlib import Path
import whisper_progress as wp
import torch
import ffmpeg

def check_gpu_memory_for_large_v3():
    """Проверяет, достаточно ли видеопамяти для модели large-v3."""
    if not torch.cuda.is_available():
        return False
    
    # Примерный объем памяти, необходимый для large-v3 (около 10GB)
    required_memory = 10 * 1024 * 1024 * 1024  # 10GB в байтах
    
    # Получаем информацию о доступной памяти GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    
    # Проверяем, достаточно ли свободной памяти
    return free_memory >= required_memory

def extract_audio_from_video(video_path):
    """Извлекает аудио из видео файла."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    
    # Создаем имя выходного файла
    output_path = os.path.splitext(video_path)[0] + '.wav'
    
    print(f"Извлечение аудио из {video_path}...")
    try:
        # Извлекаем аудио с помощью ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Аудио успешно извлечено и сохранено в: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print(f"Ошибка при извлечении аудио: {e.stderr.decode()}")
        return None

def process_video_folder(folder_path):
    """Обрабатывает все видео файлы в указанной папке."""
    # Поддерживаемые форматы видео
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp')
    
    # Получаем список всех видео файлов
    video_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"В папке {folder_path} не найдено видео файлов.")
        return
    
    print(f"Найдено {len(video_files)} видео файлов для обработки.")
    
    # Определяем устройство для обработки
    device = "cuda" if check_gpu_memory_for_large_v3() else "cpu"
    print(f"Используется устройство: {device}")
    
    # Обрабатываем каждый видео файл
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"\nОбработка видео: {video_file}")
        
        # Извлекаем аудио
        print("Извлечение аудио...")
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            print(f"Пропуск файла {video_file} из-за ошибки извлечения аудио.")
            continue
        
        try:
            # Транскрибируем аудио
            print("Транскрибация аудио...")
            wp.transcribe_with_progress(
                audio_path,
                "large-v3",  # Используем модель large-v3
                ("txt", "srt"),
                device
            )
            
            print(f"Обработка {video_file} завершена успешно.")
            
        except Exception as e:
            print(f"Ошибка при обработке {video_file}: {e}")
            continue

if __name__ == "__main__":
    # Путь к папке с видео
    video_folder = "video"
    
    # Проверяем существование папки
    if not os.path.exists(video_folder):
        print(f"Папка {video_folder} не существует.")
        exit(1)
    
    # Запускаем обработку
    process_video_folder(video_folder) 