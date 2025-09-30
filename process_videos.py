# process_videos.py

import os
import whisper_progress as wp
import torch
import ffmpeg

def check_gpu_memory_for_model(model_name="large-v3"):
    """Проверяет, достаточно ли видеопамяти для выбранной модели."""
    if not torch.cuda.is_available():
        return False
    
    # Требования к памяти для оптимизированных моделей (примерные)
    # large-v3: ~8GB, medium: ~5GB, small: ~2GB
    model_memory_req = {
        "large-v3": 8, "large-v2": 8, "large": 8,
        "medium": 5, "medium.en": 5,
        "small": 2, "small.en": 2,
    }
    
    # Для маленьких моделей всегда считаем, что памяти хватит
    required_gb = model_memory_req.get(model_name, 1) 
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Для модели '{model_name}' требуется ~{required_gb} GB. В наличии: {gpu_memory_gb:.2f} GB.")
    return gpu_memory_gb >= required_gb

def extract_audio_from_video(video_path):
    """Извлекает аудио из видео файла в формате WAV 16kHz mono."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    
    output_path = os.path.splitext(video_path)[0] + '.wav'
    
    print(f"Извлечение аудио из {video_path}...")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ar=16000, ac=1, y=None) # ar=16kHz, ac=mono, y=overwrite
            .run(quiet=True, overwrite_output=True)
        )
        print(f"Аудио успешно извлечено и сохранено в: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print(f"Ошибка при извлечении аудио: {e.stderr.decode()}")
        return None

def process_video_folder(folder_path):
    """Обрабатывает все видео файлы в указанной папке."""
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp')
    
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"В папке '{folder_path}' не найдено видео файлов.")
        return
    
    print(f"Найдено {len(video_files)} видео файлов для обработки.")
    
    model_to_use = "large-v3"
    device = "cuda" if check_gpu_memory_for_model(model_to_use) else "cpu"
    print(f"Будет использоваться модель: '{model_to_use}' на устройстве: '{device}'")
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"\n--- Обработка видео: {video_file} ---")
        
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            print(f"Пропуск файла {video_file} из-за ошибки извлечения аудио.")
            continue
        
        try:
            print("Транскрибация аудио...")
            # Вызываем исправленную функцию
            wp.transcribe_with_progress(
                audio_path,
                model_to_use,
                ("txt", "srt"),
                device
            )
            print(f"Обработка {video_file} завершена успешно.")
            os.remove(audio_path) # Удаляем временный wav файл

        except Exception as e:
            print(f"Критическая ошибка при обработке {video_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    video_folder = "video"
    
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        print(f"Папка '{video_folder}' была создана. Пожалуйста, поместите в нее видеофайлы для обработки.")
        exit(0)
    
    process_video_folder(video_folder)