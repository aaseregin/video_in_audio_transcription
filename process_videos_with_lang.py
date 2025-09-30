# process_videos_with_lang.py

import os
import torch
import ffmpeg
import math
from faster_whisper import WhisperModel
import whisper_progress as wp

def format_timestamp_srt(seconds: float) -> str:
    # ... (код без изменений)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def extract_audio_from_video(video_path):
    # ... (код без изменений, но убедитесь, что он есть)
    output_path = os.path.splitext(video_path)[0] + '.wav'
    try:
        (ffmpeg.input(video_path).output(output_path, ar=16000, ac=1).run(quiet=True, overwrite_output=True))
        return output_path
    except ffmpeg.Error as e:
        print(f"Ошибка извлечения аудио: {e.stderr.decode()}")
        return None

def detect_language(audio_path, model):
    """Определяет язык по первым 30 секундам аудио."""
    print("Определение языка...")
    _, info = model.transcribe(audio_path, initial_prompt="Detect language.")
    lang = info.language
    print(f"Обнаружен язык: {lang} (вероятность {info.language_probability:.2f})")
    return lang

def process_video_folder(folder_path):
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"В папке {folder_path} не найдено видео.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Используется устройство: {device}")

    print("Загрузка моделей...")
    # Используем одну модель для всех задач, это эффективнее
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    print("Модель загружена.")

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"\n--- Обработка: {video_file} ---")
        
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            continue
            
        try:
            lang = detect_language(audio_path, model)
            
            if lang == "en":
                choice = input("Видео на английском. Транскрибировать (1) или перевести на русский (2)? [1]: ")
                task = "translate" if choice.strip() == "2" else "transcribe"
                
                print(f"Выполняется задача: {task}...")
                segments, _ = model.transcribe(audio_path, task=task, language="en")

                base_name = os.path.splitext(audio_path)[0]
                suffix = "_ru" if task == "translate" else ""
                
                with open(base_name + suffix + ".txt", "w", encoding="utf-8") as txt_f, \
                     open(base_name + suffix + ".srt", "w", encoding="utf-8") as srt_f:
                    for i, seg in enumerate(segments, 1):
                        txt_f.write(seg.text.strip() + " ")
                        srt_f.write(f"{i}\n{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n{seg.text.strip()}\n\n")

            else:
                print(f"Видео на языке '{lang}'. Обычная транскрипция...")
                # Вызываем основной, уже исправленный скрипт
                wp.transcribe_with_progress(audio_path, "large-v3", ("txt", "srt"), device, language=lang)
                
            print(f"Обработка {video_file} завершена.")
            os.remove(audio_path)
            
        except Exception as e:
            print(f"Ошибка при обработке {video_file}: {e}")
            continue

if __name__ == "__main__":
    video_folder = "video"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    process_video_folder(video_folder)