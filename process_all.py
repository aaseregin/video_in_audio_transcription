# process_all.py

import os
import sys
import torch
import ffmpeg
import math
import librosa
from tqdm import tqdm
from faster_whisper import WhisperModel

# --- 1. КОНФИГУРАЦИЯ И НАСТРОЙКА ---

INPUT_DIR = "input"
OUTPUT_DIR = "output"
SUPPORTED_VIDEO_EXT = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv')
SUPPORTED_AUDIO_EXT = ('.mp3', '.wav', 'm4a', '.flac', '.ogg')
MODEL_SIZE = "large-v3"

# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def setup_directories():
    """Проверяет и создает необходимые папки input и output."""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"INFO: Создана папка '{INPUT_DIR}'. Поместите в нее ваши медиафайлы.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"INFO: Создана папка '{OUTPUT_DIR}' для сохранения результатов.")

def format_timestamp_srt(seconds: float) -> str:
    """Форматирует время в стандартный для SRT формат HH:MM:SS,ms."""
    assert seconds >= 0, "Время не может быть отрицательным"
    millis = round(seconds * 1000.0)
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    seconds = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def extract_audio(video_path: str, output_path: str, format: str) -> bool:
    """Извлекает аудиодорожку из видеофайла с помощью FFmpeg."""
    print(f"-> Извлечение аудио из '{os.path.basename(video_path)}'...")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ac=1, ar=16000, vn=None) # ac=1 (mono), ar=16kHz, vn=no video
            .run(quiet=True, overwrite_output=True)
        )
        print(f"   Аудио успешно сохранено в формате .{format}")
        return True
    except ffmpeg.Error as e:
        print(f"   ERROR: Не удалось извлечь аудио. Ошибка FFmpeg: {e.stderr.decode()}")
        return False

# --- 3. ОСНОВНЫЕ ФУНКЦИИ ПРОЦЕССА ---

def run_transcription(model, audio_path: str, task: str, language: str or None):
    """
    Выполняет транскрибацию или перевод, отображая посегментный прогресс.
    Возвращает полный текст и содержимое SRT файла.
    """
    try:
        total_duration = librosa.get_duration(path=audio_path)
    except Exception as e:
        print(f"   ERROR: Не удалось получить длительность аудиофайла: {e}")
        return None, None
        
    segments, info = model.transcribe(audio_path, beam_size=1, task=task, language=language)

    print(f"-> Распознан язык: '{info.language}' (вероятность {info.language_probability:.2f})")
    print(f"-> Начало транскрибации (задача: {task})...")

    full_text = ""
    srt_content = ""
    
    for i, segment in enumerate(segments, 1):
        percent_done = (segment.end / total_duration) * 100 if total_duration > 0 else 0
        progress_line = f"   [{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}] {segment.text.strip()} ({percent_done:.1f}%)"
        print(progress_line)
        
        full_text += segment.text.strip() + " "
        srt_content += f"{i}\n{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}\n{segment.text.strip()}\n\n"
        
    return full_text.strip(), srt_content

# --- 4. ГЛАВНЫЙ СКРИПТ ---

def main():
    """Основная функция, управляющая всем процессом."""
    setup_directories()

    # Поиск файлов в папке input
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED_VIDEO_EXT + SUPPORTED_AUDIO_EXT)]
    
    if not files_to_process:
        print("\nПапка 'input' пуста. Нечего обрабатывать.")
        sys.exit()

    print(f"\nНайдено {len(files_to_process)} файлов для обработки:")
    for filename in files_to_process:
        print(f"- {filename}")
    
    # Автоматический выбор устройства и типа вычислений
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"\nИспользуется устройство: {device.upper()} (тип вычислений: {compute_type})")
    print(f"Загрузка модели Whisper '{MODEL_SIZE}'... (Это может занять некоторое время)")
    try:
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"FATAL: Не удалось загрузить модель. Ошибка: {e}")
        sys.exit()

    # Основной цикл обработки файлов с общим прогресс-баром
    with tqdm(total=len(files_to_process), desc="Общий прогресс", unit="файл") as pbar:
        for filename in files_to_process:
            pbar.set_description(f"Обработка: {filename}")
            
            input_path = os.path.join(INPUT_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            output_subdir = os.path.join(OUTPUT_DIR, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            audio_for_transcription = None
            is_temp_wav = False
            
            try:
                # --- Шаг 1: Определение типа файла и действия пользователя ---
                if filename.lower().endswith(SUPPORTED_VIDEO_EXT):
                    print(f"\n--- Файл '{filename}' определен как ВИДЕО ---")
                    choice = input("   Выберите действие:\n   1. Извлечь аудиодорожку (в .mp3)\n   2. Транскрибировать видео в текст\n   Ваш выбор (1/2): ").strip()
                    
                    if choice == '1':
                        output_mp3_path = os.path.join(output_subdir, "audio.mp3")
                        extract_audio(input_path, output_mp3_path, "mp3")
                        pbar.update(1)
                        continue # Переходим к следующему файлу
                    
                    elif choice == '2':
                        temp_wav_path = os.path.join(output_subdir, f"{base_name}_temp.wav")
                        if extract_audio(input_path, temp_wav_path, "wav"):
                            audio_for_transcription = temp_wav_path
                            is_temp_wav = True
                        else:
                            pbar.update(1)
                            continue # Ошибка извлечения, пропускаем
                    else:
                        print("   Неверный выбор. Файл пропущен.")
                        pbar.update(1)
                        continue

                elif filename.lower().endswith(SUPPORTED_AUDIO_EXT):
                    print(f"\n--- Файл '{filename}' определен как АУДИО ---")
                    audio_for_transcription = input_path
                
                # --- Шаг 2: Транскрибация (если требуется) ---
                if audio_for_transcription:
                    # Определение языка
                    _, info = model.transcribe(audio_for_transcription, beam_size=1)
                    language = info.language
                    
                    task = "transcribe"
                    # Интерактивный перевод для английского
                    if language == 'en':
                        translate_choice = input(f"-> Обнаружен английский язык. Перевести на русский? (y/n): ").strip().lower()
                        if translate_choice == 'y':
                            task = "translate"
                    
                    # Запуск основной функции
                    full_text, srt_content = run_transcription(model, audio_for_transcription, task, language if task == "transcribe" else "en")
                    
                    # --- Шаг 3: Сохранение результатов ---
                    if full_text and srt_content:
                        suffix = "_ru" if task == "translate" else ""
                        txt_filename = f"transcription{suffix}.txt"
                        srt_filename = f"subtitles{suffix}.srt"
                        
                        with open(os.path.join(output_subdir, txt_filename), 'w', encoding='utf-8') as f:
                            f.write(full_text)
                        with open(os.path.join(output_subdir, srt_filename), 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                        
                        print(f"-> Результаты сохранены в папке: '{output_subdir}'")
            
            except KeyboardInterrupt:
                print("\nПроцесс прерван пользователем.")
                sys.exit()
            except Exception as e:
                print(f"\n   КРИТИЧЕСКАЯ ОШИБКА при обработке файла '{filename}': {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # --- Шаг 4: Очистка временных файлов ---
                if is_temp_wav and os.path.exists(audio_for_transcription):
                    os.remove(audio_for_transcription)
                    print(f"-> Временный файл '{os.path.basename(audio_for_transcription)}' удален.")
                pbar.update(1)

    print("\n\nВсе файлы успешно обработаны!")

if __name__ == "__main__":
    main()