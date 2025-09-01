# скрипт абсолютно сырой
# 
import os
import whisper_progress as wp
import torch
import ffmpeg
import whisper


def check_gpu_memory_for_large_v3():
    if not torch.cuda.is_available():
        return False
    required_memory = 10 * 1024 * 1024 * 1024  # 10GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    return free_memory >= required_memory


def extract_audio_from_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    output_path = os.path.splitext(video_path)[0] + '.wav'
    print(f"Извлечение аудио из {video_path}...")
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Аудио успешно извлечено и сохранено в: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print(f"Ошибка при извлечении аудио: {e.stderr.decode()}")
        return None


def detect_language(audio_path, model_name="base"):
    print("Определение языка аудио...")
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path, task="transcribe", language=None, fp16=False)
    lang = result.get("language", "unknown")
    print(f"Обнаружен язык: {lang}")
    return lang


def process_video_folder(folder_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp')
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
    if not video_files:
        print(f"В папке {folder_path} не найдено видео файлов.")
        return
    print(f"Найдено {len(video_files)} видео файлов для обработки.")
    device = "cuda" if check_gpu_memory_for_large_v3() else "cpu"
    print(f"Используется устройство: {device}")
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"\nОбработка видео: {video_file}")
        print("Извлечение аудио...")
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            print(f"Пропуск файла {video_file} из-за ошибки извлечения аудио.")
            continue
        try:
            lang = detect_language(audio_path)
            if lang == "en":
                print("Видео на английском языке.")
                choice = input("Транскрибировать на английском (1) или сразу перевести на русский (2)? Введите 1 или 2: ")
                if choice.strip() == "2":
                    print("Переводим на русский...")
                    # Прямой вызов whisper для перевода
                    model = whisper.load_model("large-v3", device=device)
                    result = model.transcribe(audio_path, task="translate", language="en")
                    # Сохраняем результат в txt и srt
                    base_name = os.path.splitext(audio_path)[0]
                    with open(base_name + "_ru.txt", "w", encoding="utf-8") as f:
                        f.write(result["text"])
                    with open(base_name + "_ru.srt", "w", encoding="utf-8") as f:
                        for i, seg in enumerate(result["segments"], 1):
                            f.write(f"{i}\n{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n{seg['text'].strip()}\n\n")
                else:
                    print("Транскрибируем на английском...")
                    wp.transcribe_with_progress(
                        audio_path,
                        "large-v3",
                        ("txt", "srt"),
                        device
                    )
            else:
                print(f"Видео на языке: {lang}. Обычная транскрипция...")
                wp.transcribe_with_progress(
                    audio_path,
                    "large-v3",
                    ("txt", "srt"),
                    device,
                    task="transcribe",
                    language=lang
                )
            print(f"Обработка {video_file} завершена успешно.")
        except Exception as e:
            print(f"Ошибка при обработке {video_file}: {e}")
            continue

if __name__ == "__main__":
    video_folder = "video"
    if not os.path.exists(video_folder):
        print(f"Папка {video_folder} не существует.")
        exit(1)
    process_video_folder(video_folder) 