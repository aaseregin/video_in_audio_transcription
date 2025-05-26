import argparse
import ffmpeg
import os

def extract_audio(video_path, output_format='mp3'):
    """
    Извлекает аудио из видеофайла
    
    Args:
        video_path (str): Путь к видеофайлу
        output_format (str): Формат выходного аудиофайла (mp3, wav, m4a)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    
    # Создаем имя выходного файла
    output_path = os.path.splitext(video_path)[0] + f'.{output_format}'
    
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
        raise

def main():
    parser = argparse.ArgumentParser(description="Извлечение аудио из видеофайла")
    parser.add_argument("--video_file", required=True, help="Путь к видеофайлу")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "m4a"],
                      help="Формат выходного аудиофайла")
    
    args = parser.parse_args()
    
    try:
        output_path = extract_audio(args.video_file, args.format)
        print("\nТеперь вы можете транскрибировать аудио с помощью команды:")
        print(f"python whisper_progress.py --audio_file {output_path}")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main() 