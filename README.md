# Whisper - Распознавание речи

Этот проект использует OpenAI Whisper для распознавания речи из аудиофайлов.

## Установка

1. Убедитесь, что у вас установлен Python 3.8 или выше. Скачайте и установите Python с [официального сайта](https://www.python.org/downloads/), обязательно отметив галочку "Add Python to PATH".

2. Установите FFmpeg:
   - Windows: Скачайте архив с [официального сайта](https://ffmpeg.org/download.html), распакуйте его (например, в `C:\ffmpeg`) и добавьте путь к папке `bin` (например, `C:\ffmpeg\bin`) в переменную среды PATH.
   - Linux: `sudo apt update && sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

3. Клонируйте или скачайте этот проект в удобную папку.

4. Обновите pip до последней версии:
   ```bash
   python -m pip install --upgrade pip
   ```

5. Установите зависимости проекта:
   ```bash
   pip install -r requirements.txt
   ```
   Если возникнут ошибки с Whisper, выполните:
   ```bash
   pip install git+https://github.com/openai/whisper.git
   ```

6. Обновите зависимости до последних версий:
   ```bash
   pip install --upgrade torch numpy ffmpeg-python tqdm
   pip install --upgrade git+https://github.com/openai/whisper.git
   ```

## Использование

### Извлечение аудио из видео

1. Поместите ваш видеофайл (например, `video.mp4`) в папку проекта.
2. Запустите скрипт извлечения аудио:
   ```bash
   python extract_audio.py --video_file video.mp4
   ```
   По умолчанию будет создан файл `video.mp3`. Если нужно указать другой формат, используйте параметр `--format`:
   ```bash
   python extract_audio.py --video_file video.mp4 --format wav
   ```

### Транскрибация аудио

1. После извлечения аудио запустите транскрибацию с помощью модели Whisper. Для самой качественной транскрибации используйте модель `large`:
   ```bash
   python transcribe.py --audio_file video.mp3 --model large
   ```
   Результат транскрибации будет выведен в терминале.

2. Если у вас нет видеокарты (GPU), транскрибация будет выполняться на CPU. Это может занять больше времени, но результат будет таким же качественным.

## Запуск в Cursor IDE и Visual Studio Code

1. Откройте папку проекта в Cursor IDE или Visual Studio Code.
2. Откройте терминал (Terminal → New Terminal).
3. Выполните команды из инструкции выше.
4. Для запуска скриптов можно использовать встроенный терминал или кнопки "Run" (если IDE их поддерживает).

## Поддерживаемые форматы

- MP3
- WAV
- M4A
- FLAC
- и другие форматы, поддерживаемые FFmpeg 