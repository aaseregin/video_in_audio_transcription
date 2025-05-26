import argparse
import os
import math
import torch
import whisper
import numpy as np
from scipy import signal

def format_timestamp(seconds: float) -> str:
    """Format seconds to [HH:MM:SS] or [MM:SS] timestamp (omit hours if 0)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Предварительная обработка аудио для улучшения качества."""
    # Создаем копию массива для безопасной модификации и приводим к float32
    audio = audio.copy().astype(np.float32)
    
    # Нормализация
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Применение фильтра высоких частот для уменьшения шума
    nyquist = sample_rate / 2
    cutoff = 100  # Hz
    b, a = signal.butter(4, cutoff/nyquist, btype='high')
    audio = signal.filtfilt(b, a, audio)
    
    # Убеждаемся, что массив непрерывен в памяти и имеет правильный тип
    return np.ascontiguousarray(audio, dtype=np.float32)

def transcribe_with_progress(audio_path: str, model_name: str, output_formats: tuple):
    # Load Whisper model (on GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)
    
    # Load audio file and get duration
    print(f"Loading audio file {audio_path}...")
    audio = whisper.load_audio(audio_path)  # returns waveform at 16 kHz
    SAMPLE_RATE = 16000
    
    # Предварительная обработка аудио
    print("Preprocessing audio...")
    audio = preprocess_audio(audio, SAMPLE_RATE)
    
    total_samples = len(audio)
    total_duration = total_samples / SAMPLE_RATE

    # Увеличиваем размер чанка для лучшего контекста
    chunk_size = 60 * SAMPLE_RATE  # 60 секунд вместо 30

    # Prepare output files for selected formats
    base_name = os.path.splitext(audio_path)[0]
    txt_file = open(base_name + ".txt", "w", encoding="utf-8") if "txt" in output_formats else None
    srt_file = open(base_name + ".srt", "w", encoding="utf-8") if "srt" in output_formats else None

    # (Optional) Detect language once to reuse for all segments (avoids repeating detection)
    language = None
    try:
        # Use up to the first 30s of audio for language detection
        sample = audio[:min(total_samples, SAMPLE_RATE * 30)]
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(sample))
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        print(f"Detected language: {language}")
    except Exception:
        pass

    print(f"Transcribing audio ({total_duration:.1f} seconds) with Whisper model '{model_name}'...")
    # Process audio in 30-second chunks (Whisper's default window):contentReference[oaicite:3]{index=3}
    segment_count = 0  # counter for segments (for numbering and tracking first segment)
    for offset in range(0, total_samples, chunk_size):
        # Extract a 30s chunk of audio (pad the last chunk if shorter)
        chunk_audio = audio[offset: offset + chunk_size]
        chunk_start_time = offset / SAMPLE_RATE
        if len(chunk_audio) < chunk_size:
            chunk_audio = whisper.pad_or_trim(chunk_audio)  # pad last chunk to 30s

        # Transcribe the chunk. We pass the chunk waveform directly to Whisper.
        result = model.transcribe(chunk_audio, verbose=False, language=language)
        for segment in result["segments"]:
            # Compute actual timestamps for this segment in the full audio
            start = segment["start"] + chunk_start_time
            end   = segment["end"]   + chunk_start_time
            # Calculate percentage completed based on segment end time vs total duration
            percent = min((end / total_duration) * 100, 100.0)
            # Display progress in console with timecodes and percent
            print(f"[{format_timestamp(start)} → {format_timestamp(end)}] {percent:.1f}% done")

            # Write segment text to output files incrementally
            text = segment["text"]
            if txt_file:
                # Write to TXT: append text (ensure proper spacing between segments)
                if segment_count == 0:
                    txt_file.write(text.lstrip())  # remove leading space on first segment if present
                else:
                    txt_file.write(text)
                txt_file.flush()  # flush so data is written immediately
            if srt_file:
                # Write to SRT: each segment as a subtitle entry
                segment_count += 1
                srt_file.write(f"{segment_count}\n")
                srt_file.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
                srt_file.write(text.strip() + "\n\n")  # strip to avoid extra spaces/newlines
                srt_file.flush()
            else:
                # Increment segment_count even if only TXT (to keep spacing logic consistent)
                segment_count += 1

    # Close files
    if txt_file: txt_file.close()
    if srt_file: srt_file.close()
    print("Transcription complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with progress using OpenAI Whisper.")
    parser.add_argument("--audio_file", "-a", required=True, help="Path to the audio file to transcribe.")
    parser.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2","large-v3"],
                        help="Whisper model name to use.")
    parser.add_argument("--output_format", "-f", choices=["txt", "srt", "both"],
                        help="Output format: 'txt', 'srt', or 'both'. If omitted, you will be prompted.")
    args = parser.parse_args()

    # Determine output format (prompt user if not provided as flag)
    if args.output_format is None:
        choice = input("Select output format ('txt', 'srt', or 'both'): ").strip().lower()
        if choice not in {"txt", "srt", "both"}:
            print("Invalid choice. Defaulting to 'txt'.")
            choice = "txt"
        args.output_format = choice
    # Convert format choice to tuple for processing
    formats = ("txt", "srt") if args.output_format == "both" else (args.output_format,)
    # Run transcription with progress
    transcribe_with_progress(args.audio_file, args.model, formats)
