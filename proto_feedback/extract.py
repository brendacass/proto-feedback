import subprocess
from pathlib import Path


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract mono 16kHz WAV from video for Whisper."""
    audio_path = out_dir / "audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def extract_frame(video_path: Path, timestamp_sec: float, out_path: Path) -> Path:
    """Extract a single frame from video at the given timestamp (seconds)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def get_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())
