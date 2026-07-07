from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe(
    audio_path: Path,
    model_size: str = "medium",
    min_duration: float = 1.0,
) -> list[Segment]:
    """
    Transcribe audio with faster-whisper, returning time-aligned segments.
    Segments shorter than min_duration seconds are dropped (silence artifacts).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(audio_path), beam_size=5)

    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        duration = seg.end - seg.start
        if not text or duration < min_duration:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text))

    return segments
