# proto-feedback

Turn a screen recording (with narrated audio) into structured UI feedback: it transcribes what you said, grabs a screenshot at each spoken segment, and uses an LLM to classify each comment into a structured feedback item (type, severity, UI element, issue, recommendation).

## Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) and `ffprobe` on your `PATH` (used to extract audio and frames)
- An [Anthropic API key](https://console.anthropic.com/) — unless using `--lm-studio` or `--no-analyze`

## Setup

```bash
# from the project root
pip install -e .
```

Copy the env template and add your API key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

The CLI reads `ANTHROPIC_API_KEY` from the environment, so either export it in your shell or load `.env` however you normally do (e.g. `direnv`, `python-dotenv`, or your shell profile).

## Usage

```bash
proto-feedback path/to/recording.mp4
```

This will:
1. Read the video duration and extract its audio track.
2. Transcribe the audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), splitting it into timestamped speech segments.
3. Extract a screenshot at the midpoint of each segment.
4. Send each screenshot + transcript to Claude (or a local model) for structured analysis.
5. Write everything to `feedback.json`, with screenshots saved under `screenshots/`.

### Options

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `feedback.json` | Output JSON path |
| `-s, --screenshots-dir` | `screenshots` | Directory to save frame images |
| `-m, --model` | `medium` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--min-duration` | `1.0` | Skip speech segments shorter than N seconds |
| `--no-analyze` | off | Skip LLM analysis; output transcript + screenshots only |
| `--lm-studio` | off | Use a local [LM Studio](https://lmstudio.ai/) model instead of Claude |
| `--lm-studio-url` | `http://localhost:1234/v1` | LM Studio API base URL |
| `--lm-studio-model` | `local-model` | Model name as shown in LM Studio |
| `--lm-studio-vision` | off | Send screenshots to LM Studio (requires a vision-capable model) |

### Examples

Skip analysis, just get transcript + screenshots (no API key needed):

```bash
proto-feedback recording.mp4 --no-analyze
```

Use a local LM Studio model with vision support instead of Claude:

```bash
proto-feedback recording.mp4 --lm-studio --lm-studio-vision --lm-studio-model llava-13b
```

Use a faster/smaller Whisper model for a quick pass:

```bash
proto-feedback recording.mp4 -m small
```

## Output format

`feedback.json` contains:

```json
{
  "source": "path/to/recording.mp4",
  "duration_sec": 65.267,
  "generated_at": "2026-07-07T23:47:48.870426+00:00",
  "items": [
    {
      "id": 1,
      "timestamp_start": 0.0,
      "timestamp_end": 8.28,
      "timestamp_label": "0:00",
      "screenshot": "screenshots/frame_00004.png",
      "transcript": "...what the reviewer said...",
      "feedback": {
        "type": "usability",
        "severity": "medium",
        "ui_element": "submit button",
        "issue": "...",
        "recommendation": "...",
        "sentiment": "negative"
      }
    }
  ]
}
```

`feedback` is `null` when analysis was skipped (`--no-analyze`) or failed for that item.

## Notes

- `feedback.json` and `screenshots/` are generated output and are git-ignored — don't commit them.
- Larger Whisper models (`medium`, `large`) are more accurate but slower on CPU; drop to `small` or `base` for quick iteration.
