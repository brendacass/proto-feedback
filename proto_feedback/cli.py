import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .extract import extract_audio, extract_frame, get_duration
from .transcribe import transcribe

console = Console(legacy_windows=False)


def _timestamp_label(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


@click.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", default="feedback.json", show_default=True, help="Output JSON path.")
@click.option("--screenshots-dir", "-s", default="screenshots", show_default=True, help="Directory to save frame images.")
@click.option(
    "--model",
    "-m",
    default="medium",
    show_default=True,
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    help="Whisper model size.",
)
@click.option("--min-duration", default=1.0, show_default=True, help="Skip speech segments shorter than N seconds.")
@click.option("--no-analyze", is_flag=True, default=False, help="Skip analysis entirely; output transcript + screenshots only.")
@click.option("--lm-studio", "use_lm_studio", is_flag=True, default=False, help="Use a local LM Studio model instead of Claude.")
@click.option("--lm-studio-url", default="http://localhost:1234/v1", show_default=True, help="LM Studio API base URL.")
@click.option("--lm-studio-model", default="local-model", show_default=True, help="Model name as shown in LM Studio.")
@click.option("--lm-studio-vision", is_flag=True, default=False, help="Send screenshots to LM Studio (requires a vision-capable model).")
def main(
    video_path: Path,
    output: str,
    screenshots_dir: str,
    model: str,
    min_duration: float,
    no_analyze: bool,
    use_lm_studio: bool,
    lm_studio_url: str,
    lm_studio_model: str,
    lm_studio_vision: bool,
) -> None:
    """Convert a screen recording with audio into structured UI prototype feedback."""

    output_path = Path(output)
    screenshots_path = Path(screenshots_dir)
    screenshots_path.mkdir(parents=True, exist_ok=True)

    analyze_fn = None
    anthropic_client = None

    if not no_analyze:
        if use_lm_studio:
            from .analyze import analyze_item_lm_studio
            analyze_fn = lambda path, text: analyze_item_lm_studio(
                path, text, lm_studio_url, lm_studio_model, use_vision=lm_studio_vision
            )
            console.print(
                f"[dim]Using LM Studio at [bold]{lm_studio_url}[/bold] "
                f"(model: {lm_studio_model}"
                + (", vision" if lm_studio_vision else "")
                + ")[/dim]"
            )
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise click.ClickException(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Use --lm-studio for local analysis or --no-analyze to skip."
                )
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            from .analyze import analyze_item
            analyze_fn = lambda path, text: analyze_item(path, text, anthropic_client)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Step 1: Get video duration
        task = progress.add_task("Reading video metadata...", total=None)
        duration = get_duration(video_path)
        progress.update(task, description=f"Video duration: {_timestamp_label(duration)}", completed=True)

        # Step 2: Extract audio
        task = progress.add_task("Extracting audio...", total=None)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = extract_audio(video_path, tmp_path)
            progress.update(task, description="Audio extracted", completed=True)

            # Step 3: Transcribe
            task = progress.add_task(f"Transcribing with Whisper [{model}]...", total=None)
            segments = transcribe(audio_path, model_size=model, min_duration=min_duration)
            progress.update(task, description=f"Found {len(segments)} speech segments", completed=True)

        if not segments:
            console.print("[yellow]No speech segments found. Check the recording has audible narration.[/yellow]")
            return

        # Step 4: Extract frames + analyze
        items = []
        task = progress.add_task("Processing segments...", total=len(segments))

        for i, seg in enumerate(segments, start=1):
            midpoint = seg.midpoint
            frame_filename = f"frame_{int(midpoint):05d}.png"
            frame_path = screenshots_path / frame_filename

            extract_frame(video_path, midpoint, frame_path)

            feedback = analyze_fn(frame_path, seg.text) if analyze_fn else None

            items.append(
                {
                    "id": i,
                    "timestamp_start": round(seg.start, 3),
                    "timestamp_end": round(seg.end, 3),
                    "timestamp_label": _timestamp_label(seg.start),
                    "screenshot": str(frame_path),
                    "transcript": seg.text,
                    "feedback": feedback,
                }
            )
            progress.update(
                task,
                advance=1,
                description=f"[{_timestamp_label(seg.start)}] {seg.text[:60]}{'…' if len(seg.text) > 60 else ''}",
            )

    # Step 5: Write output
    result = {
        "source": str(video_path),
        "duration_sec": round(duration, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    failed = sum(1 for item in items if item["feedback"] is None and analyze_fn is not None)
    console.print(f"\n[green]Done.[/green] {len(items)} feedback items written to [bold]{output_path}[/bold]")
    if failed:
        console.print(f"[yellow]{failed} item(s) could not be analyzed (feedback: null).[/yellow]")
