import argparse
import json
import sys
from pathlib import Path
from typing import Callable


def _format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_text(segments, file) -> None:
    for seg in segments:
        file.write(seg.text.strip() + "\n")


def _write_srt(segments, file) -> None:
    for i, seg in enumerate(segments, 1):
        file.write(f"{i}\n")
        file.write(
            f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}\n"
        )
        file.write(seg.text.strip() + "\n\n")


def _write_vtt(segments, file) -> None:
    file.write("WEBVTT\n\n")
    for seg in segments:
        start = _format_timestamp(seg.start, always_include_hours=True)
        end = _format_timestamp(seg.end, always_include_hours=True)
        file.write(f"{start} --> {end}\n")
        file.write(seg.text.strip() + "\n\n")


def _write_json(segments, file) -> None:
    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
    json.dump(results, file, ensure_ascii=False, indent=2)
    file.write("\n")


FORMATTERS: dict[str, Callable] = {
    "text": _write_text,
    "srt": _write_srt,
    "vtt": _write_vtt,
    "json": _write_json,
}


def transcribe(
    audio_path: str,
    model_size: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "auto",
    beam_size: int = 5,
):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, language=language, beam_size=beam_size)

    print(f"Detected language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from audio files using faster-whisper"
    )
    parser.add_argument("audio", type=str, help="Path to the audio file")
    parser.add_argument("-m", "--model", default="small", help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("-l", "--language", default=None, help="Language code (e.g. zh, en). Auto-detect if omitted.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("-f", "--format", choices=list(FORMATTERS.keys()), default="text", help="Output format (default: text)")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Device to run on (default: cpu)")
    parser.add_argument("-c", "--compute-type", default="auto", help="Compute type: auto, float16, float32, int8_float16, int8")
    parser.add_argument("-b", "--beam-size", type=int, default=5, help="Beam size for decoding (default: 5)")

    args = parser.parse_args()

    if not Path(args.audio).is_file():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    segments = transcribe(
        audio_path=args.audio,
        model_size=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )

    writer = FORMATTERS[args.format]
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            writer(segments, f)
        print(f"Output written to {out_path}", file=sys.stderr)
    else:
        writer(segments, sys.stdout)


if __name__ == "__main__":
    main()