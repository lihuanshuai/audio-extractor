import argparse
import sys
from pathlib import Path

from audio_extractor.config import ExtractorConfig, parse_configured_args
from audio_extractor.core import FORMATTERS, load_model, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from audio files using faster-whisper"
    )
    parser.add_argument("audio", type=str, help="Path to the audio file")
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="Whisper model size (tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code (e.g. zh, en). Auto-detect if omitted.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=list(FORMATTERS.keys()),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "-c",
        "--compute-type",
        default="auto",
        help="Compute type: auto, float16, float32, int8_float16, int8",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )

    args, _ = parse_configured_args(parser, "extractor", ExtractorConfig)

    if not Path(args.audio).is_file():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    model = load_model(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    segments, info = transcribe(
        model,
        audio_path=args.audio,
        language=args.language,
        beam_size=args.beam_size,
    )
    print(
        f"Detected language: {info.language} (p={info.language_probability:.2f})",
        file=sys.stderr,
    )

    writer = FORMATTERS[args.format]
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            writer(segments, f)
        print(f"Output written to {out_path}", file=sys.stderr)
    else:
        writer(segments, sys.stdout)


if __name__ == "__main__":
    main()
