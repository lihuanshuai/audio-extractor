import argparse
from pathlib import Path

from audio_extractor.config import BatchConfig, parse_configured_args
from audio_extractor.core import AUDIO_EXTS, FORMATTERS, load_model, transcribe


def collect_audio_files(root: str) -> list[Path]:
    files = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe audio files to SRT")
    parser.add_argument("root", type=str, help="Root directory to scan")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--language", default="zh", help="Language code")
    parser.add_argument("--device", default="cuda", help="Device (cpu/cuda)")
    parser.add_argument("--compute-type", default="auto", help="Compute type")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max files to process (0 = all)"
    )
    args, _ = parse_configured_args(parser, "batch", BatchConfig)

    files = collect_audio_files(args.root)

    def is_pending(f: Path) -> bool:
        srt = f.with_suffix(".srt")
        return not srt.is_file() or srt.stat().st_size == 0

    pending = [f for f in files if is_pending(f)]
    if args.limit:
        pending = pending[: args.limit]
    total = len(pending)
    if total == 0:
        print("Nothing to do.")
        return

    print(f"Total audio: {len(files)}, pending: {total}", flush=True)
    model = load_model(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    writer = FORMATTERS["srt"]

    errors = []
    for i, audio in enumerate(pending, 1):
        out_path = audio.with_suffix(".srt")
        try:
            segments, _ = transcribe(
                model,
                audio_path=str(audio),
                language=args.language,
                beam_size=args.beam_size,
            )
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                writer(segments, f)
            print(f"[{i}/{total}] OK  {audio.name}", flush=True)
        except Exception as exc:
            errors.append((str(audio), str(exc)))
            print(f"[{i}/{total}] ERR {audio.name}: {exc}", flush=True)

    print(f"Done. {total - len(errors)} succeeded, {len(errors)} failed.", flush=True)
    for path, msg in errors:
        print(f"FAILED: {path} -> {msg}", flush=True)


if __name__ == "__main__":
    main()
