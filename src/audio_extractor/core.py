import json
import os
import sys
from pathlib import Path
from typing import Callable

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".wma"}


def setup_cuda_dll_paths() -> None:
    site = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia = site / "nvidia"
    if not nvidia.is_dir():
        return
    dll_dirs = []
    for pkg in ("cublas", "cuda_runtime", "cuda_nvrtc", "cudnn"):
        dll_dir = nvidia / pkg / "bin"
        if dll_dir.is_dir():
            dll_dirs.append(str(dll_dir))
    os.environ["PATH"] = ";".join(dll_dirs) + ";" + os.environ.get("PATH", "")


def load_model(
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "auto",
):
    setup_cuda_dll_paths()
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(model, audio_path: str, language: str | None = None, beam_size: int = 5):
    return model.transcribe(audio_path, language=language, beam_size=beam_size)


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
