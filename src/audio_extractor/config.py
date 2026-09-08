import argparse
import sys
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class TranscriptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(default="small", min_length=1)
    language: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = Field(default="auto", min_length=1)
    beam_size: int = Field(default=5, gt=0)


class ExtractorConfig(TranscriptionConfig):
    format: Literal["text", "srt", "vtt", "json"] = "text"


class BatchConfig(TranscriptionConfig):
    language: str | None = "zh"
    device: Literal["cpu", "cuda"] = "cuda"
    limit: int = Field(default=0, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)


def load_config(path: Path | None = None) -> AppConfig:
    target = path if path is not None else Path.cwd() / "audio-extractor.toml"
    if path is None and not target.exists():
        return AppConfig()
    with target.open("r", encoding="utf-8", newline="\n") as stream:
        return AppConfig.model_validate(tomllib.loads(stream.read()))


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def parse_configured_args(
    parser: argparse.ArgumentParser,
    section: Literal["extractor", "batch"],
    config_type: type[ConfigT],
) -> tuple[argparse.Namespace, ConfigT]:
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file (default: audio-extractor.toml in the current directory)",
    )
    # Suppress option defaults so only explicitly supplied CLI values override TOML.
    for action in parser._actions:
        if action.dest in config_type.model_fields:
            action.default = argparse.SUPPRESS
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        selected = getattr(config, section)
        values = selected.model_dump()
        if section == "batch":
            shared = config.extractor.model_dump(exclude_unset=True)
            values.update(
                {
                    key: value
                    for key, value in shared.items()
                    if key in TranscriptionConfig.model_fields
                }
            )
            values.update(config.batch.model_dump(exclude_unset=True))
        values.update(
            {
                key: value
                for key, value in vars(args).items()
                if key in config_type.model_fields
            }
        )
        settings = config_type.model_validate(values)
    except (OSError, ValueError) as error:
        parser.error(f"Invalid configuration: {error}")
    for key, value in settings.model_dump().items():
        setattr(args, key, value)
    return args, settings
