import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from audio_extractor import batch, cli
from audio_extractor.config import load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "settings.toml"

    def write_config(self, text):
        with self.config.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

    def test_defaults_without_file(self):
        with patch("audio_extractor.config.Path.cwd", return_value=self.root):
            config = load_config()
        self.assertEqual(config.extractor.device, "cpu")
        self.assertEqual(config.batch.device, "cuda")

    def test_default_discovery(self):
        self.write_config('[extractor]\nmodel = "large-v3"\n')
        self.config.rename(self.root / "audio-extractor.toml")
        with patch("audio_extractor.config.Path.cwd", return_value=self.root):
            self.assertEqual(load_config().extractor.model, "large-v3")

    def test_invalid_config(self):
        for text in (
            "[extractor]\nbeam_size = 0",
            '[extractor]\nbeam_size = "5"',
            '[extractor]\ndevice = "bad"',
            '[extractor]\nformat = "bad"',
            "[batch]\nlimit = -1",
            "[unknown]\nvalue = 1",
            "[extractor",
        ):
            with self.subTest(text=text):
                self.write_config(text)
                with self.assertRaises((ValueError, ValidationError)):
                    load_config(self.config)
        with self.assertRaises(FileNotFoundError):
            load_config(self.root / "missing.toml")

    def test_extractor_cli_overrides_toml(self):
        self.write_config(
            '[extractor]\nmodel = "large-v3"\nbeam_size = 7\nformat = "srt"\n'
        )
        audio = self.root / "audio.wav"
        audio.touch()
        info = SimpleNamespace(language="en", language_probability=1)
        with (
            patch(
                "sys.argv",
                [
                    "audio-extractor",
                    str(audio),
                    "--config",
                    str(self.config),
                    "--model",
                    "tiny",
                ],
            ),
            patch.object(cli, "load_model") as model,
            patch.object(cli, "transcribe", return_value=([], info)) as transcribe,
            patch.dict(
                cli.FORMATTERS, {"srt": lambda segments, stream: stream.write("SRT")}
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            cli.main()
        self.assertEqual(model.call_args.kwargs["model_size"], "tiny")
        self.assertEqual(transcribe.call_args.kwargs["beam_size"], 7)
        self.assertEqual(output.getvalue(), "SRT")

    def test_batch_inheritance_and_override(self):
        self.write_config(
            '[extractor]\nmodel = "medium"\nbeam_size = 8\ndevice = "cpu"\n[batch]\nbeam_size = 4\nlimit = 1\n'
        )
        (self.root / "a.wav").touch()
        (self.root / "b.wav").touch()
        with (
            patch(
                "sys.argv",
                [
                    "audio-extractor-batch",
                    str(self.root),
                    "--config",
                    str(self.config),
                    "--beam-size",
                    "2",
                ],
            ),
            patch.object(batch, "load_model") as model,
            patch.object(batch, "transcribe", return_value=([], None)) as transcribe,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            batch.main()
        self.assertEqual(model.call_args.kwargs["model_size"], "medium")
        self.assertEqual(model.call_args.kwargs["device"], "cpu")
        self.assertEqual(transcribe.call_args.kwargs["beam_size"], 2)
        self.assertEqual(transcribe.call_count, 1)

    def test_help_without_reading_config(self):
        with (
            patch("sys.argv", ["audio-extractor", "--help"]),
            patch("audio_extractor.config.load_config") as load,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as exit_info:
                cli.main()
        self.assertEqual(exit_info.exception.code, 0)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
