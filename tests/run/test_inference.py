import os
import tempfile

import pytest

from speech_recognition.run.inference import main, parser

from ..const import (
    DEFAULT_DS_CHECKPOINT,
    DEFAULT_DS_CONFIG,
    DEFAULT_LAS_CHECKPOINT,
    DEFAULT_LAS_CONFIG,
    DEFAULT_LIBRI_CONFIG,
    SP_MODEL_LIBRI,
    TEST_DATA_DIR,
)

AUDIO_FILE = os.path.join(TEST_DATA_DIR, "audio_files", "test.flac")


@pytest.mark.interferable
@pytest.mark.parametrize("beam_search", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize(
    "model", [(DEFAULT_LAS_CONFIG, DEFAULT_LAS_CHECKPOINT), (DEFAULT_DS_CONFIG, DEFAULT_DS_CHECKPOINT)]
)
def test_inference(model, mixed_precision, beam_search):
    model_config_path, model_checkpoint = model

    with tempfile.NamedTemporaryFile() as tmpfile:
        arguments = [
            "--data-config",
            DEFAULT_LIBRI_CONFIG,
            "--model-config",
            model_config_path,
            "--audio-files",
            AUDIO_FILE,
            "--model-path",
            model_checkpoint,
            "--output-path",
            tmpfile.name,
            "--sp-model-path",
            SP_MODEL_LIBRI,
            "--batch-size",
            "4",
            "--device",
            "CPU",
        ]
        if mixed_precision:
            arguments.append("--mixed-precision")
        if beam_search:
            arguments.extend(["--beam-size", "2"])

        assert main(parser.parse_args(arguments)) is None
        assert tmpfile.read()
