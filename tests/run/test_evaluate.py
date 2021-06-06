import tempfile

import pytest

from speech_recognition.run.evaluate import main, parser

from ..const import (
    DEFAULT_DS_CHECKPOINT,
    DEFAULT_DS_CONFIG,
    DEFAULT_LAS_CHECKPOINT,
    DEFAULT_LAS_CONFIG,
    DEFAULT_LIBRI_CONFIG,
    SPM_LIBRI_CONFIG,
    TFRECORD_DATASET_PATH,
    WAV_DATASET_PATH,
)


@pytest.mark.parametrize("use_tfrecord", [False, True])
@pytest.mark.parametrize("beam_search", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize(
    "model", [(DEFAULT_LAS_CONFIG, DEFAULT_LAS_CHECKPOINT), (DEFAULT_DS_CONFIG, DEFAULT_DS_CHECKPOINT)]
)
def test_evaluate(model, mixed_precision, beam_search, use_tfrecord):
    model_config_path, model_checkpoint = model

    with tempfile.NamedTemporaryFile() as tmpfile:
        arguments = [
            "--data-config",
            DEFAULT_LIBRI_CONFIG,
            "--model-config",
            model_config_path,
            "--dataset-paths",
            TFRECORD_DATASET_PATH if use_tfrecord else WAV_DATASET_PATH,
            "--model-path",
            model_checkpoint,
            "--output-path",
            tmpfile.name,
            "--sp-model-path",
            SPM_LIBRI_CONFIG,
            "--batch-size",
            "4",
            "--device",
            "CPU",
        ]
        if mixed_precision:
            arguments.append("--mixed-precision")
        if use_tfrecord:
            arguments.append("--use-tfrecord")
        if beam_search:
            arguments.extend(["--beam-size", "2"])

        assert main(parser.parse_args(arguments)) is None
        assert tmpfile.read()
