import os
import random
import tempfile

import pytest

from speech_recognition.configs import TrainConfig
from speech_recognition.run.train import main, parser

from ..const import (
    DEFAULT_LIBRI_CONFIG,
    SP_MODEL_LIBRI,
    TEST_DS_CONFIG,
    TEST_LAS_CONFIG,
    TFRECORD_DATASET_PATH,
    WAV_DATASET_PATH,
)


@pytest.mark.interferable
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("model_config_path", [TEST_LAS_CONFIG, TEST_DS_CONFIG])
def test_train(model_config_path, mixed_precision):
    max_over_policy = random.choice([None, "slice", "filter"])
    use_tfrecord = random.choice([True, False])

    with tempfile.TemporaryDirectory() as tmpdir:
        arguments = [
            "--data-config",
            DEFAULT_LIBRI_CONFIG,
            "--model-config",
            model_config_path,
            "--sp-model-path",
            SP_MODEL_LIBRI,
            "--train-dataset-paths",
            TFRECORD_DATASET_PATH if use_tfrecord else WAV_DATASET_PATH,
            "--dev-dataset-paths",
            TFRECORD_DATASET_PATH if use_tfrecord else WAV_DATASET_PATH,
            "--output-path",
            tmpdir,
            "--steps-per-epoch",
            "2",
            "--epochs",
            "1",
            "--shuffle-buffer-size",
            "30",
            "--device",
            "CPU",
            "--batch-size",
            "2",
            "--dev-batch-size",
            "2",
            "--learning-rate",
            "1e-3",
            "--train-dataset-size",
            "1",
        ]
        if mixed_precision:
            arguments.append("--mixed-precision")
        if use_tfrecord:
            arguments.append("--use-tfrecord")
        if max_over_policy is not None:
            arguments.extend(["--max-over-policy", max_over_policy])

        assert main(TrainConfig(**vars(parser.parse_args(arguments)))) is None
        assert os.path.exists(os.path.join(tmpdir, "logs", "train"))
        assert os.path.exists(os.path.join(tmpdir, "models", "checkpoint"))
