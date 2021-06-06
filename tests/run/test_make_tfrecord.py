import os
import tempfile

from speech_recognition.run.make_tfrecord import main, parser

from ..const import DEFAULT_LIBRI_CONFIG, SPM_LIBRI_CONFIG, WAV_DATASET_PATH


def test_make_tfrecord():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = [
            "--data-config",
            DEFAULT_LIBRI_CONFIG,
            "--dataset-paths",
            WAV_DATASET_PATH,
            "--sp-model-path",
            SPM_LIBRI_CONFIG,
            "--output-dir",
            tmpdir,
        ]

        assert main(parser.parse_args(args)) is None
        assert os.path.exists(os.path.join(tmpdir, "wav_dataset.tfrecord"))
