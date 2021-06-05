import os
from dataclasses import asdict

import pytest
from pydantic import ValidationError

from speech_recognition.configs.train_config import TrainConfig

from ..const import RESOURCE_DIR


def test_model_config():
    data_config_path = os.path.join(RESOURCE_DIR, "configs", "libri_config.yml")
    model_config_path = os.path.join(RESOURCE_DIR, "configs", "las_small.yml")

    correct_train_config = {
        "data_config": data_config_path,
        "model_config": model_config_path,
        "train_dataset_paths": "hi",
        "dev_dataset_paths": "hello",
        "train_dataset_size": 10,
        "epochs": 1,
        "learning_rate": 1.0,
        "batch_size": 10,
        "dev_batch_size": 20,
    }

    with pytest.raises(ValidationError):
        TrainConfig()

    with pytest.raises(FileNotFoundError):
        TrainConfig(**{**correct_train_config, **{"data_config": "nofile"}})

    config = TrainConfig(**correct_train_config)
    assert config.audio_pad_length is None

    del correct_train_config["data_config"]
    del correct_train_config["model_config"]
    assert all(item in asdict(config).items() for item in correct_train_config.items())
