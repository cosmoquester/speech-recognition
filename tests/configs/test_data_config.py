import os
from dataclasses import asdict

import pytest
import yaml
from pydantic import ValidationError

from speech_recognition.configs.data_config import DataConfig

from ..const import RESOURCE_DIR


def test_data_config():
    with open(os.path.join(RESOURCE_DIR, "configs", "libri_config.yml")) as f:
        correct_config = yaml.load(f, yaml.SafeLoader)

    with pytest.raises(TypeError):
        DataConfig()

    with pytest.raises(ValidationError):
        DataConfig(**{**correct_config, **{"file_format": "hello"}})

    config = DataConfig(**correct_config)
    assert asdict(config) == correct_config

    assert config.feature_dim == 1
