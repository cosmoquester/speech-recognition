import os

import pytest
import yaml
from pydantic import ValidationError

from speech_recognition.configs.model_config import DeepSpeechConfig, LASConfig, get_model_config

from ..const import RESOURCE_DIR


def test_model_config():
    las_config_path = os.path.join(RESOURCE_DIR, "configs", "las_small.yml")
    ds_config_path = os.path.join(RESOURCE_DIR, "configs", "deepspeech.yml")

    with open(las_config_path) as f:
        correct_las_config = yaml.load(f, yaml.SafeLoader)

    with open(ds_config_path) as f:
        correct_dsconfig = yaml.load(f, yaml.SafeLoader)

    with pytest.raises(TypeError):
        LASConfig()

    with pytest.raises(TypeError):
        DeepSpeechConfig()

    with pytest.raises(ValidationError):
        wrong = {**correct_las_config, **{"vocab_size": "good"}}
        LASConfig(**wrong)

    with pytest.raises(ValidationError):
        wrong = {**correct_dsconfig, **{"channels": 55}}
        DeepSpeechConfig(**wrong)

    assert LASConfig(**correct_las_config) == get_model_config(las_config_path)
    assert DeepSpeechConfig(**correct_dsconfig) == get_model_config(ds_config_path)
