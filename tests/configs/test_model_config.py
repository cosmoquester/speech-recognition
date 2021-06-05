import os

import pytest
import yaml
from pydantic import ValidationError

from speech_recognition.configs.model_config import DeepSpeechConfig, LASConfig, ModelConfig

from ..const import RESOURCE_DIR


def test_model_config():
    las_config_path = os.path.join(RESOURCE_DIR, "configs", "las_small.yml")
    ds_config_path = os.path.join(RESOURCE_DIR, "configs", "deepspeech.yml")

    with open(las_config_path) as f:
        correct_las_config = yaml.load(f, yaml.SafeLoader)
        del correct_las_config["model_name"]

    with open(ds_config_path) as f:
        correct_dsconfig = yaml.load(f, yaml.SafeLoader)
        del correct_dsconfig["model_name"]

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

    assert LASConfig(**correct_las_config) == ModelConfig.from_yaml(las_config_path)
    assert DeepSpeechConfig(**correct_dsconfig) == ModelConfig.from_yaml(ds_config_path)
