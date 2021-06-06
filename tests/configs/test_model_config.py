import os

import pytest
import yaml
from pydantic import ValidationError

from speech_recognition.configs.model_config import DeepSpeechConfig, LASConfig, get_model_config

from ..const import DEFAULT_DS_CONFIG, DEFAULT_LAS_CONFIG


def test_model_config():
    with open(DEFAULT_LAS_CONFIG) as f:
        correct_las_config = yaml.load(f, yaml.SafeLoader)

    with open(DEFAULT_DS_CONFIG) as f:
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

    assert LASConfig(**correct_las_config) == get_model_config(DEFAULT_LAS_CONFIG)
    assert DeepSpeechConfig(**correct_dsconfig) == get_model_config(DEFAULT_DS_CONFIG)
