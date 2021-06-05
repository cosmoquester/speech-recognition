from typing import List

import yaml
from pydantic.dataclasses import dataclass


class ModelConfig:
    @staticmethod
    def from_yaml(model_config_path: str) -> "ModelConfig":
        """
        Load model config file and return ModelConfig instance

        :param model_config_path: model config file path
        :returns: ModelConfig type instance
        """
        with open(model_config_path) as f:
            model_config_dict = yaml.load(f, yaml.SafeLoader)

        model_name = model_config_dict.pop("model_name").lower()

        if model_name in ["ds2", "deepspeech2"]:
            return DeepSpeechConfig(**model_config_dict)
        if model_name in ["las"]:
            return LASConfig(**model_config_dict)
        raise ValueError(f"Model Name: {model_name} is invalid!")


@dataclass
class LASConfig(ModelConfig):
    """Config for LAS model initialize"""

    # RNN Type: one of ['rnn', 'lstm', 'gru']
    rnn_type: str
    # Vocab Size
    vocab_size: int
    # Encoder Hidden Dimension
    encoder_hidden_dim: int
    # Decoder Hidden Dimension
    decoder_hidden_dim: int
    # Encoder Layers
    num_encoder_layers: int
    # Decoder Layers
    num_decoder_layers: int
    # Dropout Rate
    dropout: float
    # teacher forcing rate
    teacher_forcing_rate: float
    # Pad Token ID
    pad_id: int


@dataclass
class DeepSpeechConfig(ModelConfig):
    """Config for DeepSpeech2 model initialize"""

    # number of convolution layers
    num_conv_layers: int
    # number of channel for each layers
    channels: List[int]
    # number of filter size for each layers
    filter_sizes: List[List[int]]
    # number of stride for each layers
    strides: List[List[int]]
    # type of rnn, one of ['rnn', 'lstm', 'gru']
    rnn_type: str
    # number of reccurent layers
    num_reccurent_layers: int
    # hidden dimension size of rnn
    hidden_dim: int
    # dropout rate
    dropout: float
    # reccurent dropout rate
    recurrent_dropout: float
    # size of vocabulary
    vocab_size: int
    # the index of blank token separating token
    blank_index: int
    # the index of pad token
    pad_index: int
