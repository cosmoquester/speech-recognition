from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List, Union

import yaml
from pydantic.dataclasses import dataclass

from ..models import LAS, DeepSpeech2


class ModelConfig(metaclass=ABCMeta):
    @abstractmethod
    def create_model(self):
        pass

    @abstractproperty
    def model_name(self):
        pass


def get_model_config(model_config_path: str) -> Union["LASConfig", "DeepSpeechConfig"]:
    """
    Load model config file and return ModelConfig instance

    :param model_config_path: model config file path
    :returns: ModelConfig type instance
    """
    with open(model_config_path) as f:
        model_config_dict = yaml.load(f, yaml.SafeLoader)

    model_name = model_config_dict["model_name"].lower()

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

    # Model name
    model_name: str = "LAS"

    def create_model(self) -> LAS:
        return LAS(
            rnn_type=self.rnn_type,
            vocab_size=self.vocab_size,
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            teacher_forcing_rate=self.teacher_forcing_rate,
            pad_id=self.pad_id,
        )


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

    # Model name
    model_name: str = "DeepSpeech2"

    def create_model(self) -> DeepSpeech2:
        return DeepSpeech2(
            num_conv_layers=self.num_conv_layers,
            channels=self.channels,
            filter_sizes=self.filter_sizes,
            strides=self.strides,
            rnn_type=self.rnn_type,
            num_reccurent_layers=self.num_reccurent_layers,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            vocab_size=self.vocab_size,
            blank_index=self.blank_index,
            pad_index=self.pad_index,
        )
