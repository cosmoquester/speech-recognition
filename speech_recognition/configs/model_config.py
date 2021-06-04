from typing import List

from pydantic.dataclasses import dataclass


@dataclass
class LASConfig:
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
class DeepSpeechConfig:
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
