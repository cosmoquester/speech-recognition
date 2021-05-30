from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Optional

import tensorflow as tf


class ModelProto(tf.keras.Model, metaclass=ABCMeta):
    """Prototype structure of ASR models"""

    def __init__(self, *args, **kwargs):
        super(ModelProto, self).__init__(*args, **kwargs)

    @abstractmethod
    def call(self, inputs, training: Optional[bool] = None) -> tf.Tensor:
        pass

    @abstractproperty
    def loss_fn(self) -> tf.Tensor:
        pass

    @abstractproperty
    def metrics(self):
        """Tensorflow metrics for model compile"""
        pass

    @abstractstaticmethod
    def get_batching_shape(
        audio_pad_length: Optional[int], token_pad_length: Optional[int], num_mel_bins: int, feature_dim: int
    ):
        """
        Return shapes of padded batch.

        :param audio_pad_length: audio input length
        :param token_pad_length: target token pad length
        :param num_mel_bins: number of mel bins
        :param feature_dim: feature dimension of each time and frequency, 3 if use delta accelerate else 1
        """
        pass

    @abstractstaticmethod
    def make_example(audio: tf.Tensor, tokens: tf.Tensor):
        """
        Make training example from audio input and token output.
        Output should be (MODEL_INPUT, Y_TRUE)

        :param audio: input audio tensor
        :param tokens: target tokens shaped [NumTokens]
        """
        pass
