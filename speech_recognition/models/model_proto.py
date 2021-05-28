from typing import Optional

import tensorflow as tf


class ModelProto(tf.keras.Model):
    """Prototype structure of ASR models"""

    def __init__(self, *args, **kwargs):
        super(ModelProto, self).__init__(*args, **kwargs)

    def call(self, inputs, training: Optional[bool] = None) -> tf.Tensor:
        raise NotImplementedError("Should implement call function!")

    def loss_fn(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Should implement loss_fn!")

    @property
    def metrics(self):
        """Tensorflow metrics for model compile"""
        raise NotImplementedError("Should set metrics")

    @staticmethod
    def get_batching_shape(audio_pad_length: int, token_pad_length: int, num_mel_bins: int):
        """
        Return shapes of padded batch.

        :param audio_pad_length: audio input length
        :param token_pad_length: target token pad length
        :param num_mel_bins: number of mel bins
        """
        raise NotImplementedError("Should implement get_batching_fn!")

    @staticmethod
    def make_example(audio: tf.Tensor, tokens: tf.Tensor):
        """
        Make training example from audio input and token output.
        Output should be (MODEL_INPUT, Y_TRUE)

        :param audio: input audio tensor
        :param tokens: target tokens shaped [NumTokens]
        :returns: return input as output by default
        """
        return audio, tokens
