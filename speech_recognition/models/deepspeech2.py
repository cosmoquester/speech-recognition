from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense

from ..measure import CTCLoss
from .las import BiRNN
from .model_proto import ModelProto


class Convolution(tf.keras.layers.Layer):
    """
    Convolution layer of deepspeech2 model.

    Arguments:
        num_layers: Integer, the number of convolution layers.
        channels: Integer, the number of channel for each layers.
        kernel_sizes: Integer, the number of filter size for each layers.
        strides: Integer, the number of stride for each layers.

    Call arguments:
        audio_input: A 4D tensor, with shape of `[BatchSize, TimeStep, FreqStep, FeatureDim]`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        2D tensor with shape:
            `[BatchSize, ReducedTimeStep, ReducedFreqStep * HiddenDim]`
    """

    def __init__(
        self,
        num_layers: int,
        channels: List[int],
        kernel_sizes: List[List[int]],
        strides: List[List[int]],
        **kwargs,
    ):
        super(Convolution, self).__init__(**kwargs)

        assert (
            num_layers == len(channels) == len(kernel_sizes) == len(strides)
        ), f"Convolution parameter number is invalid!"

        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.conv_layers = [
            Conv2D(channel, kernel_size, stride, name=f"conv{i}")
            for i, (channel, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides))
        ]
        self.AUDIO_PAD_VALUE = 0.0

    def call(self, audio_input: tf.Tensor, training=None):
        # [BatchSize, ReducedTimeStep]
        mask = self._audio_mask(audio_input)

        for conv_layer in self.conv_layers:
            # [BatchSize, ReducedTimeStep, ReducedFreqStep, HiddenDim]
            audio_input = conv_layer(audio_input)

        batch_size = tf.shape(audio_input)[0]
        sequence_length = audio_input.shape[1] or tf.shape(audio_input)[1]

        # [BatchSize, ReducedTimeStep, ReducedFreqStep * HiddenDim]
        output = tf.reshape(audio_input, [batch_size, sequence_length, audio_input.shape[2] * audio_input.shape[3]])
        return output, mask

    def _audio_mask(self, audio):
        batch_size, sequence_length = tf.unstack(tf.shape(audio)[:2], 2)
        mask = tf.reduce_any(tf.reshape(audio, [batch_size, sequence_length, -1]) != self.AUDIO_PAD_VALUE, axis=2)
        for (time_kernel_size, _), (time_stride, _) in zip(self.kernel_sizes, self.strides):
            sequence_length -= time_kernel_size - time_stride
            sequence_length = sequence_length // time_stride
        stride_complex = tf.reduce_prod([time_stride, _ in self.strides])

        mask = tf.reshape(mask[:, : sequence_length * stride_complex], [batch_size, sequence_length, stride_complex])
        mask = tf.reduce_any(mask, axis=2)
        return mask


class Recurrent(tf.keras.layers.Layer):
    """
    Recurrent layer of deepspeech2 model.
    Use custom bidirectional, refer to https://github.com/tensorflow/tensorflow/issues/48880

    Arguments:
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
        num_layers: Integer, the number of reccurent layers.
        units: Integer, the hidden dimension size of seq2seq rnn.
        dropout: Float, dropout rate.
        recurrent_dropout: Float, reccurent dropout rate.
    Call arguments:
        inputs: [BatchSize, SequenceLength, HiddenDim]
    Output Shape:
        output: `[BatchSize, SequenceLength, HiddenDim]`
    """

    def __init__(
        self,
        rnn_type: str,
        num_layers: int,
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs,
    ):
        super(Recurrent, self).__init__(**kwargs)

        self.rnn_layers = [
            BiRNN(rnn_type, units, dropout, recurrent_dropout, name=f"reccurent_layer{i}") for i in range(num_layers)
        ]
        self.batch_norm = [BatchNormalization(name=f"batch_normalization{i}") for i in range(num_layers)]

    def call(self, audio_input: tf.Tensor, mask: tf.Tensor, training: bool = False) -> List:
        states = None
        for rnn_layer, batch_norm in zip(self.rnn_layers, self.batch_norm):
            output, *states = rnn_layer(audio_input, mask, states, training=training)
            audio_input = batch_norm(output)
        return audio_input


class DeepSpeech2(ModelProto):
    """
    This is DeepSpeech2 model for speech recognition.

    Arguments:
        num_conv_layers: Integer, the number of convolution layers.
        channels: Integer, the number of channel for each layers.
        kernel_sizes: Integer, the number of filter size for each layers.
        strides: Integer, the number of stride for each layers.
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
        num_reccurent_layers: Integer, the number of reccurent layers.
        hidden_dim: Integer, the hidden dimension size of rnn.
        dropout: Float, dropout rate.
        recurrent_dropout: Float, reccurent dropout rate.
        vocab_size: Integer, the size of vocabulary.
        blank_index: Integer, the index of blank token separating token.
        pad_index: Integer, the index of pad token.
    Call arguments:
        audio_input: [BatchSize, TimeStep, FreqStep, FeatureDim]
    Output Shape:
        output: `[BatchSize, SequenceLength, VocabSize]`
    """

    model_checkpoint_path = "model-{epoch}epoch-{val_loss:.4f}loss.ckpt"

    def __init__(
        self,
        num_conv_layers: int,
        channels: List[int],
        kernel_sizes: List[List[int]],
        strides: List[List[int]],
        rnn_type: str,
        num_reccurent_layers: int,
        hidden_dim: int,
        dropout: float,
        recurrent_dropout: float,
        vocab_size: int,
        blank_index: int,
        pad_index: int = 0,
        **kwargs,
    ):
        super(DeepSpeech2, self).__init__(**kwargs)

        self.blank_index = blank_index
        self.pad_index = pad_index

        self.convolution = Convolution(num_conv_layers, channels, kernel_sizes, strides, name="convolution")
        self.recurrent = Recurrent(
            rnn_type, num_reccurent_layers, hidden_dim, dropout, recurrent_dropout, name="recurrent"
        )
        self.fully_connected = Dense(vocab_size)

    def call(self, audio_input, training: bool = False):
        audio, mask = self.convolution(audio_input)
        audio = self.recurrent(audio, mask, training=training) * tf.cast(mask[:, :, tf.newaxis], audio.dtype)
        output = self.fully_connected(audio)
        return output

    def get_loss_fn(self):
        return CTCLoss(self.blank_index, self.pad_index)

    def get_metrics(self):
        return []

    @staticmethod
    def get_batching_shape(
        audio_pad_length: Optional[int], token_pad_length: Optional[int], frequency_dim: int, feature_dim: int
    ):
        return ([audio_pad_length, frequency_dim, feature_dim], [token_pad_length])

    @staticmethod
    def make_example(audio: tf.Tensor, tokens: tf.Tensor):
        """
        Make training example from audio input and token output.
        Output should be (MODEL_INPUT, Y_TRUE)

        :param audio: input audio tensor
        :param tokens: target tokens shaped [NumTokens]
        :returns: use input directly as example
        """
        return audio, tokens
