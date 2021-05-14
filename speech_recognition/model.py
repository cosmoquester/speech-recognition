from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Embedding


class AdditiveAttention(tf.keras.layers.Layer):
    """
    Attention to inform decoder layers of encoder output that is related decoder input.

    Arguments:
        hidden_dim: Integer, the hidden dimension size of SampleModel.
    Call arguments:
        query: A 3D tensor, with shape of `[BatchSize, HiddenDim]`.
        key: A 3D tensor, with shape of `[BatchSize, SequenceLength, HiddenDim]`.
        value: A 3D tensor, with shape of `[BatchSize, SequenceLength, HiddenDim]`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        2D tensor with shape:
            `[BatchSize, HiddenDim]`
    """

    def __init__(self, hidden_dim: int, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)

        self.query_weight = Dense(hidden_dim, name="convert_query")
        self.key_weight = Dense(hidden_dim, name="convert_key")

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, training=None):
        # [BatchSize, 1, HiddenDim]
        query = self.query_weight(query)[:, tf.newaxis, :]
        # [BatchSize, HiddenDim, SequenceLength]
        key = tf.transpose(self.key_weight(key), [0, 2, 1])

        # [BatchSize, 1, SequenceLength]
        attention_probs = tf.nn.softmax(tf.matmul(query, key), axis=-1)
        # [BatchSize, 1, HiddenDim]
        context = tf.matmul(attention_probs, value)
        return context


class BiLSTM(tf.keras.layers.Layer):
    """
    Custom Bi-directional RNN Wrapper because of issue.
    https://github.com/tensorflow/tensorflow/issues/48880

    Arguments:
        units: Integer, the hidden dimension size of seq2seq rnn.
        dropout: Float, dropout rate.
        recurrent_dropout: Float, reccurent dropout rate.
    Call arguments:
        inputs: [BatchSize, SequenceLength, HiddenDim]
    Output Shape:
        output: `[BatchSize, SequenceLength, HiddenDim]`
        state: `[BatchSize, HiddenDim]`
    """

    def __init__(
        self,
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs,
    ):
        super(BiLSTM, self).__init__(**kwargs)

        self.forward_rnn = LSTM(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name="forward_rnn",
        )
        self.backward_rnn = LSTM(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            go_backwards=True,
            name="backward_rnn",
        )

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, initial_state: Optional[tf.Tensor] = None) -> List:
        if initial_state is None:
            forward_states = None
            backward_states = None
        else:
            forward_states = initial_state[:2]
            backward_states = initial_state[2:]

        forward_output, *forward_states = self.forward_rnn(inputs, mask=mask, initial_state=forward_states)
        backward_output, *backward_states = self.backward_rnn(inputs, mask=mask, initial_state=backward_states)
        output = tf.concat([forward_output, backward_output], axis=-1)
        return [output] + forward_states + backward_states


class Listener(tf.keras.layers.Layer):
    """
    Listener of LAS model.

    Arguments:
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        pad_id: Float, pad id for audio padding
    Call arguments:
        audio: A 3D tensor, with shape of `[BatchSize, TimeStep, DimAudio]`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        audio: `[BatchSize, ReducedTimeStep, HiddenDim]`
    """

    def __init__(self, hidden_dim: int, num_encoder_layers: int, pad_id: float, **kwargs):
        super(Listener, self).__init__(**kwargs)

        self.filter_sizes = (3, 3)
        self.strides = 2
        self.pad_id = tf.constant(pad_id, tf.float32)

        self.conv1 = Conv2D(32, self.filter_sizes, strides=self.strides, name="conv1")
        self.conv2 = Conv2D(32, self.filter_sizes, strides=self.strides, name="conv2")
        self.encoder_layers = [BiLSTM(hidden_dim, name=f"encoder_layer{i}") for i in range(num_encoder_layers)]

    def call(self, audio: tf.Tensor, training: Optional[bool] = None) -> List[tf.Tensor]:
        # [BatchSize, ReducedTimeStep]
        mask = self._audio_mask(audio)
        batch_size = tf.shape(audio)[0]

        # [BatchSize, ReducedTimeStep, ReducedFrequencyDim, 32]
        audio = self.conv2(self.conv1(audio))
        sequence_length = -1 if audio.shape[1] is None else audio.shape[1]
        audio = tf.reshape(audio, [batch_size, sequence_length, audio.shape[2] * audio.shape[3]])

        # Encode
        # audio: [BatchSize, ReducedTimeStep, HiddenDim]
        states = None
        for encoder_layer in self.encoder_layers:
            audio, *states = encoder_layer(audio, mask, states)
        return [audio] + states

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None])])
    def _audio_mask(self, audio):
        filter_size = self.filter_sizes[0]
        batch_size, sequence_length = tf.unstack(tf.shape(audio)[:2], 2)
        mask = tf.reduce_any(tf.reshape(audio, [batch_size, sequence_length, -1]) != self.pad_id, axis=2)
        sequence_length -= filter_size - self.strides
        sequence_length = sequence_length // self.strides
        sequence_length -= filter_size - self.strides
        sequence_length = sequence_length // self.strides
        sequence_length *= self.strides ** 2

        mask = tf.reshape(mask[:, :sequence_length], [batch_size, -1, self.strides ** 2])
        mask = tf.reduce_any(mask, axis=2)
        return mask


class AttendAndSpeller(tf.keras.layers.Layer):
    """
    Attend and Speller of LAS model.

    Arguments:
        vocab_size: Integer, the size of vocabulary.
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        pad_id: Integer, the id of padding token.
    Call arguments:
        audio_output: A 3D tensor, with shape of `[BatchSize, NumFrames, HiddenDim]`.
        decoder_input: A 3D tensor, with shape of `[BatchSize, NumTokens]`.
                            all values are in [0, VocabSize).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]`
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_decoder_layers: int, pad_id: int, **kwargs):
        super(AttendAndSpeller, self).__init__(**kwargs)

        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.decoder_layers = [
            LSTM(hidden_dim, return_state=True, return_sequences=True, name=f"decoder_layer{i}")
            for i in range(num_decoder_layers)
        ]
        self.attention = AdditiveAttention(hidden_dim, name="attention")
        self.feedforward = Dense(vocab_size, name="feedfoward")

    def call(
        self, audio_output: tf.Tensor, decoder_input: tf.Tensor, states: List, training: Optional[bool] = None
    ) -> tf.Tensor:
        # [BatchSize, NumTokens]
        mask = decoder_input != self.pad_id
        # [BatchSize, NumTokens, HiddenDim]
        decoder_input = self.embedding(decoder_input)

        # Decode
        # decoder_input: [BatchSize, NumTokens, HiddenDim]
        states = tf.concat(states[::2], axis=-1), tf.concat(states[1::2], axis=-1)
        decoder_input, *states = self.decoder_layers[0](decoder_input, initial_state=states, mask=mask)
        mask = tf.concat([mask[:, :1], mask], axis=1)
        for decoder_layer in self.decoder_layers[1:]:
            context = self.attention(states[0], audio_output, audio_output)
            decoder_input, *states = decoder_layer(
                tf.concat([context, decoder_input], axis=1), initial_state=states, mask=mask
            )
            decoder_input = decoder_input[:, 1:, :]

        # [BatchSize, VocabSize]
        output = self.feedforward(decoder_input[:, -1, :])
        return output


class LAS(tf.keras.Model):
    """
    This is Listen, Attend and Spell(LAS) model for speech recognition.

    Arguments:
        vocab_size: Integer, the size of vocabulary.
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        pad_id: Integer, the id of padding token.
    Call arguments:
        inputs: A tuple (encoder_tokens, decoder_tokens)
            audio: A 3D tensor, with shape of `[BatchSize, TimeStep, DimAudio]`.
            decoder_input: A 3D tensor, with shape of `[BatchSize, NumTokens]`.
                                all values are in [0, VocabSize).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]`
    """

    def __init__(
        self,
        vocab_size=16000,
        hidden_dim=1024,
        num_encoder_layers=4,
        num_decoder_layers=2,
        pad_id=0,
        **kwargs,
    ):
        super(LAS, self).__init__(**kwargs)

        self.listener = Listener(hidden_dim // 2, num_encoder_layers, pad_id, name="listener")
        self.attend_and_speller = AttendAndSpeller(
            vocab_size, hidden_dim, num_decoder_layers, pad_id, name="attend_and_speller"
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        # audio: [BatchSize, TimeStep, DimAudio], decoder_input: [BatchSize, NumTokens]
        audio_input, decoder_input = inputs

        audio_output, *states = self.listener(audio_input)
        output = self.attend_and_speller(audio_output, decoder_input, states)
        return output
