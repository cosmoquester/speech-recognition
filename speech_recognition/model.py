from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Embedding, Masking


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

    def call(self, inputs: tf.Tensor, initial_state: Optional[tf.Tensor] = None) -> List:
        if initial_state is None:
            forward_states = None
            backward_states = None
        else:
            forward_states = initial_state[:2]
            backward_states = initial_state[2:]

        forward_output, *forward_states = self.forward_rnn(inputs, initial_state=forward_states)
        backward_output, *backward_states = self.backward_rnn(inputs, initial_state=backward_states)
        output = tf.concat([forward_output, backward_output], axis=-1)
        return [output] + forward_states + backward_states


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
            audio: A 3D tensor, with shape of `[BatchSize, TimeStep]`.
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

        self.conv1 = Conv1D(32, 3, strides=2, name="conv1")
        self.conv2 = Conv1D(32, 3, strides=2, name="conv2")

        self.encoder_layers = [BiLSTM(hidden_dim // 2, name=f"encoder_layer{i}") for i in range(num_encoder_layers)]

        self.embedding = Embedding(vocab_size, hidden_dim)
        self.masking = Masking(pad_id)
        self.decoder_layers = [
            LSTM(hidden_dim, return_state=True, return_sequences=True, name=f"decoder_layer{i}")
            for i in range(num_decoder_layers)
        ]
        self.attention = AdditiveAttention(hidden_dim, name="attention")
        self.feedforward = Dense(vocab_size, name="feedfoward")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        # audio: [NumBatch, TimeStep, DimAudio], decoder_input: [NumBatch, NumTokens]
        audio, decoder_input = inputs

        # [NumBatch, TimeStep // 4, 32]
        audio = self.conv2(self.conv1(audio))
        # [NumBatch, NumTokens, HiddenDim]
        decoder_input = self.masking(self.embedding(decoder_input))

        # Encode
        # audio: [NumBatch, TimeStep // 4, HiddenDim // 2]
        states = None
        for encoder_layer in self.encoder_layers:
            audio, *states = encoder_layer(audio, states)

        # Decode
        # decoder_input: [NumBatch, NumTokens, HiddenDim]
        states = tf.concat(states[::2], axis=-1), tf.concat(states[1::2], axis=-1)
        decoder_input, *states = self.decoder_layers[0](decoder_input, states)
        for decoder_layer in self.decoder_layers[1:]:
            context = self.attention(states[0], audio, audio)
            decoder_input, *states = decoder_layer(tf.concat([context, decoder_input], axis=1), states)
            decoder_input = decoder_input[:, 1:, :]

        # [NumBatch, VocabSize]
        output = self.feedforward(decoder_input[:, -1, :])
        return output
