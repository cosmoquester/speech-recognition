from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, BatchNormalization, Conv2D, Dense, Dropout, Embedding, SimpleRNN

from ..measure import SparseCategoricalAccuracy, SparseCategoricalCrossentropy
from .model_proto import ModelProto


def get_rnn_cls(rnn_type: str) -> Union[SimpleRNN, LSTM, GRU]:
    if rnn_type == "rnn":
        return SimpleRNN
    if rnn_type == "lstm":
        return LSTM
    if rnn_type == "gru":
        return GRU
    raise ValueError(f"rnn_type: {rnn_type} is invalid!")


class AdditiveAttention(tf.keras.layers.Layer):
    """
    Attention to inform decoder layers of encoder output that is related decoder input.

    Arguments:
        hidden_dim: Integer, the hidden dimension size of SampleModel.
    Call arguments:
        query: A 3D tensor, with shape of `[BatchSize, HiddenDim]`.
        key: A 3D tensor, with shape of `[BatchSize, SequenceLength, HiddenDim]`.
        value: A 3D tensor, with shape of `[BatchSize, SequenceLength, HiddenDim]`.
        attention_mask: A 2D bool Tensor, with shape of `[BatchSize, SequenceLength]`.
                        The values of timestep which should be ignored is `False`.
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

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, attention_mask: tf.Tensor, training=None):
        # [BatchSize, 1, HiddenDim]
        query = self.query_weight(query)[:, tf.newaxis, :]
        # [BatchSize, HiddenDim, SequenceLength]
        key = tf.transpose(self.key_weight(key), [0, 2, 1])

        # [BatchSize, 1, SequenceLength]
        weight = tf.matmul(query, key)
        weight -= 1e9 * (1.0 - tf.cast(tf.expand_dims(attention_mask, axis=1), query.dtype))
        attention_probs = tf.nn.softmax(weight, axis=-1)

        # [BatchSize, HiddenDim]
        context = tf.squeeze(tf.matmul(attention_probs, value), axis=1)
        return context


class BiRNN(tf.keras.layers.Layer):
    """
    Custom Bi-directional RNN Wrapper because of issue.
    https://github.com/tensorflow/tensorflow/issues/48880

    Arguments:
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
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
        rnn_type: str,
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs,
    ):
        super(BiRNN, self).__init__(**kwargs)

        rnn_cls = get_rnn_cls(rnn_type)
        self.forward_rnn = rnn_cls(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name="forward_rnn",
        )
        self.backward_rnn = rnn_cls(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            go_backwards=True,
            name="backward_rnn",
        )

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor, initial_state: Optional[tf.Tensor] = None, training: bool = False
    ) -> List:
        if initial_state is None:
            forward_states = None
            backward_states = None
        else:
            num_states = len(initial_state) // 2
            forward_states = initial_state[:num_states]
            backward_states = initial_state[num_states:]

        forward_output, *forward_states = self.forward_rnn(
            inputs, mask=mask, initial_state=forward_states, training=training
        )
        backward_output, *backward_states = self.backward_rnn(
            inputs, mask=mask, initial_state=backward_states, training=training
        )
        output = tf.concat([forward_output, tf.reverse(backward_output, axis=[1])], axis=-1)
        return [output] + forward_states + backward_states


class Listener(tf.keras.layers.Layer):
    """
    Listener of LAS model.

    Arguments:
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
        encoder_hidden_dim: Integer, the hidden dimension size of SampleModel encoder.
        decoder_hidden_dim: Integer, the hidden dimension size of SampleModel decoder.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        dropout: Float,
    Call arguments:
        audio: A 3D tensor, with shape of `[BatchSize, TimeStep, DimAudio]`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    Output Shape:
        audio: `[BatchSize, ReducedTimeStep, HiddenDim]`
    """

    def __init__(
        self,
        rnn_type: str,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        num_encoder_layers: int,
        dropout: float,
        **kwargs,
    ):
        super(Listener, self).__init__(**kwargs)

        self.kernel_sizes = (3, 3)
        self.strides = 2
        self.AUDIO_PAD_VALUE = 0.0

        self.conv1 = Conv2D(32, self.kernel_sizes, strides=self.strides, name="conv1")
        self.conv2 = Conv2D(32, self.kernel_sizes, strides=self.strides, name="conv2")

        self.encoder_layers = [
            BiRNN(rnn_type, encoder_hidden_dim, dropout, name=f"encoder_layer{i}") for i in range(num_encoder_layers)
        ]
        self.projection = [Dense(encoder_hidden_dim * 2, name=f"proejction{i}") for i in range(num_encoder_layers)]
        self.batch_norm = [BatchNormalization(name=f"batch_normalization{i}") for i in range(num_encoder_layers)]
        self.hidden_states_proj = Dense(decoder_hidden_dim, name="hidden_state_proj")
        if rnn_type == "lstm":
            self.cell_states_proj = Dense(decoder_hidden_dim, name="cell_state_proj")

        self.dropout = Dropout(dropout, name="dropout")

    def call(self, audio: tf.Tensor, training: Optional[bool] = None) -> List[tf.Tensor]:
        # [BatchSize, ReducedTimeStep]
        mask = self._audio_mask(audio)
        batch_size = tf.shape(audio)[0]

        # [BatchSize, ReducedTimeStep, ReducedFrequencyDim, 32]
        audio = self.dropout(self.conv1(audio), training=training)
        audio = self.dropout(self.conv2(audio), training=training)
        sequence_length = -1 if audio.shape[1] is None else audio.shape[1]
        audio = tf.reshape(audio, [batch_size, sequence_length, audio.shape[2] * audio.shape[3]])

        # Encode
        # audio: [BatchSize, ReducedTimeStep, HiddenDim]
        states = None
        for encoder_layer, projection, batch_norm in zip(self.encoder_layers, self.projection, self.batch_norm):
            audio, *states = encoder_layer(audio, mask, states, training=training)
            audio = tf.nn.relu(batch_norm(projection(audio)))

        # Concat states of two directions
        if len(states) == 2:
            states = [self.hidden_states_proj(tf.concat(states, axis=-1))]
        elif len(states) == 4:
            states = [
                self.hidden_states_proj(tf.concat(states[::2], axis=-1)),
                self.cell_states_proj(tf.concat(states[1::2], axis=-1)),
            ]
        return [audio, mask] + states

    def _audio_mask(self, audio):
        kernel_size = self.kernel_sizes[0]
        batch_size, sequence_length = tf.unstack(tf.shape(audio)[:2], 2)
        mask = tf.reduce_any(tf.reshape(audio, [batch_size, sequence_length, -1]) != self.AUDIO_PAD_VALUE, axis=2)
        sequence_length -= kernel_size - self.strides
        sequence_length = sequence_length // self.strides
        sequence_length -= kernel_size - self.strides
        sequence_length = sequence_length // self.strides
        sequence_length *= self.strides**2

        mask = tf.reshape(mask[:, :sequence_length], [batch_size, -1, self.strides**2])
        mask = tf.reduce_any(mask, axis=2)
        return mask


class AttendAndSpeller(tf.keras.layers.Layer):
    """
    Attend and Speller of LAS model.

    Arguments:
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
        vocab_size: Integer, the size of vocabulary.
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        dropout: Float,
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

    def __init__(
        self,
        rnn_type: str,
        vocab_size: int,
        hidden_dim: int,
        num_decoder_layers: int,
        dropout: float,
        pad_id: int,
        **kwargs,
    ):
        super(AttendAndSpeller, self).__init__(**kwargs)

        rnn_cls = get_rnn_cls(rnn_type)

        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.decoder_layers = [
            rnn_cls(hidden_dim, dropout=dropout, return_state=True, name=f"decoder_layer{i}")
            for i in range(num_decoder_layers)
        ]
        self.attention = AdditiveAttention(hidden_dim, name="attention")
        self.feedforward = Dense(vocab_size, name="feedfoward")
        self.dropout = Dropout(dropout, name="dropout")

    def call(
        self,
        audio_output: tf.Tensor,
        decoder_input: tf.Tensor,
        attention_mask: tf.Tensor,
        states: List,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        # [BatchSize, 1]
        mask = tf.expand_dims(decoder_input != self.pad_id, axis=1)
        # [BatchSize, HiddenDim]
        decoder_input = self.dropout(self.embedding(decoder_input), training=training)

        # Decode
        # decoder_input: [BatchSize, HiddenDim]
        context = self.attention(states[0], audio_output, audio_output, attention_mask)
        decoder_input = tf.concat([decoder_input, context], axis=-1)

        for decoder_layer in self.decoder_layers:
            decoder_input, *states = decoder_layer(
                tf.expand_dims(decoder_input, axis=1), initial_state=states, mask=mask, training=training
            )

        # [BatchSize, VocabSize]
        output = self.feedforward(self.dropout(decoder_input))
        return [output] + states


class LAS(ModelProto):
    """
    This is Listen, Attend and Spell(LAS) model for speech recognition.

    Arguments:
        rnn_type: String, the type of rnn. one of ['rnn', 'lstm', 'gru'].
        vocab_size: Integer, the size of vocabulary.
        encoder_hidden_dim: Integer, the hidden dimension size of SampleModel encoder.
        decoder_hidden_dim: Integer, the hidden dimension size of SampleModel decoder.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        teacher_forcing_rate: Float, the rate of using teacher forcing.
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

    model_checkpoint_path = "model-{epoch}epoch-{val_loss:.4f}loss_{val_accuracy:.4f}acc.ckpt"

    def __init__(
        self,
        rnn_type: str,
        vocab_size: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        teacher_forcing_rate: float,
        pad_id: int = 0,
        **kwargs,
    ):
        super(LAS, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.teacher_forcing_rate = teacher_forcing_rate

        self.listener = Listener(
            rnn_type, encoder_hidden_dim, decoder_hidden_dim, num_encoder_layers, dropout, name="listener"
        )
        self.attend_and_speller = AttendAndSpeller(
            rnn_type, vocab_size, decoder_hidden_dim, num_decoder_layers, dropout, pad_id, name="attend_and_speller"
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        # audio: [BatchSize, TimeStep, DimAudio], decoder_input: [BatchSize, NumTokens]
        audio_input, decoder_input = inputs

        # Use range on TPU because of issue https://github.com/tensorflow/tensorflow/issues/49469
        if decoder_input.shape[1]:
            token_length = decoder_input.shape[1]
            index_iter = range(token_length)
        else:
            token_length = tf.shape(decoder_input)[1]
            index_iter = tf.range(token_length)

        audio_output, attention_mask, *states = self.listener(audio_input, training=training)
        outputs = tf.TensorArray(
            audio_output.dtype, size=token_length, infer_shape=False, element_shape=[None, self.vocab_size]
        )

        use_teacher_forcing = tf.random.uniform((), 0, 1) < self.teacher_forcing_rate
        output = tf.zeros([tf.shape(audio_output)[0], self.vocab_size], dtype=audio_output.dtype)
        for i in index_iter:
            if use_teacher_forcing or i == 0:
                decoder_input_t = tf.gather(decoder_input, i, axis=1)
            else:
                decoder_input_t = tf.argmax(output, axis=-1, output_type=tf.int32)

            output, *states = self.attend_and_speller(
                audio_output, decoder_input_t, attention_mask, states, training=training
            )
            outputs = outputs.write(i, output)

        result = tf.transpose(outputs.stack(), [1, 0, 2])
        return result

    def get_loss_fn(self):
        return SparseCategoricalCrossentropy(self.pad_id)

    def get_metrics(self):
        return [SparseCategoricalAccuracy(self.pad_id)]

    @staticmethod
    def get_batching_shape(
        audio_pad_length: Optional[int], token_pad_length: Optional[int], frequency_dim: int, feature_dim: int
    ):
        if token_pad_length is not None:
            token_pad_length = token_pad_length - 1
        return (([audio_pad_length, frequency_dim, feature_dim], [token_pad_length]), [token_pad_length])

    @staticmethod
    def make_example(audio: tf.Tensor, tokens: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Make training example from audio input and token output.
        Output should be (MODEL_INPUT, Y_TRUE)

        :param audio: input audio tensor
        :param tokens: target tokens shaped [NumTokens]
        :returns: return input as output by default
        """
        return (audio, tokens[:-1]), tokens[1:]
