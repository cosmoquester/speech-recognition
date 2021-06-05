import pytest
import tensorflow as tf

from speech_recognition.models.las import LAS, AdditiveAttention, BiRNN


@pytest.mark.parametrize(
    "hidden_dim,sequence_length,batch_size", [(128, 13, 5), (256, 33, 43), (111, 111, 111), (1, 1, 1)]
)
def test_additive_attention(hidden_dim, sequence_length, batch_size):
    attention = AdditiveAttention(hidden_dim)
    query = tf.random.normal([batch_size, hidden_dim])
    key = tf.random.normal([batch_size, sequence_length, hidden_dim])
    value = tf.random.normal([batch_size, sequence_length, hidden_dim])
    mask = tf.random.normal([batch_size, sequence_length]) > 0.5

    output = attention(query, key, value, mask)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, hidden_dim])


@pytest.mark.parametrize(
    "rnn_type,units,dropout,batch_size,sequence_length,feature_dim,pad_length",
    [
        ("rnn", 13, 0.1, 23, 11, 8, 3),
        ("lstm", 33, 0.2, 34, 41, 2, 4),
        ("gru", 111, 0.3, 55, 3, 99, 5),
    ],
)
def test_bi_rnn(rnn_type, units, dropout, batch_size, sequence_length, feature_dim, pad_length):
    rnn = BiRNN(rnn_type, units, dropout, dropout)

    input = tf.random.normal([batch_size, sequence_length, feature_dim])
    mask = tf.cast(tf.random.normal([batch_size, sequence_length]) > 0.1, tf.int32)
    output, *states = rnn(input, mask)
    assert output.shape == [batch_size, sequence_length, units * 2]
    assert states[0].shape == [batch_size, units]

    padded_input = tf.concat([input, tf.random.normal([batch_size, pad_length, feature_dim])], axis=1)
    padded_mask = tf.concat([mask, tf.zeros([batch_size, pad_length], tf.int32)], axis=1)
    padded_output, *padded_states = rnn(padded_input, padded_mask)
    assert padded_output.shape == [batch_size, sequence_length + pad_length, units * 2]
    assert padded_states[0].shape == [batch_size, units]

    tf.debugging.assert_equal(output, padded_output[:, :-pad_length])


@pytest.mark.parametrize(
    "rnn_type,vocab_size,hidden_dim,num_encoder_layers,num_decoder_layers,dropout,teacher_forcing_rate,batch_size,audio_dim,audio_sequence_length,num_tokens",
    [
        ("rnn", 12345, 122, 1, 2, 0.1, 0.234, 3, 88, 12, 8),
        ("lstm", 3030, 320, 3, 5, 0.234, 0.4, 1, 34, 33, 1),
        ("gru", 12, 12, 12, 12, 0.4, 0.99, 12, 12, 12, 12),
    ],
)
def test_las(
    rnn_type,
    vocab_size,
    hidden_dim,
    num_encoder_layers,
    num_decoder_layers,
    dropout,
    teacher_forcing_rate,
    batch_size,
    audio_dim,
    audio_sequence_length,
    num_tokens,
):
    las = LAS(
        rnn_type,
        vocab_size,
        hidden_dim,
        hidden_dim,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        teacher_forcing_rate,
    )
    audio = tf.random.normal([batch_size, audio_sequence_length, audio_dim, 3])
    tokens = tf.random.uniform([batch_size, num_tokens], 0, vocab_size, tf.int32)
    output = las((audio, tokens))
    tf.debugging.assert_equal(tf.shape(output), [batch_size, num_tokens, vocab_size])
