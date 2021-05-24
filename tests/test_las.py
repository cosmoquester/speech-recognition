import pytest
import tensorflow as tf

from speech_recognition.models.las import LAS, AdditiveAttention


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
    "rnn_type,vocab_size,hidden_dim,num_encoder_layers,num_decoder_layers,batch_size,audio_dim,audio_sequence_length,num_tokens",
    [
        ("rnn", 12345, 122, 1, 2, 3, 88, 12, 8),
        ("lstm", 3030, 320, 3, 5, 1, 34, 33, 1),
        ("gru", 12, 12, 12, 12, 12, 12, 12, 12),
    ],
)
def test_las(
    rnn_type,
    vocab_size,
    hidden_dim,
    num_encoder_layers,
    num_decoder_layers,
    batch_size,
    audio_dim,
    audio_sequence_length,
    num_tokens,
):
    las = LAS(rnn_type, vocab_size, hidden_dim, hidden_dim, num_encoder_layers, num_decoder_layers, 0.1)
    audio = tf.random.normal([batch_size, audio_sequence_length, audio_dim, 3])
    tokens = tf.random.uniform([batch_size, num_tokens], 0, vocab_size, tf.int32)

    output = las((audio, tokens))
    tf.debugging.assert_equal(tf.shape(output), [batch_size, num_tokens, vocab_size])
