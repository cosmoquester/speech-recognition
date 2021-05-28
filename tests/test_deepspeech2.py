import pytest
import tensorflow as tf

from speech_recognition.models.deepspeech2 import Convolution, DeepSpeech2, Recurrent


@pytest.mark.parametrize(
    "num_layers,channels,filter_sizes,strides,batch_size,sequence_length,frequency_bins,feature_dim",
    [
        (1, [32], [[41, 11]], [[2, 2]], 7, 111, 33, 1),
        (2, [32, 32], [[41, 11], [21, 11]], [[2, 2], [2, 1]], 12, 333, 45, 2),
        (3, [32, 32, 32], [[41, 11], [21, 11], [21, 11]], [[2, 2], [2, 1], [2, 1]], 33, 242, 56, 3),
        (3, [32, 32, 96], [[41, 11], [21, 11], [21, 11]], [[2, 2], [2, 1], [2, 1]], 5, 553, 62, 4),
    ],
)
def test_convolution(
    num_layers, channels, filter_sizes, strides, batch_size, sequence_length, frequency_bins, feature_dim
):
    convolution = Convolution(num_layers, channels, filter_sizes, strides)

    audio = tf.random.normal([batch_size, sequence_length, frequency_bins, feature_dim])
    output, mask = convolution(audio)

    output_batch_size, output_length, hidden_dim = output.shape
    assert batch_size == output_batch_size
    assert sequence_length > output_length == mask.shape[1]
    assert hidden_dim > channels[-1]


@pytest.mark.parametrize(
    "run_type,num_layers,units,recurrent_dropout,batch_size,sequence_length,feature_dim,pad_length",
    [
        ("rnn", 1, 240, 0.1, 88, 12, 142, 3),
        ("lstm", 3, 188, 0.2, 32, 121, 134, 4),
        ("gru", 5, 151, 0.3, 12, 124, 64, 5),
        ("gru", 7, 128, 0.4, 55, 333, 55, 6),
    ],
)
def test_recurrent(
    run_type, num_layers, units, recurrent_dropout, batch_size, sequence_length, feature_dim, pad_length
):
    recurrent = Recurrent(run_type, num_layers, units, recurrent_dropout)

    # Check Shape
    audio = tf.random.normal([batch_size, sequence_length, feature_dim])
    mask = tf.cast(tf.random.normal([batch_size, sequence_length]) > 0.1, tf.int32)
    output = recurrent(audio, mask)
    tf.debugging.assert_equal(output.shape, [batch_size, sequence_length, units * 2])

    padded_audio = tf.concat([audio, tf.random.normal([batch_size, pad_length, feature_dim])], axis=1)
    padded_mask = tf.concat([mask, tf.zeros([batch_size, pad_length], dtype=tf.int32)], axis=1)
    padded_output = recurrent(padded_audio, padded_mask)
    tf.debugging.assert_equal(padded_output.shape, [batch_size, sequence_length + pad_length, units * 2])

    # Check Mask for PAD
    tf.debugging.assert_equal(output, padded_output[:, :-pad_length])


# fmt: off
@pytest.mark.parametrize(
    "num_conv_layers,channels,filter_sizes,strides,rnn_type,num_reccurent_layers,hidden_dim,dropout,vocab_size,batch_size,sequence_length,freq_bins,feature_dim",
    [
        (1, [32], [[41, 11]], [[2, 2]], "rnn", 1, 240, 0.1, 88,7, 111, 33, 1),
        (2, [32, 32], [[41, 11], [21, 11]], [[2, 2], [2, 1]], "lstm", 3, 188, 0.2, 32,12, 333, 45, 2),
        (3, [32, 32, 32], [[41, 11], [21, 11], [21, 11]], [[2, 2], [2, 1], [2, 1]], "gru", 5, 151, 0.3, 12,33, 242, 56, 3),
        (3, [32, 32, 96], [[41, 11], [21, 11], [21, 11]], [[2, 2], [2, 1], [2, 1]], "gru", 7, 128, 0.4, 55,5, 553, 62, 4),
    ],
)
# fmt: on
def test_deepspeech2(
    num_conv_layers,
    channels,
    filter_sizes,
    strides,
    rnn_type,
    num_reccurent_layers,
    hidden_dim,
    dropout,
    vocab_size,
    batch_size,
    sequence_length,
    freq_bins,
    feature_dim,
):
    deepspeech2 = DeepSpeech2(
        num_conv_layers,
        channels,
        filter_sizes,
        strides,
        rnn_type,
        num_reccurent_layers,
        hidden_dim,
        dropout,
        dropout,
        vocab_size,
        10,
    )
    audio = tf.random.normal([batch_size, sequence_length, freq_bins, feature_dim])
    output = deepspeech2(audio)

    output_batch_size, output_length, output_vocab_size = output.shape
    assert batch_size == output_batch_size
    assert sequence_length > output_length
    assert output_vocab_size == vocab_size
