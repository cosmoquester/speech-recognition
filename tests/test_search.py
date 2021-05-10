import tensorflow as tf

from speech_recognition.model import LAS
from speech_recognition.search import Searcher


def test_search():
    model = LAS(
        vocab_size=100,
        hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    batch_size = 8
    encoder_sequence = 10
    encoder_input_dim = 123
    decoder_sequence = 15
    bos_id = 2
    eos_id = 3
    max_sequence_length = 17
    output = model(
        (
            tf.random.uniform((batch_size, encoder_sequence, encoder_input_dim), maxval=100, dtype=tf.float32),
            tf.random.uniform((batch_size, decoder_sequence), maxval=100, dtype=tf.int32),
        )
    )

    encoder_input = tf.random.uniform((batch_size, encoder_sequence, encoder_input_dim), maxval=100, dtype=tf.float32)
    decoder_sequence = tf.random.uniform((batch_size, decoder_sequence), maxval=100, dtype=tf.int32)

    searcher = Searcher(model, max_sequence_length, bos_id, eos_id)
    beam_result, beam_ppl = searcher.beam_search(encoder_input, 1)
    greedy_result, greedy_ppl = searcher.greedy_search(encoder_input)

    tf.debugging.assert_equal(beam_result[:, 0, :], greedy_result)
    tf.debugging.assert_near(tf.squeeze(beam_ppl), greedy_ppl)
