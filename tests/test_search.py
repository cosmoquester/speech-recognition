import tensorflow as tf

from speech_recognition.models import LAS, DeepSpeech2
from speech_recognition.search import DeepSpeechSearcher, LAS_Searcher


def test_deepspeech_search():
    blank_index = 33
    model = DeepSpeech2(1, [32], [[41, 11]], [[2, 2]], "lstm", 1, 240, 0.1, 0.1, 111, blank_index, 1)

    batch_size = 8
    encoder_sequence = 300
    encoder_input_dim = 123
    model((tf.keras.Input([None, encoder_input_dim, 3], dtype=tf.float32)))

    encoder_input = tf.random.uniform(
        (batch_size, encoder_sequence, encoder_input_dim, 3), maxval=100, dtype=tf.float32
    )

    searcher = DeepSpeechSearcher(model, blank_index)
    beam_result, beam_neg_sum_logits = searcher.beam_search(encoder_input, 1)
    greedy_result, greedy_neg_sum_logits = searcher.greedy_search(encoder_input)

    tf.debugging.assert_equal(beam_result[:, 0, :], greedy_result)
    tf.debugging.assert_near(beam_neg_sum_logits[:, 0], greedy_neg_sum_logits)


def test_las_search():
    model = LAS(
        rnn_type="lstm",
        vocab_size=100,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
        teacher_forcing_rate=0.99,
    )

    batch_size = 8
    encoder_sequence = 10
    encoder_input_dim = 123
    decoder_sequence = 15
    bos_id = 2
    eos_id = 3
    max_sequence_length = 17
    model((tf.keras.Input([None, encoder_input_dim, 3], dtype=tf.float32), tf.keras.Input([None], dtype=tf.int32),))

    encoder_input = tf.random.uniform(
        (batch_size, encoder_sequence, encoder_input_dim, 3), maxval=100, dtype=tf.float32
    )
    decoder_sequence = tf.random.uniform((batch_size, decoder_sequence), maxval=100, dtype=tf.int32)

    searcher = LAS_Searcher(model, max_sequence_length, bos_id, eos_id)
    beam_result, beam_ppl = searcher.beam_search(encoder_input, 1)
    greedy_result, greedy_ppl = searcher.greedy_search(encoder_input)

    tf.debugging.assert_equal(beam_result[:, 0, :], greedy_result)
    tf.debugging.assert_near(tf.squeeze(beam_ppl), greedy_ppl)
