import os

import pytest
import tensorflow as tf

from speech_recognition.data import get_dataset, make_spectogram

DATA_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(DATA_DIR, "data", "dataset.tsv")


@pytest.fixture(scope="module")
def tokenizer():
    class PseudoTokenizer:
        def tokenize(self, sentence):
            return tf.strings.unicode_decode(sentence, "UTF8")

    return PseudoTokenizer()


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return get_dataset(DATASET_PATH, tokenizer)


def test_get_dataset(dataset):
    data = list(dataset)
    audio_sample, token_sample = data[0]

    assert len(list(dataset)) == 2
    assert len(data[0]) == 2
    tf.debugging.assert_equal(tf.shape(audio_sample), [66150, 1])
    tf.debugging.assert_equal(tf.shape(token_sample), [22])


@pytest.mark.parametrize(
    "frame_length,frame_step,fft_length",
    [
        (1024, 1024, 1024),
        (128, 64, 256),
        (128, 80, None),
        (512, 512, 256),
    ],
)
def test_make_spectogram(dataset, frame_length, frame_step, fft_length):
    audio_timestep = tf.shape(next(iter(dataset))[0])[0]
    dataset = dataset.map(lambda audio, _: make_spectogram(audio, frame_length, frame_step, fft_length))
    audio_sample = next(iter(dataset))

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, 1, fft_length // 2 + 1]
    )
