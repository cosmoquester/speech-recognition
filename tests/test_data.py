import os

import pytest
import tensorflow as tf

from speech_recognition.data import get_dataset, get_tfrecord_dataset, make_log_mel_spectrogram, make_spectrogram

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAV_DATASET_PATH = os.path.join(DATA_DIR, "wav_dataset.tsv")
PCM_DATASET_PATH = os.path.join(DATA_DIR, "pcm_dataset.tsv")
MP3_DATASET_PATH = os.path.join(DATA_DIR, "mp3_dataset.tsv")
TFRECORD_DATASET_PATH = os.path.join(DATA_DIR, "dataset.tfrecord")


class PseudoTokenizer:
    @staticmethod
    def tokenize(sentence):
        return tf.strings.unicode_decode(sentence, "UTF8")


wav_dataset = get_dataset(WAV_DATASET_PATH, "wav", 22050, PseudoTokenizer)
pcm_dataset = get_dataset(PCM_DATASET_PATH, "pcm", 22050, PseudoTokenizer)
mp3_dataset = get_dataset(MP3_DATASET_PATH, "mp3", 22050, PseudoTokenizer)
tfrecord_dataset = get_tfrecord_dataset(TFRECORD_DATASET_PATH)


def test_get_dataset():
    data = list(wav_dataset)
    audio_sample, token_sample = data[0]
    pcm_audio_sample = next(iter(pcm_dataset))[0]
    mp3_audio_sample = next(iter(mp3_dataset))[0]

    assert len(data) == 2
    assert len(data[0]) == 2
    tf.debugging.assert_equal(tf.shape(audio_sample), [66150])
    tf.debugging.assert_equal(tf.shape(token_sample), [22])
    tf.debugging.assert_equal(data[0][0], data[1][0], pcm_audio_sample, mp3_audio_sample)


def test_get_tfrecord_dataset():
    data = list(tfrecord_dataset)
    log_mel_spectrogram_sample, token_sample = data[0]

    assert len(data) == 2
    assert len(data[0]) == 2
    tf.debugging.assert_equal(tf.shape(log_mel_spectrogram_sample), [128, 80])
    tf.debugging.assert_equal(tf.shape(token_sample), [22])


def test_make_tfrecord_dataset():
    dataset = wav_dataset.map(
        lambda audio, text: (make_log_mel_spectrogram(audio, 22050, 1024, 512, 1024, 80, 80.0, 7600.0), text)
    )
    for data, tf_data in zip(dataset, tfrecord_dataset):
        tf.debugging.assert_equal(data[0], tf_data[0])
        tf.debugging.assert_equal(data[1], tf_data[1])


@pytest.mark.parametrize(
    "frame_length,frame_step,fft_length",
    [
        (1024, 1024, 1024),
        (128, 64, 256),
        (128, 80, None),
        (512, 512, 256),
    ],
)
def test_make_spectrogram(frame_length, frame_step, fft_length):
    audio_timestep = tf.shape(next(iter(wav_dataset))[0])[0]
    dataset = wav_dataset.map(lambda audio, _: make_spectrogram(audio, frame_length, frame_step, fft_length))
    audio_sample = next(iter(dataset))

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, fft_length // 2 + 1]
    )


@pytest.mark.parametrize(
    "sample_rate,frame_length,frame_step,fft_length,num_mel_bins,lower_edge_hertz,upper_edge_hertz",
    [
        (22050, 1024, 1024, 1024, 80, 10, 10000),
        (16000, 128, 64, 256, 123, 12, 88),
        (32000, 128, 80, 128, 321, 32, 16000),
        (44100, 512, 512, 256, 333, 333, 3333),
    ],
)
def test_make_log_mel_spectrogram(
    sample_rate, frame_length, frame_step, fft_length, num_mel_bins, lower_edge_hertz, upper_edge_hertz
):
    audio_timestep = tf.shape(next(iter(wav_dataset))[0])[0]
    dataset = wav_dataset.map(
        lambda audio, _: make_log_mel_spectrogram(
            audio, sample_rate, frame_length, frame_step, fft_length, num_mel_bins, lower_edge_hertz, upper_edge_hertz
        )
    )
    audio_sample = next(iter(dataset))

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, num_mel_bins]
    )
