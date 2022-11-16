import os

import pytest
import tensorflow as tf

from speech_recognition.data import (
    get_dataset,
    get_tfrecord_dataset,
    make_log_mel_spectrogram,
    make_mfcc,
    make_spectrogram,
    spec_augment,
)

from .const import MP3_DATASET_PATH, PCM_DATASET_PATH, TFRECORD_DATASET_PATH, WAV_DATASET_PATH


class PseudoTokenizer:
    @staticmethod
    def tokenize(sentence):
        return tf.strings.unicode_decode(sentence, "UTF8")


wav_dataset = get_dataset(WAV_DATASET_PATH, "wav", 22050, PseudoTokenizer, False)
pcm_dataset = get_dataset(PCM_DATASET_PATH, "pcm", 22050, PseudoTokenizer, False)
mp3_dataset = get_dataset(MP3_DATASET_PATH, "mp3", 22050, PseudoTokenizer, False)
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
    tf.debugging.assert_equal(tf.shape(log_mel_spectrogram_sample), [412, 80, 1])
    tf.debugging.assert_equal(tf.shape(token_sample), [22])


def test_make_tfrecord_dataset():
    dataset = wav_dataset.map(make_log_mel_spectrogram(16000, 320, 160, 320, 80, 80.0, 7600.0))
    for data, tf_data in zip(dataset, tfrecord_dataset):
        tf.debugging.assert_equal(data[0], tf_data[0])
        tf.debugging.assert_equal(data[1], tf_data[1])


@pytest.mark.parametrize(
    "frame_length,frame_step,fft_length", [(1024, 1024, 1024), (128, 64, 256), (128, 80, None), (512, 512, 256),],
)
def test_make_spectrogram(frame_length, frame_step, fft_length):
    audio_timestep = tf.shape(next(iter(wav_dataset))[0])[0]
    dataset = wav_dataset.map(make_spectrogram(frame_length, frame_step, fft_length))
    audio_sample = next(iter(dataset))[0]

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, fft_length // 2 + 1, 1]
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
        make_log_mel_spectrogram(
            sample_rate, frame_length, frame_step, fft_length, num_mel_bins, lower_edge_hertz, upper_edge_hertz,
        )
    )
    audio_sample = next(iter(dataset))[0]

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, num_mel_bins, 1]
    )


@pytest.mark.parametrize(
    "sample_rate,frame_length,frame_step,fft_length,num_mel_bins,num_mfcc,lower_edge_hertz,upper_edge_hertz",
    [
        (22050, 1024, 1024, 1024, 80, 40, 10, 10000),
        (16000, 128, 64, 256, 123, 33, 12, 88),
        (32000, 128, 80, 128, 321, 100, 32, 16000),
        (44100, 512, 512, 256, 333, 333, 333, 3333),
    ],
)
def test_make_mfcc(
    sample_rate, frame_length, frame_step, fft_length, num_mel_bins, num_mfcc, lower_edge_hertz, upper_edge_hertz
):
    audio_timestep = tf.shape(next(iter(wav_dataset))[0])[0]
    dataset = wav_dataset.map(
        make_mfcc(
            sample_rate,
            frame_length,
            frame_step,
            fft_length,
            num_mel_bins,
            num_mfcc,
            lower_edge_hertz,
            upper_edge_hertz,
        )
    )
    audio_sample = next(iter(dataset))[0]

    if fft_length is None:
        fft_length = frame_length
    tf.debugging.assert_equal(
        tf.shape(audio_sample), [(audio_timestep - frame_length + frame_step) // frame_step, num_mfcc, 1]
    )


@pytest.mark.parametrize("W,F,m_F,T,p,m_T", [(80, 27, 1, 100, 1.0, 1), (40, 15, 2, 70, 0.2, 2)])
def test_spec_augment(W, F, m_F, T, p, m_T):
    num_time = 234
    num_frequency = 80

    spec_augment_fn = spec_augment(num_frequency, W, F, m_F, T, p, m_T)
    data = tf.random.uniform([num_time, num_frequency, 1], 0.1, 1.0)
    augmented = spec_augment_fn(data)
    is_zero = tf.reduce_all(augmented == 0.0, axis=2)
    all_zero_freq = tf.math.count_nonzero(tf.reduce_all(is_zero, axis=0))
    all_zero_time = tf.math.count_nonzero(tf.reduce_all(is_zero, axis=1))

    tf.debugging.assert_less_equal(all_zero_freq, tf.cast(F * m_F, tf.int64))
    tf.debugging.assert_less_equal(all_zero_time, tf.cast(T * m_T, tf.int64))
    tf.debugging.assert_equal(data.shape, augmented.shape)
    tf.debugging.assert_equal(tf.math.reduce_any(data != augmented), True)
