import os
import random
from functools import partial
from typing import Callable, Optional, Tuple

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_text as text


def get_dataset(
    dataset_paths: str,
    file_format: str,
    sample_rate: int,
    tokenizer: text.SentencepieceTokenizer,
    shuffle: bool = False,
    resample: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Load dataset from tsv file. The dataset file has to header.
    The first column has audio file path and second column has recognized sentence.

    :param dataset_paths: dataset file path glob pattern.
    :param file_format: audio file format. one of ["wav", "flac", "mp3", "pcm"]
    :param sample_rate: audio sample rate
    :param tokenizer: sentencepiece tokenizer
    :param shuffle: whether shuffle files or not
    :param resample: resample rate (default no resample)
    :return: PCM audio and tokenized sentence dataset
    """
    dataset_list = tf.io.gfile.glob(dataset_paths)
    if shuffle:
        random.shuffle(dataset_list)

    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)

    def _get_data_dir_path(dataset_path):
        dataset_path = dataset_path.numpy().decode("utf-8")

        if dataset_path.startswith("gs://"):
            data_dir_path = os.path.dirname(dataset_path) + "/"
        else:
            dataset_file_path = os.path.abspath(dataset_path)
            data_dir_path = os.path.dirname(dataset_file_path) + os.sep
        return data_dir_path

    def _to_dataset(dataset_path):
        data_dir_path = tf.py_function(_get_data_dir_path, [dataset_path], [tf.string])[0]
        load_audio_file_fn = load_audio_file(sample_rate, file_format, resample)

        load_example = tf.function(
            lambda file_path, text: (load_audio_file_fn(data_dir_path + file_path), tokenizer.tokenize(text))
        )
        dataset = tf.data.experimental.CsvDataset(
            dataset_path, [tf.string, tf.string], header=True, field_delim="\t", use_quote_delim=False
        ).map(load_example)

        return dataset

    return dataset.interleave(_to_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_tfrecord_dataset(dataset_paths: str) -> tf.data.Dataset:
    """Read TFRecord dataset file and construct tensorflow dataset"""

    dataset_list = tf.io.gfile.glob(dataset_paths)
    decompose = tf.function(
        lambda serialized_example: (
            tf.io.parse_tensor(serialized_example[0], tf.float32),
            tf.io.parse_tensor(serialized_example[1], tf.int32),
        )
    )
    dataset = (
        tf.data.TFRecordDataset(dataset_list, "GZIP")
        .map(partial(tf.io.parse_tensor, out_type=tf.string))
        .map(decompose)
    )
    return dataset


def load_audio_file(
    sample_rate: int, file_format: str, resample: Optional[float] = None
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Make function wrapper to load audio file and tokenize sentence.

    :param sample_rate: sample rate of audio
    :param file_format: audio format, one of [flac, wav, pcm, mp3]
    :param resample: resample rate if it needs, else None
    :return: loaded audio tensor shaped [TimeStep]
    """

    @tf.function
    def _wrapper(audio_file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # audio: [TimeStep, NumChannel]
        if file_format in ["flac", "wav"]:
            audio_io_tensor = tfio.audio.AudioIOTensor(audio_file_path, tf.int16)
            audio = tf.cast(audio_io_tensor.to_tensor(), tf.float32) / 32768.0
        elif file_format == "pcm":
            audio_binary = tf.io.read_file(audio_file_path)
            if tf.strings.length(audio_binary) % 2 == 1:
                audio_binary += "\x00"
            audio_int_tensor = tf.io.decode_raw(audio_binary, tf.int16)
            audio = tf.cast(audio_int_tensor, tf.float32)[:, tf.newaxis] / 32768.0
        elif file_format == "mp3":
            audio = tfio.audio.AudioIOTensor(audio_file_path, tf.float32).to_tensor()
        else:
            raise ValueError(f"File Format: {file_format} is not valid!")

        # Resample
        if resample is not None:
            audio = tfio.audio.resample(audio, sample_rate, resample, name="resampling")

        # Reduce Channel
        audio = tf.reduce_mean(audio, 1)
        return audio

    return _wrapper


def make_spectrogram(frame_length: int, frame_step: int, fft_length=None):
    """
    Make function wrapper to convert to spectrogram from PCM audio dataset.

    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :return: spectrogram audio tensor shaped [NumFrame, NumFFTUniqueBins, 1]
    """

    @tf.function
    def _wrapper(audio: tf.Tensor, text: Optional[tf.Tensor] = None):
        # Shape: [NumFrame, NumFFTUniqueBins]
        spectrogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
        spectrogram = tf.abs(spectrogram)[:, :, tf.newaxis]

        if text is None:
            return spectrogram
        return spectrogram, text

    return _wrapper


def make_log_mel_spectrogram(
    sample_rate: int,
    frame_length: int,
    frame_step: int,
    fft_length: int,
    num_mel_bins: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    epsilon: float = 1e-12,
):
    """
    Make function wrapper to convert to log mel spectrogram from PCM audio dataset.

    :param sample_rate: sampling rate of audio
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :param num_mel_bins: how many bands in the resulting mel spectrum
    :param lower_edge_hertz: lower bound on the frequencies to be included in the mel spectrum
    :param upper_edge_hertz: desired top edge of the highest frequency band
    :param epsilon: added to mel spectrogram before log to prevent nan calculation
    :return: log mel sectrogram of audio tensor shaped [NumFrame, NumMelFilterbank, 1]
    """

    @tf.function
    def _wrapper(audio: tf.Tensor, text: Optional[tf.Tensor] = None):
        # Shape: [NumFrame, NumFFTUniqueBins]
        spectrogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
        spectrogram = tf.abs(spectrogram)

        num_spectrogram_bins = fft_length // 2 + 1
        # Shape: [NumFFTUniqueBins, NumMelFilterbank]
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
        )

        # Sahpe: [NumFrame, NumMelFilterbank]
        mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_filterbank)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + epsilon)[:, :, tf.newaxis]

        if text is None:
            return log_mel_spectrogram
        return log_mel_spectrogram, text

    return _wrapper


def make_mfcc(
    sample_rate: int,
    frame_length: int,
    frame_step: int,
    fft_length: int,
    num_mel_bins: int = 80,
    num_mfcc: int = 40,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    epsilon: float = 1e-12,
):
    """
    Make function wrapper to convert to dct-2 type mfcc from PCM audio dataset.

    :param sample_rate: sampling rate of audio
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :param num_mel_bins: how many bands in the resulting mel spectrum
    :param num_mfcc: the number of mfcc
    :param lower_edge_hertz: lower bound on the frequencies to be included in the mel spectrum
    :param upper_edge_hertz: desired top edge of the highest frequency band
    :param epsilon: added to mel spectrogram before log to prevent nan calculation
    :return: log mel sectrogram of audio tensor shaped [NumFrame, NumMelFilterbank, 1]
    """

    @tf.function
    def _wrapper(audio: tf.Tensor, text: Optional[tf.Tensor] = None):
        # Shape: [NumFrame, NumFFTUniqueBins]
        spectrogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
        spectrogram = tf.abs(spectrogram)

        num_spectrogram_bins = fft_length // 2 + 1
        # Shape: [NumFFTUniqueBins, NumMelFilterbank]
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
        )

        # Sahpe: [NumFrame, NumMelFilterbank]
        mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_filterbank)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + epsilon)

        # Sahpe: [NumFrame, NumMFCC]
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :num_mfcc, tf.newaxis]

        if text is None:
            return mfcc
        return mfcc, text

    return _wrapper


@tf.function
def delta_accelerate(audio: tf.Tensor, text: Optional[tf.Tensor] = None):
    """
    Append delta and deltas from audio feature.

    :param audio: audio data shaped [TimeStep, AudioDim, 1]
    :param text: text data
    :return: enhanced audio feature shaped [TimeStep, AudioDim, 3]
    """
    zero_head = tf.zeros_like(audio[:1])
    delta = audio - tf.concat([zero_head, audio[:-1]], axis=0)
    deltas = delta - tf.concat([zero_head, delta[:-1]], axis=0)

    # [TimeStep, AudioDim, 3]
    audio = tf.concat([audio, delta, deltas], axis=2)

    if text is None:
        return audio
    return audio, text


def filter_example(max_audio_length, max_token_length):
    """Filter examples whose sequence length is over than the max"""

    def _wrapper(dataset):
        @tf.function
        def filter_fn(audio, text):
            return tf.math.logical_and(tf.shape(audio)[0] <= max_audio_length, tf.size(text) <= max_token_length)

        return dataset.filter(filter_fn)

    return _wrapper


def slice_example(max_audio_length, max_token_length):
    """Slice examples whose sequence length is over than the max"""

    def _wrapper(dataset):
        @tf.function
        def slice_fn(audio, text):
            return audio[:max_audio_length], text[:max_token_length]

        return dataset.map(slice_fn)

    return _wrapper
