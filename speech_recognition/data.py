import os
from functools import partial
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_text as text


def get_dataset(
    dataset_paths: str,
    file_format: str,
    sample_rate: int,
    tokenizer: text.SentencepieceTokenizer,
    resample: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Load dataset from tsv file. The dataset file has to header.
    The first column has audio file path and second column has recognized sentence.

    :param dataset_paths: dataset file path glob pattern. all dataset files is in same directory.
    :param file_format: audio file format. one of ["wav", "flac", "mp3", "pcm"]
    :param sample_rate: audio sample rate
    :param tokenizer: sentencepiece tokenizer
    :param resample: resample rate (default no resample)
    :return: PCM audio and tokenized sentence dataset
    """
    dataset_list = tf.io.gfile.glob(dataset_paths)
    dataset = tf.data.experimental.CsvDataset(
        dataset_list, [tf.string, tf.string], header=True, field_delim="\t", use_quote_delim=False
    )
    if dataset_list[0].startswith("gs://"):
        data_dir_path = os.path.dirname(dataset_list[0]) + "/"
    else:
        dataset_file_path = os.path.abspath(dataset_list[0])
        data_dir_path = os.path.dirname(dataset_file_path) + os.sep

    load_example = tf.function(
        lambda file_path, text: (
            load_audio_file(data_dir_path + file_path, sample_rate, file_format, resample),
            tokenizer.tokenize(text),
        )
    )

    return dataset.map(load_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)


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


@tf.function
def load_audio_file(
    audio_file_path: tf.Tensor, sample_rate: int, file_format: str, resample: Optional[float] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load audio file and tokenize sentence.

    :param audio_file_path: string tensor that is audio file path
    :param sample_rate: sample rate of audio
    :param file_format: audio format, one of [flac, wav, pcm, mp3]
    :param resample: resample rate if it needs, else None
    :return: loaded audio tensor shaped [TimeStep]
    """
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


@tf.function
def make_spectrogram(audio: tf.Tensor, frame_length: int, frame_step: int, fft_length=None) -> tf.Tensor:
    """
    Make spectrogram from PCM audio dataset.

    :param audio: pcm format audio tensor shaped [TimeStep]
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :return: spectrogram audio tensor shaped [NumFrame, NumFFTUniqueBins]
    """
    # Shape: [NumFrame, NumFFTUniqueBins]
    spectrogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


@tf.function
def make_log_mel_spectrogram(
    audio: tf.Tensor,
    sample_rate: int,
    frame_length: int,
    frame_step: int,
    fft_length: int,
    num_mel_bins: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    epsilon: float = 1e-12,
) -> tf.Tensor:
    """
    Make log mel spectrogram from PCM audio dataset.

    :param audio: pcm format audio tensor shaped [TimeStep]
    :param sample_rate: sampling rate of audio
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :param num_mel_bins: how many bands in the resulting mel spectrum
    :param lower_edge_hertz: lower bound on the frequencies to be included in the mel spectrum
    :param upper_edge_hertz: desired top edge of the highest frequency band
    :param epsilon: added to mel spectrogram before log to prevent nan calculation
    :return: log mel sectrogram of audio
    """
    # Shape: [NumFrame, NumFFTUniqueBins]
    spectrogram = make_spectrogram(audio, frame_length, frame_step, fft_length)

    num_spectrogram_bins = fft_length // 2 + 1
    # Shape: [NumFFTUniqueBins, NumMelFilterbank]
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
    )

    # Sahpe: [NumFrame, NumMelFilterbank]
    mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_filterbank)
    log_mel_sepctrogram = tf.math.log(mel_spectrogram + epsilon)
    return log_mel_sepctrogram


@tf.function
def delta_accelerate(audio: tf.Tensor):
    """
    Append delta and deltas from audio feature.

    :param: audio: audio data shaped [TimeStep, AudioDim]
    :return: enhanced audio feature shaped [TimeStep, AudioDim, 3]
    """
    zero_head = tf.zeros_like(audio[:1])
    delta = audio - tf.concat([zero_head, audio[:-1]], axis=0)
    deltas = delta - tf.concat([zero_head, delta[:-1]], axis=0)

    # [TimeStep, AudioDim, 3]
    audio = tf.stack([audio, delta, deltas], axis=2)
    return audio


@tf.function
def make_train_examples(source_tokens: tf.Tensor, target_tokens: tf.Tensor):
    """Make training examples from source and target tokens."""
    # Make training example
    num_examples = tf.shape(target_tokens)[0] - 1

    # [NumExamples, EncoderSequence]
    encoder_input = tf.repeat([source_tokens], repeats=[num_examples], axis=0)
    # [NumExamples, DecoderSequence]
    decoder_input = target_tokens * tf.sequence_mask(tf.range(1, num_examples + 1), num_examples + 1, tf.int32)
    # [NumExamples]
    labels = target_tokens[1:]

    return (encoder_input, decoder_input), labels
