import os
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_text as text


def get_dataset(
    dataset_file_path: str,
    tokenizer: text.SentencepieceTokenizer,
    desired_channels: int = -1,
    resample: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Load dataset from tsv file. The dataset file has to header.
    The first column has audio file path and second column has recognized sentence.

    :param dataset_file_path: dataset file path
    :param tokenizer: sentencepiece tokenizer
    :param desired_channels: number of sample channels wanted. default is auto
    :param resample: resample rate (default no resample)
    :return: PCM audio and tokenized sentence dataset
    """
    dataset = tf.data.experimental.CsvDataset(
        dataset_file_path, [tf.string, tf.string], header=True, field_delim="\t", use_quote_delim=False
    )
    if dataset_file_path.startswith("gs://"):
        data_dir_path = os.path.dirname(dataset_file_path) + "/"
    else:
        dataset_file_path = os.path.abspath(dataset_file_path)
        data_dir_path = os.path.dirname(dataset_file_path) + os.sep

    @tf.function
    def _load_example(audio_file_path: tf.Tensor, sentence: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load audio file and tokenize sentence.

        :param audio_file_path: string tensor that is audio file path
        :param sentence: string tensor that is recognized sentence from audio
        :return: audio and sentence tensor
        """
        # audio: [TimeStep, NumChannel]
        audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(data_dir_path + audio_file_path), desired_channels)
        # tokens: [NumTokens]
        tokens = tokenizer.tokenize(sentence)

        # Resample
        if resample is not None:
            audio = tfio.audio.resample(audio, sample_rate, resample, name="resampling")
        return audio, tokens

    return dataset.map(_load_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@tf.function
def make_spectrogram(audio: tf.Tensor, frame_length: int, frame_step: int, fft_length=None) -> tf.Tensor:
    """
    Make spectrogram from PCM audio dataset.

    :param audio: pcm format audio tensor shaped [TimeStep, NumChannel]
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :return: spectrogram audio tensor shaped [NumFrame, NumChannel, NumFFTUniqueBins]
    """
    # Shape: [NumChannel, TimeStep]
    audio = tf.transpose(audio)
    # Shape: [NumChannel, NumFrame, NumFFTUniqueBins]
    spectrogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    spectrogram = tf.abs(spectrogram)

    # Shape: [NumFrame, NumChannel, NumFFTUniqueBins]
    spectrogram = tf.transpose(spectrogram, [1, 0, 2])
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

    :param audio: pcm format audio tensor shaped [TimeStep, NumChannel]
    :param sample_rate: sampling rate of audio
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :param num_mel_bins: how many bands in the resulting mel spectrum
    :param lower_edge_hertz: lower bound on the frequencies to be included in the mel spectrum
    :param upper_edge_hertz: desired top edge of the highest frequency band
    :param epsilon: added to mel spectrogram before log to prevent nan calculation
    """
    # Shape: [NumFrame, NumChannel, NumFFTUniqueBins]
    spectrogram = make_spectrogram(audio, frame_length, frame_step, fft_length)

    num_spectrogram_bins = fft_length // 2 + 1
    # Shape: [NumFFTUniqueBins, NumMelFilterbank]
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
    )

    # Sahpe: [NumFrame, NumChannel, NumMelFilterbank]
    mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_filterbank)
    log_mel_sepctrogram = tf.math.log(mel_spectrogram + epsilon)
    return log_mel_sepctrogram


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
