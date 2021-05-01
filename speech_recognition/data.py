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
    data_dir_path = os.path.dirname(dataset_file_path)
    data_dir_path += os.sep if not data_dir_path.startswith("gs://") else "/"

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
def make_spectogram(audio: tf.Tensor, frame_length: int, frame_step: int, fft_length=None) -> tf.Tensor:
    """
    Make spectogram from PCM audio dataset.

    :param audio: pcm format audio tensor shaped [TimeStep, NumChannel]
    :param frame_length: window length in samples
    :param frame_step: number of samples to step
    :param fft_length: size of the FFT to apply. By default, uses the smallest power of 2 enclosing frame_length
    :return: spectogram audio tensor shaped [NumFrame, NumChannel, NumFFTUniqueBins]
    """
    # Shape: [NumChannel, TimeStep]
    audio = tf.transpose(audio)
    # Shape: [NumChannel, NumFrame, NumFFTUniqueBins]
    spectogram = tf.signal.stft(audio, frame_length, frame_step, fft_length)
    spectogram = tf.abs(spectogram)

    # Shape: [NumFrame, NumChannel, NumFFTUniqueBins]
    spectogram = tf.transpose(spectogram, [1, 0, 2])
    return spectogram
