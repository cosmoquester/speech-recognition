from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_text as text


def get_dataset(
    dataset_file_path: str,
    tokenizer: text.SentencepieceTokenizer,
    resample: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Load dataset from tsv file. The dataset file has to header.
    The first column has audio file path and second column has recognized sentence.

    :param dataset_file_path: dataset file path
    :param tokenizer: sentencepiece tokenizer
    :param resample: resample rate (default no resample)
    :return: tensorflow dataset
    """
    dataset = tf.data.experimental.CsvDataset(
        dataset_file_path, [tf.string, tf.string], header=True, field_delim="\t", use_quote_delim=False
    )

    def _load_example(audio_file_path: tf.Tensor, sentence: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load audio file and tokenize sentence.

        :param audio_file_path: string tensor that is audio file path
        :param sentence: string tensor that is recognized sentence from audio
        :return: audio and sentence tensor
        """
        # audio: [TimeStep, NumChannel]
        audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_file_path))
        # tokens: [NumTokens]
        tokens = tokenizer.tokenize(sentence)

        # Resample
        if resample is not None:
            audio = tfio.audio.resample(audio, sample_rate, resample, name="resampling")
        return audio, tokens

    return dataset.map(_load_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
