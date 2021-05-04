import argparse
import glob
import os

import tensorflow as tf
import tensorflow_text as text
from omegaconf import OmegaConf
from tqdm import tqdm

from speech_recognition.data import get_dataset, make_log_mel_spectrogram
from speech_recognition.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, required=True, help="config file for processing dataset")
parser.add_argument("--dataset-paths", type=str, required=True, help="dataset file path glob pattern")
parser.add_argument("--output-dir", type=str, help="output directory path, default is input dataset file directoruy")
parser.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model", help="sentencepiece model path")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()
    logger = get_logger("make-tfrecord")

    input_files = glob.glob(args.dataset_paths)
    logger.info(f"[+] Number of Dataset Files: {len(input_files)}")

    # Load Config
    logger.info(f"[+] Load Config From {args.config_path}")
    with tf.io.gfile.GFile(args.config_path) as f:
        config = OmegaConf.load(f)

    # Load Sentencepiece model
    logger.info(f"[+] Load Tokenizer From {args.sp_model_path}")
    with open(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    shape_squeeze = lambda x: tf.reshape(x, [tf.shape(x)[0], -1])
    map_log_mel_spectrogram = tf.function(
        lambda audio, text: (
            shape_squeeze(
                make_log_mel_spectrogram(
                    audio,
                    config.sample_rate,
                    config.frame_length,
                    config.frame_step,
                    config.fft_length,
                    config.num_mel_bins,
                    config.lower_edge_hertz,
                    config.upper_edge_hertz,
                )
            ),
            text,
        )
    )
    serialize = tf.function(
        lambda audio, text: tf.io.serialize_tensor(
            tf.stack(
                [
                    tf.io.serialize_tensor(audio),
                    tf.io.serialize_tensor(text),
                ]
            )
        )
    )

    logger.info(f"[+] Start Saving Dataset...")
    for file_path in tqdm(input_files):
        output_dir = args.output_dir if args.output_dir else os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".tfrecord")

        # Write TFRecordFile
        dataset = (
            get_dataset(file_path, tokenizer, config.desired_channels)
            .map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(serialize)
        )
        writer = tf.data.experimental.TFRecordWriter(output_path, "GZIP")
        writer.write(dataset)

    logger.info(f"[+] Done")