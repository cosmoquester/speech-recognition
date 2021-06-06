import argparse
import glob
import os
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml
from tqdm import tqdm

from ..configs import DataConfig
from ..data import get_dataset, make_log_mel_spectrogram
from ..utils import get_logger

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--data-config", type=str, required=True, help="data processing config file")
parser.add_argument("--dataset-paths", type=str, required=True, help="dataset file path glob pattern")
parser.add_argument("--output-dir", type=str, help="output directory path, default is input dataset file directoruy")
parser.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model", help="sentencepiece model path")
# fmt: on


def main(args: argparse.Namespace):
    logger = get_logger("make-tfrecord")

    input_files = glob.glob(args.dataset_paths)
    logger.info(f"[+] Number of Dataset Files: {len(input_files)}")

    # Load Config
    logger.info(f"[+] Load Config From {args.data_config}")
    config = DataConfig.from_yaml(args.data_config)

    # Load Sentencepiece model
    logger.info(f"[+] Load Tokenizer From {args.sp_model_path}")
    with open(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    serialize = tf.function(
        lambda audio, text: tf.io.serialize_tensor(
            tf.stack([tf.io.serialize_tensor(audio), tf.io.serialize_tensor(text)])
        )
    )

    logger.info("[+] Start Saving Dataset...")
    for file_path in tqdm(input_files):
        output_dir = args.output_dir if args.output_dir else os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".tfrecord")

        # Write TFRecordFile
        dataset = (
            get_dataset(file_path, config.file_format, config.sample_rate, tokenizer)
            .map(config.audio_feature_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(serialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )
        writer = tf.data.experimental.TFRecordWriter(output_path, "GZIP")
        writer.write(dataset)

    logger.info("[+] Done")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
