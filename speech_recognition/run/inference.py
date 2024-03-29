import argparse
import csv
import sys
from functools import partial

import tensorflow as tf
import tensorflow_text as text
import yaml

from ..configs import DataConfig, get_model_config
from ..data import delta_accelerate, load_audio_file
from ..models import LAS, DeepSpeech2
from ..search import DeepSpeechSearcher, LAS_Searcher
from ..utils import get_device_strategy, get_logger

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
parser.add_argument("--data-config", type=str, required=True, help="data processing config file")
parser.add_argument("--model-config", type=str, required=True, help="model config file")
parser.add_argument("--audio-files", required=True, help="an audio file or glob pattern of multiple files ex) *.pcm")
parser.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
parser.add_argument("--output-path", default="output.tsv", help="output tsv file path to save generated sentences")
parser.add_argument("--sp-model-path", type=str, required=True, help="sentencepiece model path")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on


def main(args: argparse.Namespace):
    strategy = get_device_strategy(args.device)

    logger = get_logger("inference")

    if args.mixed_precision:
        mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
        policy = tf.keras.mixed_precision.Policy(mixed_type)
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("[+] Use Mixed Precision FP16")

    # Construct Dataset
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()

    dataset_files = sorted(tf.io.gfile.glob(args.audio_files))
    if not dataset_files:
        logger.error("[Error] Dataset path is invalid!")
        sys.exit(1)

    # Load Config
    logger.info(f"Load Data Config from {args.data_config}")
    config = DataConfig.from_yaml(args.data_config)

    with strategy.scope():
        dataset = (
            tf.data.Dataset.from_tensor_slices(dataset_files)
            .map(load_audio_file(config.sample_rate, config.file_format, config.sample_rate))
            .map(config.audio_feature_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

        # Delta Accelerate
        if config.use_delta_accelerate:
            logger.info("[+] Use delta and deltas accelerate")
            dataset = dataset.map(delta_accelerate)

        dataset = dataset.padded_batch(args.batch_size, [None, config.frequency_dim, config.feature_dim]).prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # Model Initialize & Load pretrained model
        model_config = get_model_config(args.model_config)
        model = model_config.create_model()

        model_input, _ = model.make_example(
            tf.keras.Input([None, config.frequency_dim, config.feature_dim], dtype=tf.float32),
            tf.keras.Input([None], dtype=tf.int32),
        )
        model(model_input)
        tf.train.Checkpoint(model).restore(args.model_path).expect_partial()
        logger.info(f"Loaded weights of model from {args.model_path}")
        model.summary()

        if isinstance(model, LAS):
            searcher = LAS_Searcher(model, config.max_token_length, bos_id, eos_id, model_config.pad_id)
        elif isinstance(model, DeepSpeech2):
            searcher = DeepSpeechSearcher(model, model_config.blank_index)

        # Inference
        logger.info("Start Inference")
        outputs = []

        for batch_input in dataset:
            if args.beam_size > 0:
                batch_output = searcher.beam_search(batch_input, args.beam_size)
                batch_output = batch_output[0][:, 0, :].numpy()
            else:
                batch_output = searcher.greedy_search(batch_input)[0].numpy()
            outputs.extend(batch_output)
        outputs = [tokenizer.detokenize(output).numpy().decode("UTF8") for output in outputs]
        logger.info("Ended Inference, Start to save...")

        # Save file
        with open(args.output_path, "w") as fout:
            wtr = csv.writer(fout, delimiter="\t")
            wtr.writerow(["AudioPath", "DecodedSentence"])

            for audio_path, decoded_sentence in zip(dataset_files, outputs):
                wtr.writerow((audio_path, decoded_sentence))
        logger.info(f"Saved (audio path,decoded sentence) pairs to {args.output_path}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
