import argparse
import csv
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml

from speech_recognition.configs import DataConfig, ModelConfig
from speech_recognition.data import delta_accelerate, get_dataset, get_tfrecord_dataset, make_log_mel_spectrogram
from speech_recognition.models import LAS, DeepSpeech2
from speech_recognition.search import DeepSpeechSearcher, LAS_Searcher
from speech_recognition.utils import create_model, get_device_strategy, get_logger, levenshtein_distance

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
parser.add_argument("--data-config", type=str, required=True, help="data processing config file")
parser.add_argument("--model-config", type=str, required=True, help="model config file")
parser.add_argument("--dataset-paths", required=True, help="a tsv/tfrecord dataset file or multiple files ex) *.tsv")
parser.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
parser.add_argument("--sp-model-path", type=str, required=True, help="sentencepiece model path")
parser.add_argument("--output-path", help="output tsv file path to save generated sentences")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")
parser.add_argument("--use-tfrecord", action="store_true", help="use tfrecord dataset")
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on


def main(args: argparse.Namespace):
    strategy = get_device_strategy(args.device)

    logger = get_logger("inference")

    if args.mixed_precision:
        mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
        policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("[+] Use Mixed Precision FP16")

    # Load Tokenizer
    logger.info(f"[+] Load Tokenizer from {args.sp_model_path}")
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()

    # Load Config
    logger.info(f"[+] Load Data Config from {args.data_config}")
    config = DataConfig.from_yaml(args.data_config)

    with strategy.scope():
        # Construct Dataset
        map_log_mel_spectrogram = tf.function(
            lambda audio, text: (
                make_log_mel_spectrogram(
                    audio,
                    config.sample_rate,
                    config.frame_length,
                    config.frame_step,
                    config.fft_length,
                    config.num_mel_bins,
                    config.lower_edge_hertz,
                    config.upper_edge_hertz,
                ),
                text,
            )
        )

        if args.use_tfrecord:
            logger.info(f"[+] Load TFRecord dataset from {args.dataset_paths}")
            dataset = get_tfrecord_dataset(args.dataset_paths)
        else:
            logger.info(f"[+] Load dataset from {args.dataset_paths}")
            dataset = get_dataset(args.dataset_paths, config.file_format, config.sample_rate, tokenizer).map(
                map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        # Delta Accelerate
        if config.use_delta_accelerate:
            logger.info("[+] Use delta and deltas accelerate")
            dataset = dataset.map(delta_accelerate)

        # Model Initialize & Load pretrained model
        model_config = ModelConfig.from_yaml(args.model_config)
        model = create_model(model_config)

        model_input, _ = model.make_example(
            tf.keras.Input([None, config.num_mel_bins, config.feature_dim], dtype=tf.float32),
            tf.keras.Input([None], dtype=tf.int32),
        )
        model(model_input)
        tf.train.Checkpoint(model).restore(args.model_path).expect_partial()
        model.summary()

        audio_pad_length = None if args.device != "TPU" else config.max_audio_length
        token_pad_length = None if args.device != "TPU" else config.max_token_length
        dataset = dataset.padded_batch(
            args.batch_size, ([audio_pad_length, config.num_mel_bins, config.feature_dim], [token_pad_length])
        )

        if isinstance(model, LAS):
            searcher = LAS_Searcher(model, config.max_token_length, bos_id, eos_id, model_config.pad_id)
        elif isinstance(model, DeepSpeech2):
            searcher = DeepSpeechSearcher(model, model_config.blank_index)
        logger.info(f"[+] Loaded weights of model from {args.model_path}")

        # Inference
        logger.info("[+] Start Inference")
        outputs = []

        for batch_input, target in dataset:
            if args.beam_size > 0:
                batch_output = searcher.beam_search(batch_input, args.beam_size)
                batch_output = batch_output[0][:, 0, :]
            else:
                batch_output = searcher.greedy_search(batch_input)[0]
            outputs.extend(zip(batch_output, target))
        logger.info("[+] Ended Inference")

        to_str = lambda tokens: tokenizer.detokenize(tokens).numpy().decode("UTF8")
        outputs = [(to_str(pred), to_str(target)) for pred, target in outputs]

        # Print metric
        wers = []
        cers = []
        for pred, target in outputs:
            wers.append(levenshtein_distance(target.split(), pred.split(), True))
            cers.append(levenshtein_distance(target, pred, True))

        logger.info(f"[+] Average WER: {sum(wers) / len(wers) * 100:.4f}%")
        logger.info(f"[+] Average CER: {sum(cers) / len(cers) * 100:.4f}%")

        # Save file
        if args.output_path:
            with open(args.output_path, "w") as fout:
                wtr = csv.writer(fout, delimiter="\t")
                wtr.writerow(["Prediction", "Target", "WER", "CER"])

                for (pred, target), wer, cer in zip(outputs, wers, cers):
                    wtr.writerow((pred, target, wer, cer))
            logger.info(f"[+] Saved (Prediction, Target) pairs to {args.output_path}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
