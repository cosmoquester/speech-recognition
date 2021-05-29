import argparse
import csv
import sys
from functools import partial

import tensorflow as tf
import tensorflow_text as text
from omegaconf import OmegaConf

from speech_recognition.data import delta_accelerate, load_audio_file, make_log_mel_spectrogram
from speech_recognition.models import LAS, DeepSpeech2
from speech_recognition.search import DeepSpeechSearcher, LAS_Searcher
from speech_recognition.utils import create_model, get_device_strategy, get_logger

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
parser.add_argument("--data-config-path", type=str, required=True, help="data processing config file")
parser.add_argument("--model-config-path", type=str, required=True, help="model config file")
parser.add_argument("--audio-files", required=True, help="an audio file or glob pattern of multiple files ex) *.pcm")
parser.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
parser.add_argument("--output-path", default="output.tsv", help="output tsv file path to save generated sentences")
parser.add_argument("--sp-model-path", type=str, required=True, help="sentencepiece model path")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")
parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
parser.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()
    strategy = get_device_strategy(args.device)

    logger = get_logger("inference")

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Construct Dataset
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()

    dataset_files = sorted(tf.io.gfile.glob(args.audio_files))
    if not dataset_files:
        logger.error("[Error] Dataset path is invalid!")
        sys.exit(1)

    # Load Config
    logger.info(f"Load Data Config from {args.data_config_path}")
    with tf.io.gfile.GFile(args.data_config_path) as f:
        config = OmegaConf.load(f)

    with strategy.scope():
        dataset = (
            tf.data.Dataset.from_tensor_slices(dataset_files)
            .map(
                partial(
                    load_audio_file,
                    sample_rate=config.sample_rate,
                    file_format=config.file_format,
                    resample=config.sample_rate,
                )
            )
            .map(
                partial(
                    make_log_mel_spectrogram,
                    sample_rate=config.sample_rate,
                    frame_length=config.frame_length,
                    frame_step=config.frame_step,
                    fft_length=config.fft_length,
                    num_mel_bins=config.num_mel_bins,
                    lower_edge_hertz=config.lower_edge_hertz,
                    upper_edge_hertz=config.upper_edge_hertz,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(delta_accelerate)
            .padded_batch(args.batch_size, [None, config.num_mel_bins, 3])
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Model Initialize & Load pretrained model
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model_config = OmegaConf.load(f)
            model = create_model(model_config)

            model_input, _ = model.make_example(
                tf.keras.Input([None, config.num_mel_bins, 3], dtype=tf.float32),
                tf.keras.Input([None], dtype=tf.int32),
            )
            model(model_input)
            model.load_weights(args.model_path)
            model.summary()

        if isinstance(model, LAS):
            searcher = LAS_Searcher(model, config.max_token_length, bos_id, eos_id, model_config.pad_id)
        elif isinstance(model, DeepSpeech2):
            searcher = DeepSpeechSearcher(model, config.max_token_length, model_config.blank_index)
        logger.info(f"Loaded weights of model from {args.model_path}")

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
