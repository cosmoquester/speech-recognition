import argparse
from math import ceil

import tensorflow as tf
import tensorflow_text as text
from omegaconf import OmegaConf

from speech_recognition.data import delta_accelerate, get_dataset, get_tfrecord_dataset, make_log_mel_spectrogram
from speech_recognition.utils import (
    LRScheduler,
    create_model,
    get_device_strategy,
    get_logger,
    path_join,
    set_random_seed,
)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--data-config-path", type=str, required=True, help="data processing config file")
parser.add_argument("--model-config-path", type=str, default="resources/configs/las_small.yml", help="model config file")
parser.add_argument("--sp-model-path", type=str, default=None, help="sentencepiece model path")
parser.add_argument("--train-dataset-paths", required=True, help="a tsv/tfrecord dataset file or multiple files ex) *.tsv")
parser.add_argument("--dev-dataset-paths", required=True, help="a tsv/tfrecord dataset file or multiple files ex) *.tsv")
parser.add_argument("--train-dataset-size", type=int, required=True, help="the number of training dataset examples")
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")

parser.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--steps-per-epoch", type=int, default=None)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.00)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--dev-batch-size", type=int, default=64)
parser.add_argument("--shuffle-buffer-size", type=int, default=5000, help="shuffle buffer size")
parser.add_argument("--max-over-policy", type=str, choices=["filter", "slice"], help="policy for sequence whose length is over max")

parser.add_argument("--use-tfrecord", action="store_true", help="use tfrecord dataset")
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--validation-freq", type=int, default=10, help="validation frequency (every this epoch)")
parser.add_argument("--disable-mixed-precision", action="store_false", dest="mixed_precision", help="use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()

    logger = get_logger("train")

    if args.seed:
        set_random_seed(args.seed)
        logger.info(f"[+] Set random seed to {args.seed}")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.data_config_path, path_join(args.output_path, "data-config.yml"))
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model-config.yml"))

    with get_device_strategy(args.device).scope():
        if args.mixed_precision:
            mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
            policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
            tf.keras.mixed_precision.experimental.set_policy(policy)
            logger.info("[+] Use Mixed Precision FP16")

        # Load Config
        logger.info(f"[+] Load Data Config from {args.data_config_path}")
        with tf.io.gfile.GFile(args.data_config_path) as f:
            config = OmegaConf.load(f)

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
            logger.info(f"[+] Load TFRecord train dataset from {args.train_dataset_paths}")
            train_dataset = get_tfrecord_dataset(args.train_dataset_paths)
            logger.info(f"[+] Load TFRecord dev dataset from {args.train_dataset_paths}")
            dev_dataset = get_tfrecord_dataset(args.train_dataset_paths)
        else:
            # Load Tokenizer
            logger.info(f"[+] Load Tokenizer from {args.sp_model_path}")
            with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
                tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

            logger.info(f"[+] Load train dataset from {args.train_dataset_paths}")
            train_dataset = get_dataset(
                args.train_dataset_paths,
                config.file_format,
                config.sample_rate,
                tokenizer,
                args.shuffle_buffer_size > 1,
            ).map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            logger.info(f"[+] Load dev dataset from {args.dev_dataset_paths}")
            dev_dataset = get_dataset(
                args.dev_dataset_paths,
                config.file_format,
                config.sample_rate,
                tokenizer,
            ).map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Delta Accelerate
        if config.use_delta_accelerate:
            logger.info("[+] Use delta and deltas accelerate")
            train_dataset = train_dataset.map(delta_accelerate)
            dev_dataset = dev_dataset.map(delta_accelerate)
            feature_dim = 3
        else:
            feature_dim = 1

        # Apply max over policy
        filter_fn = tf.function(
            lambda audio, text: tf.math.logical_and(
                tf.shape(audio)[0] <= config.max_audio_length, tf.size(text) <= config.max_token_length
            )
        )
        slice_fn = tf.function(
            lambda audio, text: (
                audio[: config.max_audio_length],
                text[: config.max_token_length],
            )
        )

        if args.max_over_policy == "filter":
            logger.info("[+] Filter examples whose audio or token length is over than max value")
            train_dataset = train_dataset.filter(filter_fn)
            dev_dataset = dev_dataset.filter(filter_fn)
        elif args.max_over_policy == "slice":
            logger.info("[+] Slice examples whose audio or token length is over than max value")
            train_dataset = train_dataset.map(slice_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dev_dataset = dev_dataset.map(slice_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif args.device == "TPU":
            raise RuntimeError("You should set max-over-sequence-policy with TPU!")

        # Model Initi alize
        audio_pad_length = None if args.device != "TPU" else config.max_audio_length
        token_pad_length = None if args.device != "TPU" else config.max_token_length
        with tf.io.gfile.GFile(args.model_config_path) as f:
            logger.info("[+] Model Initialize")
            model = create_model(OmegaConf.load(f))

            model_input, _ = model.make_example(
                tf.keras.Input([audio_pad_length, config.num_mel_bins, feature_dim], dtype=tf.float32),
                tf.keras.Input([token_pad_length], dtype=tf.int32),
            )
            model(model_input)
            model.summary()

        # Load pretrained model
        if args.pretrained_model_path:
            logger.info("[+] Load weights of model")
            model.load_weights(args.pretrained_model_path)

        # Model Compile
        logger.info("[+] Model compile")
        total_steps = ceil(args.train_dataset_size / args.batch_size) * args.epochs
        model.compile(
            optimizer=tf.optimizers.Adam(
                LRScheduler(
                    total_steps, args.learning_rate, args.min_learning_rate, args.warmup_rate, args.warmup_steps
                )
            ),
            loss=model.get_loss_fn(),
            metrics=model.get_metrics(),
        )

        # Shuffle & Make train example
        train_dataset = train_dataset.map(model.make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_dataset = dev_dataset.map(model.make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if args.steps_per_epoch:
            logger.info("[+] Repeat dataset")
            train_dataset = train_dataset.repeat()

        # Padded Batch
        logger.info("[+] Pad Input data")
        padded_shape = model.get_batching_shape(audio_pad_length, token_pad_length, config.num_mel_bins, feature_dim)
        train_dataset = (
            train_dataset.shuffle(args.shuffle_buffer_size)
            .padded_batch(args.batch_size, padded_shape)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        dev_dataset = dev_dataset.padded_batch(args.dev_batch_size, padded_shape)

        # Training
        logger.info("[+] Start training")
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
            validation_freq=args.validation_freq,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    path_join(
                        args.output_path,
                        "models",
                        "model-{epoch}epoch-{loss:.4f}loss_{accuracy:.4f}acc.ckpt",
                    ),
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(args.output_path, "logs"), update_freq=args.tensorboard_update_freq
                ),
            ],
        )
