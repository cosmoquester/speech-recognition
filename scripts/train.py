import argparse
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml

from speech_recognition.configs import TrainConfig
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
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument("--from-file", type=str, help="load configs from file")

parser.add_argument("--data-config", type=str, help="data processing config file")
parser.add_argument("--model-config", type=str, help="model config file")
parser.add_argument("--sp-model-path", type=str, help="sentencepiece model path")
parser.add_argument("--train-dataset-paths", help="a tsv/tfrecord dataset file or multiple files ex) *.tsv")
parser.add_argument("--dev-dataset-paths", help="a tsv/tfrecord dataset file or multiple files ex) *.tsv")
parser.add_argument("--train-dataset-size", type=int, help="the number of training dataset examples")
parser.add_argument("--output-path", help="output directory to save log and model checkpoints")

parser.add_argument("--pretrained-model-path", type=str, help="pretrained model checkpoint")
parser.add_argument("--epochs", type=int)
parser.add_argument("--steps-per-epoch", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--min-learning-rate", type=float)
parser.add_argument("--warmup-rate", type=float)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--dev-batch-size", type=int)
parser.add_argument("--shuffle-buffer-size", type=int, help="shuffle buffer size")
parser.add_argument("--max-over-policy", type=str, choices=["filter", "slice"], help="policy for sequence whose length is over max")

parser.add_argument("--use-tfrecord", action="store_true", help="use tfrecord dataset")
parser.add_argument("--tensorboard-update-freq", type=int)
parser.add_argument("--mixed-precision", action="store_true", help="use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--skip-epochs", type=int, help="skip first N epochs and start N + 1 epoch")
parser.add_argument("--device", type=str, choices=["CPU", "GPU", "TPU"], help="device to use (TPU or GPU or CPU)")
# fmt: on


def main(cfg: TrainConfig):
    logger = get_logger("train")

    if cfg.seed:
        set_random_seed(cfg.seed)
        logger.info(f"[+] Set random seed to {cfg.seed}")

    # Copy config file
    tf.io.gfile.makedirs(cfg.output_path)
    with tf.io.gfile.GFile(path_join(cfg.output_path, "train_configs.txt"), "w") as fout:
        for k, v in vars(cfg).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(cfg.data_config_path, path_join(cfg.output_path, "data-config.yml"))
    tf.io.gfile.copy(cfg.model_config_path, path_join(cfg.output_path, "model-config.yml"))

    with get_device_strategy(cfg.device).scope():
        if cfg.mixed_precision:
            mixed_type = "mixed_bfloat16" if cfg.device == "TPU" else "mixed_float16"
            policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
            tf.keras.mixed_precision.experimental.set_policy(policy)
            logger.info("[+] Use Mixed Precision FP16")

        # Construct Dataset
        map_log_mel_spectrogram = tf.function(
            lambda audio, text: (
                make_log_mel_spectrogram(
                    audio,
                    cfg.data_config.sample_rate,
                    cfg.data_config.frame_length,
                    cfg.data_config.frame_step,
                    cfg.data_config.fft_length,
                    cfg.data_config.num_mel_bins,
                    cfg.data_config.lower_edge_hertz,
                    cfg.data_config.upper_edge_hertz,
                ),
                text,
            )
        )
        if cfg.use_tfrecord:
            logger.info(f"[+] Load TFRecord train dataset from {cfg.train_dataset_paths}")
            train_dataset = get_tfrecord_dataset(cfg.train_dataset_paths)
            logger.info(f"[+] Load TFRecord dev dataset from {cfg.train_dataset_paths}")
            dev_dataset = get_tfrecord_dataset(cfg.train_dataset_paths)
        else:
            # Load Tokenizer
            logger.info(f"[+] Load Tokenizer from {cfg.sp_model_path}")
            with tf.io.gfile.GFile(cfg.sp_model_path, "rb") as f:
                tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

            logger.info(f"[+] Load train dataset from {cfg.train_dataset_paths}")
            train_dataset = get_dataset(
                cfg.train_dataset_paths,
                cfg.data_config.file_format,
                cfg.data_config.sample_rate,
                tokenizer,
                cfg.shuffle_buffer_size > 1,
            ).map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            logger.info(f"[+] Load dev dataset from {cfg.dev_dataset_paths}")
            dev_dataset = get_dataset(
                cfg.dev_dataset_paths,
                cfg.data_config.file_format,
                cfg.data_config.sample_rate,
                tokenizer,
            ).map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Delta Accelerate
        if cfg.data_config.use_delta_accelerate:
            logger.info("[+] Use delta and deltas accelerate")
            train_dataset = train_dataset.map(delta_accelerate)
            dev_dataset = dev_dataset.map(delta_accelerate)

        # Apply max over policy
        filter_fn = tf.function(
            lambda audio, text: tf.math.logical_and(
                tf.shape(audio)[0] <= cfg.data_config.max_audio_length,
                tf.size(text) <= cfg.data_config.max_token_length,
            )
        )
        slice_fn = tf.function(
            lambda audio, text: (
                audio[: cfg.data_config.max_audio_length],
                text[: cfg.data_config.max_token_length],
            )
        )

        if cfg.max_over_policy == "filter":
            logger.info("[+] Filter examples whose audio or token length is over than max value")
            train_dataset = train_dataset.filter(filter_fn)
            dev_dataset = dev_dataset.filter(filter_fn)
        elif cfg.max_over_policy == "slice":
            logger.info("[+] Slice examples whose audio or token length is over than max value")
            train_dataset = train_dataset.map(slice_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dev_dataset = dev_dataset.map(slice_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif cfg.device == "TPU":
            raise RuntimeError("You should set max-over-sequence-policy with TPU!")

        # Model Initialize
        with tf.io.gfile.GFile(cfg.model_config_path) as f:
            logger.info("[+] Model Initialize")
            model = create_model(cfg.model_config)

            model_input, _ = model.make_example(
                tf.keras.Input(
                    [cfg.audio_pad_length, cfg.data_config.num_mel_bins, cfg.data_config.feature_dim], dtype=tf.float32
                ),
                tf.keras.Input([cfg.token_pad_length], dtype=tf.int32),
            )
            model(model_input)
            model.summary()

        # Load pretrained model
        if cfg.pretrained_model_path:
            logger.info("[+] Load weights of model")
            model.load_weights(cfg.pretrained_model_path)

        # Model Compile
        logger.info("[+] Model compile")
        model.compile(
            optimizer=tf.optimizers.Adam(
                LRScheduler(
                    cfg.total_steps,
                    cfg.learning_rate,
                    cfg.min_learning_rate,
                    cfg.warmup_rate,
                    cfg.warmup_steps,
                    cfg.offset_steps,
                )
            ),
            loss=model.get_loss_fn(),
            metrics=model.get_metrics(),
        )

        # Shuffle & Make train example
        train_dataset = train_dataset.map(model.make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_dataset = dev_dataset.map(model.make_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if cfg.steps_per_epoch:
            logger.info("[+] Repeat dataset")
            train_dataset = train_dataset.repeat()

            if cfg.skip_epochs:
                logger.info(f"[+] Skip Dataset by {cfg.skip_epochs}epoch x {cfg.steps_per_epoch} steps")
                train_dataset = train_dataset.skip(cfg.steps_per_epoch * cfg.skip_epochs)

        # Padded Batch
        logger.info("[+] Pad Input data")
        padded_shape = model.get_batching_shape(
            cfg.audio_pad_length, cfg.token_pad_length, cfg.data_config.num_mel_bins, cfg.data_config.feature_dim
        )
        train_dataset = (
            train_dataset.shuffle(cfg.shuffle_buffer_size)
            .padded_batch(cfg.batch_size, padded_shape)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        dev_dataset = dev_dataset.padded_batch(cfg.dev_batch_size, padded_shape)

        # Training
        logger.info("[+] Start training")
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=cfg.epochs,
            initial_epoch=cfg.skip_epochs,
            steps_per_epoch=cfg.steps_per_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    path_join(cfg.output_path, "models", model.model_checkpoint_path),
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(cfg.output_path, "logs"), update_freq=cfg.tensorboard_update_freq
                ),
            ],
        )


if __name__ == "__main__":
    config = vars(parser.parse_args())
    if "from_file" in config:
        with open(config.pop("from_file")) as f:
            config = {**yaml.load(f, yaml.SafeLoader), **config}
    sys.exit(main(TrainConfig(**config)))
