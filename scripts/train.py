import argparse
from math import ceil

import tensorflow as tf
import tensorflow_text as text
from omegaconf import OmegaConf

from speech_recognition.data import get_dataset, make_log_mel_spectrogram, make_train_examples
from speech_recognition.model import LAS
from speech_recognition.utils import LRScheduler, get_device_strategy, get_logger, path_join, set_random_seed

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, required=True, help="model config file")
parser.add_argument("--sp-model-path", required=True, help="sentencepiece model path")
parser.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
parser.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
parser.add_argument("--shuffle-buffer-size", type=int, default=5000)
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--steps-per-epoch", type=int, default=None)
parser.add_argument("--learning-rate", type=float, default=2e-3)
parser.add_argument("--min-learning-rate", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.06)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--dev-batch-size", type=int, default=2)
parser.add_argument("--total-dataset-size", type=int, default=1000)
parser.add_argument("--max-audio-length", type=int, default=65536, help="max audio sequence length")
parser.add_argument("--max-token-length", type=int, default=128, help="max token sequence length")
parser.add_argument("--num-dev-dataset", type=int, default=2)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--disable-mixed-precision", action="store_false", dest="mixed_precision", help="Use mixed precision FP16")
parser.add_argument("--seed", type=int, help="Set random seed")
parser.add_argument("--device", type=str, default="CPU", help="device to use (TPU or GPU or CPU)")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()

    logger = get_logger()

    if args.seed:
        set_random_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.config_path, path_join(args.output_path, "config.yml"))

    with get_device_strategy(args.device).scope():
        if args.mixed_precision:
            mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
            policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
            tf.keras.mixed_precision.experimental.set_policy(policy)
            logger.info("Use Mixed Precision FP16")

        # Construct Dataset
        with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
            tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

        # Load Config
        with tf.io.gfile.GFile(args.config_path) as f:
            config = OmegaConf.load(f)

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
        dataset = (
            get_dataset(args.dataset_path, tokenizer, config.desired_channels)
            .shuffle(args.shuffle_buffer_size)
            .map(map_log_mel_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(make_train_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .unbatch()
        )
        train_dataset = dataset.skip(args.num_dev_dataset).padded_batch(
            args.batch_size, (([args.max_audio_length, config.num_mel_bins], [args.max_token_length]), ())
        )

        dev_dataset = dataset.take(args.num_dev_dataset).padded_batch(
            args.dev_batch_size, (([args.max_audio_length, config.num_mel_bins], [args.max_token_length]), ())
        )

        if args.steps_per_epoch:
            train_dataset = train_dataset.repeat()
            logger.info("Repeat dataset")

        # Model Initialize
        with tf.io.gfile.GFile(args.config_path) as f:
            model = LAS(
                config.vocab_size,
                config.hidden_dim,
                config.num_encoder_layers,
                config.num_decoder_layers,
                config.pad_id,
            )

        # Load pretrained model
        if args.pretrained_model_path:
            model.load_weights(args.pretrained_model_path)
            logger.info("Loaded weights of model")

        # Model Compile
        total_steps = ceil((args.total_dataset_size - args.num_dev_dataset) / args.batch_size) * args.epochs
        model.compile(
            optimizer=tf.optimizers.Adam(
                LRScheduler(
                    total_steps, args.learning_rate, args.min_learning_rate, args.warmup_rate, args.warmup_steps
                )
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        logger.info("Model compiling complete")
        logger.info("Start training")

        # Training
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    path_join(
                        args.output_path,
                        "models",
                        "model-{epoch}epoch-{val_loss:.4f}loss_{val_accuracy:.4f}acc.ckpt",
                    ),
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(args.output_path, "logs"), update_freq=args.tensorboard_update_freq
                ),
            ],
        )
