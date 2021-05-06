# Speech Recognition

[![codecov](https://codecov.io/gh/cosmoquester/speech-recognition/branch/master/graph/badge.svg?token=veHoLRzJum)](https://codecov.io/gh/cosmoquester/speech-recognition)

- Develope speech recognition models with tensorflow 2

# References

## LAS Model

- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- [On the Choice of Modeling Unit for Sequence-to-Sequence Speech Recognition](https://arxiv.org/abs/1902.01955)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779v3)

# Dataset Format

- Dataset File is tsv(tab separated values) format.
- The dataset file has to **header line**.
- The 1st column is **audio file path** relative to directory that contains dataset tsv file.
- The 2nd column is **recognized text**.
- Refer to `tests/data/dataset.tsv` file.

FilePath | Text
---|---
audio/001.wav | 안녕하세요
audio/002.wav | 반갑습니다
audio/003.wav | 근데 이름이 어떻게 되세요?
... | ...
- This is tsv file example.

# Train

## Example

You can start training by running script like below example.
```sh
$ python -m scripts.train \
    --config-path resources/configs/default_config.yml \
    --sp-model-path some_sentence_piece.model \
    --dataset-path tests/data/dataset.tsv \
    --steps-per-epoch 100 \
    --epochs 10 \
    --disable-mixed-precision \
    --device GPU
```

## Arguments

```text
  --config-path CONFIG_PATH
                        model config file
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
  --dataset-paths DATASET_PATHS
                        a tsv dataset file or multiple files ex) *.tsv
  --output-path OUTPUT_PATH
                        output directory to save log and model checkpoints
  --pretrained-model-path PRETRAINED_MODEL_PATH
                        pretrained model checkpoint
  --epochs EPOCHS
  --steps-per-epoch STEPS_PER_EPOCH
  --learning-rate LEARNING_RATE
  --min-learning-rate MIN_LEARNING_RATE
  --warmup-rate WARMUP_RATE
  --warmup-steps WARMUP_STEPS
  --max-audio-length MAX_AUDIO_LENGTH
                        max audio sequence length
  --max-token-length MAX_TOKEN_LENGTH
                        max token sequence length
  --batch-size BATCH_SIZE
  --dev-batch-size DEV_BATCH_SIZE
  --total-dataset-size TOTAL_DATASET_SIZE
  --num-dev-dataset NUM_DEV_DATASET
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
  --use-tfrecord        use tfrecord dataset
  --tensorboard-update-freq TENSORBOARD_UPDATE_FREQ
  --disable-mixed-precision
                        Use mixed precision FP16
  --seed SEED           Set random seed
  --device DEVICE       device to use (TPU or GPU or CPU)
```
- `config-path` is config file path for training. example config is `resources/configs/default_config.yml`.
- `sp-model-path` is sentencepiece model path to tokenize target text.
- `pretrained-model-path` is pretrained model checkpoint path if you continue to train from pretrained model.
- `max-audio-length` is max audio frame length. This means the number of maximum timestep of log mel spectrogram.
- `max-token-length` is max token number.
- `disable-mixed-precision` option is disabling mixed precision. (default use mixed precision)d

# Make TFRecord

## Example

You can convert dataset into TFRecord format like below example.
```sh
$ python -m scripts.make_tfrecord \
    --config-path resources/configs/kspon_config.yml \
    --dataset-paths tests/data/pcm_dataset.tsv \
    --output-dir . \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model
[2021-05-07 01:30:26,802] [+] Number of Dataset Files: 1
[2021-05-07 01:30:26,802] [+] Load Config From resources/configs/kspon_config.yml
[2021-05-07 01:30:26,809] [+] Load Tokenizer From resources/sp-models/sp_model_unigram_16K_libri.model
2021-05-07 01:30:26.811088: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-05-07 01:30:26.811555: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[2021-05-07 01:30:26,841] [+] Start Saving Dataset...
  0%|           | 0/1 [00:00<?, ?it/s]2021-05-07 01:30:27.592450: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
100%|██████| 1/1 [00:00<00:00,  1.11it/s]
[2021-05-07 01:30:27,749] [+] Done
```

## Argument

```text
  --config-path CONFIG_PATH
                        config file for processing dataset
  --dataset-paths DATASET_PATHS
                        dataset file path glob pattern
  --output-dir OUTPUT_DIR
                        output directory path, default is input dataset file
                        directoruy
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
```
- The arguments is same as train script arguments.
- The output TFRecord file contains already pre-processed audio tensors and tokenized tensors, so you can train with only TFRecord file without tsv or audio files.
