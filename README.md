# Speech Recognition

[![codecov](https://codecov.io/gh/cosmoquester/speech-recognition/branch/master/graph/badge.svg?token=veHoLRzJum)](https://codecov.io/gh/cosmoquester/speech-recognition)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/speech-recognition.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/speech-recognition)


- Develope speech recognition models with tensorflow 2

# References

## LAS Model

- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- [On the Choice of Modeling Unit for Sequence-to-Sequence Speech Recognition](https://arxiv.org/abs/1902.01955)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779v3)

# Dataset Format

- Dataset File is tsv(tab separated values) format.
- The dataset file should have **header line**.
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
    --data-config-path resources/configs/libri_config.yml \
    --model-config-path resources/configs/las_small.yml \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model \
    --train-dataset-paths tests/data/pcm_dataset.tsv \
    --dev-dataset-paths tests/data/pcm_dataset.tsv
    --train-dataset-size 10000 \
    --steps-per-epoch 100 \
    --epochs 10 \
    --disable-mixed-precision \
    --device GPU
```

## Arguments

```text
  --data-config-path DATA_CONFIG_PATH
                        data processing config file
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
  --train-dataset-paths TRAIN_DATASET_PATHS
                        a tsv/tfrecord dataset file or multiple files ex)
                        *.tsv
  --dev-dataset-paths DEV_DATASET_PATHS
                        a tsv/tfrecord dataset file or multiple files ex)
                        *.tsv
  --train-dataset-size TRAIN_DATASET_SIZE
                        the number of training dataset examples
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
  --batch-size BATCH_SIZE
  --dev-batch-size DEV_BATCH_SIZE
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
  --max-over-policy {filter,slice}
                        policy for sequence whose length is over max
  --use-tfrecord        use tfrecord dataset
  --tensorboard-update-freq TENSORBOARD_UPDATE_FREQ
  --disable-mixed-precision
                        Use mixed precision FP16
  --seed SEED           Set random seed
  --device {CPU,GPU,TPU}
                        device to use (TPU or GPU or CPU)
```
- `data-config-path` is config file path for data processing. example config is `resources/configs/libri_config.yml`.
- `model-config-path` is config model file path for model initialize. default config is `resources/configs/las_small.yml`.
- `sp-model-path` is sentencepiece model path to tokenize target text.
- `pretrained-model-path` is pretrained model checkpoint path if you continue to train from pretrained model.
- `disable-mixed-precision` option is disabling mixed precision. (default use mixed precision)

# Evaluate

## Example

You can evaluate your trained model using `evaluate.py` script.
You'll get to know CER or WER as a result of evaluation like below example.

```sh
$ python -m scripts.evaluate \
  --data-config-path resources/configs/kspon_config.yml \
  --model-config-path resources/configs/las_small.yml \
  --dataset-paths asr_data/kspon/evaluate_sample.tsv \
  --model-path model-30epoch-3.1351loss_0.4070acc.ckpt \
  --sp-model-path resources/sp-models/sp_model_unigram_8K_kspon.model \
  --device GPU
...
[2021-05-12 01:54:57,000] Loaded weights of model from model-30epoch-3.1351loss_0.4070acc.ckpt
[2021-05-12 01:54:57,000] Start Inference
2021-05-12 01:54:57.031146: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-05-12 01:54:57.048736: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2198835000 Hz
2021-05-12 01:55:05.850122: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-05-12 01:55:06.198003: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-05-12 01:55:06.200574: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
[2021-05-12 01:55:09,623] Ended Inference
[2021-05-12 01:55:09,952] Average CER: 164.9703%
```

## Argument

```sh
  --data-config-path DATA_CONFIG_PATH
                        data processing config file
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --dataset-paths DATASET_PATHS
                        a tsv/tfrecord dataset file or multiple files ex)
                        *.tsv
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
  --output-path OUTPUT_PATH
                        output tsv file path to save generated sentences
  --metric {CER,WER}    metric type
  --batch-size BATCH_SIZE
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size
  --use-tfrecord        use tfrecord dataset
  --mixed-precision     Use mixed precision FP16
  --device DEVICE       device to train model
```
- `dataset-paths` is same as `dataset-paths` in train script.
- If you pass `output-path` argument, recognized text and real target text, distance metric is exported in tsv format.
- You can select your metric of CER or WER by passing `metric` argument.
# Inference

## Example

You can infer with trained model to your audio files like below example.
```sh
$ python -m scripts.inference \
  --data-config-path resources/configs/kspon_config.yml \
  --model-config-path resources/configs/las_small.yml \
  --audio-files "data/*.pcm" \
  --model-path model-30epoch-3.1351loss_0.4070acc.ckpt \
  --sp-model-path resources/sp-models/sp_model_unigram_8K_kspon.model \
  --batch-size 3 --device GPU \
  --beam-size 5
2021-05-11 00:38:45.031911: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2198800000 Hz
2021-05-11 00:38:54.666780: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-05-11 00:38:55.030704: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-05-11 00:38:55.036582: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
[2021-05-11 00:39:00,981] Ended Inference, Start to save...
[2021-05-11 00:39:00,983] Saved (audio path,decoded sentence) pairs to output.tsv
```
Then inferenced files is saved to output path.

## Argument

```sh
  --data-config-path DATA_CONFIG_PATH
                        data processing config file
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --audio-files AUDIO_FILES
                        an audio file or glob pattern of multiple files ex) *.pcm
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --output-path OUTPUT_PATH
                        output tsv file path to save generated sentences
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
  --batch-size BATCH_SIZE
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with this value as beam size
  --mixed-precision     Use mixed precision FP16
  --device DEVICE       device to train model
```
- ``audio-files`` is audio files glob pattern. i.e) "*.pcm", "data[0-9]+.wav"
- ``model-path`` is tensorflow model checkpoint path.

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
