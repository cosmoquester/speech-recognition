# Speech Recognition

[![codecov](https://codecov.io/gh/cosmoquester/speech-recognition/branch/master/graph/badge.svg?token=veHoLRzJum)](https://codecov.io/gh/cosmoquester/speech-recognition)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/speech-recognition.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/speech-recognition)


- This is for speech recognition including models and train, evaluate, inference scripts based tensorflow 2
- You can execute script examples on below descriptions with test data
- `resources/configs` directory contains default datasets (LibriSpeech, KsponSpeech, Clovacall) and models (LAS, DeepSpeech2) configs.
- `resources/sp-models` directory contains default sentencepiece tokenizer for each datasets

# References

## LAS Model

- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- [On the Choice of Modeling Unit for Sequence-to-Sequence Speech Recognition](https://arxiv.org/abs/1902.01955)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779v3)

## DeepSpeech2 Model

- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
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
$ python -m speech_recognition.run.train \
    --data-config resources/configs/libri_config.yml \
    --model-config resources/configs/las_small.yml \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model \
    --train-dataset-paths tests/data/wav_dataset.tsv \
    --dev-dataset-paths tests/data/wav_dataset.tsv \
    --train-dataset-size 1000 \
    --steps-per-epoch 100 \
    --epochs 10 \
    --batch-size 32 \
    --dev-batch-size 32 \
    --learning-rate 2e-4 \
    --mixed-precision \
    --device CPU
```
You can also start training with train configuration file using `--from-file` parameter.
```sh
$ python -m speech_recognition.run.train --from-file resources/configs/train_config_sample.yml
```
And you can override the parameter of file by command line arguments like below.
$ python -m speech_recognition.run.train \
    --from-file resources/configs/train_config_sample.yml \
    --epochs 1 \
    --batch-size 128 \
    --device GPU
```

## Arguments

```text
  --from-file FROM_FILE
                        load configs from file
  --data-config DATA_CONFIG
                        data processing config file
  --model-config MODEL_CONFIG
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
                        shuffle buffer size
  --max-over-policy {filter,slice}
                        policy for sequence whose length is over max
  --use-tfrecord        use tfrecord dataset
  --tensorboard-update-freq TENSORBOARD_UPDATE_FREQ
  --mixed-precision     use mixed precision FP16
  --seed SEED           Set random seed
  --skip-epochs SKIP_EPOCHS
                        skip first N epochs and start N + 1 epoch
  --device {CPU,GPU,TPU}
                        device to use (TPU or GPU or CPU)
```
- `data-config` is config file path for data processing. example config is `resources/configs/libri_config.yml`.
- `model-config` is config model file path for model initialize. default config is `resources/configs/las_small.yml`.
- `sp-model-path` is sentencepiece model path to tokenize target text.
- `pretrained-model-path` is pretrained model checkpoint path if you continue to train from pretrained model.
- `warmup-rate` or `warmup-steps` specify warmup steps. default is zero. `warmup-steps` is used if both of params provided.
- `max-over-policy` option is for sequences whose length is over than max sequence. You can filter longer example or slice to fit length.
- `use-tfrecord` option should be provided when using TFRecord format dataset.
- `mixed-precision` option is enabling FP16 mixed precision.

# Evaluate

## Example

You can evaluate your trained model using `evaluate.py` script.
You'll get to know CER or WER as a result of evaluation like below example.

```sh
$ python -m speech_recognition.run.evaluate \
    --data-config resources/configs/libri_config.yml \
    --model-config tests/data/model-configs/las_mini_for_test.yml \
    --dataset-paths tests/data/wav_dataset.tsv \
    --model-path tests/data/model-checkpoints/las.ckpt \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model \
    --device CPU
...
[2021-06-07 13:22:48,599] [+] Load Tokenizer from resources/sp-models/sp_model_unigram_16K_libri.model
[2021-06-07 13:22:48,626] [+] Load Data Config from resources/configs/libri_config.yml
[2021-06-07 13:22:48,629] [+] Load dataset from tests/data/wav_dataset.tsv
2021-06-07 13:22:49.018137: I tensorflow_io/core/kernels/cpu_check.cc:128] Your CPU supports instructions that this TensorFlow IO binary was not compiled to use: AVX2 FMA
[2021-06-07 13:22:49,662] [+] Use delta and deltas accelerate
[2021-06-07 13:22:53,122] [+] Load weights of model from tests/data/model-checkpoints/las.ckpt
Model: "las"
...
[2021-06-07 13:22:53,135] [+] Start Inference
2021-06-07 13:22:53.171394: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-06-07 13:22:53.188758: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2198835000 Hz
[2021-06-07 13:22:56,352] [+] Ended Inference
[2021-06-07 13:22:56,589] [+] Average WER: 2494.6429%
[2021-06-07 13:22:56,589] [+] Average CER: 7256.3131%
```

## Argument

```sh
  --data-config DATA_CONFIG
                        data processing config file
  --model-config MODEL_CONFIG
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
  --batch-size BATCH_SIZE
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size
  --use-tfrecord        use tfrecord dataset
  --mixed-precision     Use mixed precision FP16
  --device DEVICE       device to train
```
- `dataset-paths` is same as `dataset-paths` in train script.
- If you pass `output-path` argument, recognized text and real target text, distance metric is exported in tsv format.
- You can select your metric of CER or WER by passing `metric` argument.
# Inference

## Example

You can infer with trained model to your audio files like below example.
```sh
$ python -m speech_recognition.run.inference \
    --data-config resources/configs/libri_config.yml \
    --model-config tests/data/model-configs/las_mini_for_test.yml \
    --audio-files "tests/data/audio_files/*.wav"  \
    --model-path tests/data/model-checkpoints/las.ckpt \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model \
    --batch-size 3 \
    --device CPU \
    --beam-size 2

...
[2021-06-07 13:28:27,696] [+] Use delta and deltas accelerate
[2021-06-07 13:28:31,202] Loaded weights of model from tests/data/model-checkpoints/las.ckpt
Model: "las"
(MODEL SUMMARY)
[2021-06-07 13:28:31,204] Start Inference
2021-06-07 13:28:31.238552: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-06-07 13:28:31.256769: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2198835000 Hz
[2021-06-07 13:28:35,693] Ended Inference, Start to save...
[2021-06-07 13:28:35,694] Saved (audio path,decoded sentence) pairs to output.tsv
```
Then inferenced files is saved to output path.

## Argument

```sh
  --data-config DATA_CONFIG
                        data processing config file
  --model-config MODEL_CONFIG
                        model config file
  --audio-files AUDIO_FILES
                        an audio file or glob pattern of multiple files ex)
                        *.pcm
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --output-path OUTPUT_PATH
                        output tsv file path to save generated sentences
  --sp-model-path SP_MODEL_PATH
                        sentencepiece model path
  --batch-size BATCH_SIZE
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size
  --mixed-precision     Use mixed precision FP16
  --device DEVICE       device to train
```
- ``audio-files`` is audio files glob pattern. i.e) "*.pcm", "data[0-9]+.wav"
- ``model-path`` is tensorflow model checkpoint path.

# Make TFRecord

## Example

You can convert dataset into TFRecord format like below example.
```sh
$ python -m speech_recognition.run.make_tfrecord \
    --data-config resources/configs/libri_config.yml \
    --dataset-paths tests/data/wav_dataset.tsv \
    --sp-model-path resources/sp-models/sp_model_unigram_16K_libri.model \
    --output-dir .

[2021-06-07 13:31:10,444] [+] Number of Dataset Files: 1
[2021-06-07 13:31:10,445] [+] Load Config From resources/configs/libri_config.yml
[2021-06-07 13:31:10,447] [+] Load Tokenizer From resources/sp-models/sp_model_unigram_16K_libri.model
...
2021-06-07 13:31:10.491991: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
[2021-06-07 13:31:10,519] [+] Start Saving Dataset...
  0%|                                                                                                                                                                                        | 0/1 [00:00<?, ?it/s]2021-06-07 13:31:10.848397: I tensorflow_io/core/kernels/cpu_check.cc:128] Your CPU supports instructions that this TensorFlow IO binary was not compiled to use: AVX2 FMA
2021-06-07 13:31:11.530043: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-06-07 13:31:11.548833: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2198835000 Hz
100%|█| 1/1 [00:01<00:00,  1.35s/it]
[2021-06-07 13:31:11,867] [+] Done
```

## Argument

```text
  --data-config DATA_CONFIG
                        data processing config file
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
