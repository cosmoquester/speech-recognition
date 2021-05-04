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
  --dataset-path DATASET_PATH
                        a text file or multiple files ex) *.txt
  --pretrained-model-path PRETRAINED_MODEL_PATH
                        pretrained model checkpoint
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
  --output-path OUTPUT_PATH
                        output directory to save log and model checkpoints
  --epochs EPOCHS
  --steps-per-epoch STEPS_PER_EPOCH
  --learning-rate LEARNING_RATE
  --min-learning-rate MIN_LEARNING_RATE
  --warmup-rate WARMUP_RATE
  --warmup-steps WARMUP_STEPS
  --batch-size BATCH_SIZE
  --dev-batch-size DEV_BATCH_SIZE
  --total-dataset-size TOTAL_DATASET_SIZE
  --max-audio-length MAX_AUDIO_LENGTH
                        max audio sequence length
  --max-token-length MAX_TOKEN_LENGTH
                        max token sequence length
  --num-dev-dataset NUM_DEV_DATASET
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
