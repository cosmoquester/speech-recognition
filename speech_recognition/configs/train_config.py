from dataclasses import InitVar
from math import ceil
from typing import Optional

import yaml
from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from .data_config import DataConfig
from .model_config import ModelConfig, get_model_config


@dataclass
class TrainConfig:
    # config paths
    data_config_path: str = ""
    model_config_path: str = ""

    # data processing config
    data_config: InitVar[DataConfig] = Field(...)
    # model config
    model_config: InitVar[ModelConfig] = Field(...)
    # sentencepiece model path
    sp_model_path: Optional[str] = None
    # a tsv/tfrecord dataset file or multiple files ex) *.tsv
    train_dataset_paths: str = Field(...)
    # a tsv/tfrecord dataset file or multiple files ex) *.tsv
    dev_dataset_paths: str = Field(...)
    # the number of training dataset examples
    train_dataset_size: int = Field(...)
    # output directory to save log and model checkpoints
    output_path: str = "output"

    # pretrained model checkpoint
    pretrained_model_path: Optional[str] = None

    # training parameters
    epochs: int = Field(...)
    steps_per_epoch: Optional[int] = None
    learning_rate: float = Field(...)
    min_learning_rate: float = 1.0e-5
    warmup_rate: float = 0.00
    warmup_steps: Optional[int] = None
    batch_size: int = Field(...)
    dev_batch_size: int = Field(...)

    # shuffle buffer size
    shuffle_buffer_size: int = 10000
    # policy for sequence whose length is over max
    max_over_policy: Optional[Literal["filter", "slice"]] = None

    # use tfrecord dataset
    use_tfrecord: bool = False
    # tensorboard update frequency
    tensorboard_update_freq: int = 1
    # use mixed precision FP16
    mixed_precision: bool = False
    # Set random seed
    seed: Optional[int] = None
    # skip first N epochs and start N + 1 epoch
    skip_epochs: int = 0
    # device to use
    device: Literal["CPU", "GPU", "TPU"] = "CPU"

    def __post_init_post_parse__(self, data_config: str, model_config: str):
        assert isinstance(data_config, str), "should pass 'data_config' parameter"
        assert isinstance(model_config, str), "should pass 'model_config' parameter"

        self.data_config_path = data_config
        self.model_config_path = model_config

        self.data_config = DataConfig.from_yaml(data_config)
        self.model_config = get_model_config(model_config)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path) as f:
            return cls(**yaml.load(f, yaml.SafeLoader))

    @property
    def audio_pad_length(self):
        return None if self.device != "TPU" else self.data_config.max_audio_length

    @property
    def token_pad_length(self):
        return None if self.device != "TPU" else self.data_config.max_token_length

    @property
    def total_steps(self):
        return (self.steps_per_epoch or ceil(self.train_dataset_size / self.batch_size)) * self.epochs

    @property
    def offset_steps(self):
        return (self.steps_per_epoch or ceil(self.train_dataset_size / self.batch_size)) * self.skip_epochs
