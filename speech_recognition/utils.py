import logging
import os
import random
import sys
from typing import Dict, Iterable, Optional, Union

import numpy as np
import tensorflow as tf

from .models import LAS, DeepSpeech2, ModelProto


def create_model(model_config: Dict) -> ModelProto:
    """
    Create model instance from config.

    :param model_config: dictionary contains 'model_name' as ASR name and initialize arguments.
    :returns: create model instance
    """
    model_name = model_config.pop("model_name").lower()

    if model_name in ["ds2", "deepspeech2"]:
        return DeepSpeech2(**model_config)
    if model_name in ["las"]:
        return LAS(**model_config)
    raise ValueError(f"Model Name: {model_name} is invalid!")


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Schedule learning rate linearly from max_learning_rate to min_learning_rate."""

    def __init__(
        self,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float = 0.0,
        warmup_steps: Optional[int] = None,
    ):
        self.warmup_steps = int(total_steps * warmup_rate) + 1 if warmup_steps is None else warmup_steps
        self.increasing_delta = max_learning_rate / self.warmup_steps
        self.decreasing_delta = (max_learning_rate - min_learning_rate) / (total_steps - self.warmup_steps)
        self.max_learning_rate = tf.cast(max_learning_rate, tf.float32)
        self.min_learning_rate = tf.cast(min_learning_rate, tf.float32)

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        lr = tf.minimum(step * self.increasing_delta, self.max_learning_rate - step * self.decreasing_delta)
        return tf.maximum(lr, self.min_learning_rate)


class LoggingCallback(tf.keras.callbacks.Callback):
    """Callback to log losses and metrics during training"""

    def __init__(self, logger: logging.Logger, logging_step: int = 100):
        self.logger = logger
        self.logging_step = logging_step

        self.step = None
        self.epoch = 1
        self.total_step = None
        self.logs = {}

    def on_batch_end(self, batch: int, logs: Dict[str, float]):
        """
        On batch end, add values of loss and metric. When step is logging step, print averaged values and reset.

        :param batch: current batch index
        :param logs: dictionary containing losses and metrics
        """
        self.step = batch + 1
        for name, value in logs.items():
            self.logs[name] = self.logs.get(name, 0) + value
        if self.step % self.logging_step == 0:
            values = ", ".join(f"{name}: {value / self.logging_step:.4f}" for name, value in self.logs.items())
            self.logger.info(f"{self.epoch} epoch, {self.step} / {self.total_step} step | " + values)
            self.logs = {}

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """
        Print epoch losses and metrics on epoch end.

        :param batch: current epoch index
        :param logs: dictionary containing losses and metrics
        """
        values = ", ".join(f"{name}: {value:.4f}" for name, value in logs.items())
        self.logger.info(f"{self.epoch} epoch | " + values)

        self.total_step = self.step
        self.epoch += 1
        self.logs = {}


def levenshtein_distance(
    truth: Union[Iterable, str], hypothesis: Union[Iterable, str], normalize=True
) -> Union[int, float]:
    """Calculate levenshtein distance (edit distance)"""
    m = len(truth) + 1
    n = len(hypothesis) + 1

    distance_matrix = np.zeros([m, n], np.int32)
    distance_matrix[0] = np.arange(n)
    distance_matrix[:, 0] = np.arange(m)

    for i in range(1, m):
        for j in range(1, n):
            is_diff = int(truth[i - 1] != hypothesis[j - 1])
            distance_matrix[i, j] = min(
                distance_matrix[i - 1, j - 1] + is_diff, distance_matrix[i - 1, j] + 1, distance_matrix[i, j - 1] + 1
            )

    if normalize:
        return distance_matrix[m - 1, n - 1] / len(truth)
    else:
        return distance_matrix[m - 1, n - 1]


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def path_join(*paths: Iterable[str]) -> str:
    """Join paths to string local paths and google storage paths also"""
    if paths[0].startswith("gs://"):
        return "/".join((path.rstrip("/") for path in paths))
    return os.path.join(*paths)


def set_random_seed(seed: int):
    """Set random seed for random / numpy / tensorflow"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_device_strategy(device) -> tf.distribute.Strategy:
    """Return tensorflow device strategy"""
    # Use TPU
    if device.upper() == "TPU":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ["TPU_NAME"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        return strategy

    # Use GPU
    if device.upper() == "GPU":
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) == 0:
            raise RuntimeError("Cannot find GPU!")
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        if len(devices) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

        return strategy

    # Use CPU
    return tf.distribute.OneDeviceStrategy("/cpu:0")
