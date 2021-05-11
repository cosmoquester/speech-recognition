import numpy as np
import pytest

from speech_recognition.utils import LRScheduler, levenshtein_distance


@pytest.mark.parametrize(
    "num_epoch,learning_rate,min_learning_rate,warmup_rate",
    [(10, 1.1, 0.0, 0.3), (33, 1e-5, 1e-7, 0.1), (100, 100, 0, 0.5)],
)
def test_learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate, warmup_rate):
    fn = LRScheduler(num_epoch, learning_rate, min_learning_rate, warmup_rate)

    for i in range(num_epoch):
        learning_rate = fn(i)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)


@pytest.mark.parametrize(
    "truth,hypothesis,distance,normalize",
    [
        ([], [], 0, False),
        (list("abc"), [], 3, False),
        ("hello", "hello", 0, False),
        (list("kitten"), list("sitten"), 1, False),
        ("sunday", "saturday", 3, False),
        ([1, 2, 3], [4, 4, 4, 5], 4, False),
        (list("안녕하세요"), list("안녕? 해..?"), 6, False),
        ("hi", "hello", 2.0, True),
        ("byebye", "yes", 2 / 3, True),
    ],
)
def test_levenshtein_distance(truth, hypothesis, distance, normalize):
    assert levenshtein_distance(truth, hypothesis, normalize) == distance
