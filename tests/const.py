import os

RESOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
SP_MODEL_DIR = os.path.join(RESOURCE_DIR, "sp-models")
CONFIG_DIR = os.path.join(RESOURCE_DIR, "configs")

DEFAULT_LIBRI_CONFIG = os.path.join(CONFIG_DIR, "libri_config.yml")
SP_MODEL_LIBRI = os.path.join(SP_MODEL_DIR, "sp_model_unigram_16K_libri.model")

DEFAULT_LAS_CONFIG = os.path.join(CONFIG_DIR, "las_small.yml")
DEFAULT_DS_CONFIG = os.path.join(CONFIG_DIR, "deepspeech.yml")


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAV_DATASET_PATH = os.path.join(TEST_DATA_DIR, "wav_dataset.tsv")
PCM_DATASET_PATH = os.path.join(TEST_DATA_DIR, "pcm_dataset.tsv")
MP3_DATASET_PATH = os.path.join(TEST_DATA_DIR, "mp3_dataset.tsv")
TFRECORD_DATASET_PATH = os.path.join(TEST_DATA_DIR, "wav_dataset.tfrecord")

TEST_CHECKPOINT_DIR = os.path.join(TEST_DATA_DIR, "model-checkpoints")
DEFAULT_DS_CHECKPOINT = os.path.join(TEST_CHECKPOINT_DIR, "ds.ckpt")
DEFAULT_LAS_CHECKPOINT = os.path.join(TEST_CHECKPOINT_DIR, "las.ckpt")

TEST_MODEL_CONFIG_DIR = os.path.join(TEST_DATA_DIR, "model-configs")
TEST_LAS_CONFIG = os.path.join(TEST_MODEL_CONFIG_DIR, "las_mini_for_test.yml")
TEST_DS_CONFIG = os.path.join(TEST_MODEL_CONFIG_DIR, "deepspeech_mini_for_test.yml")
