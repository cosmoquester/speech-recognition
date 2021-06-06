from typing import Optional

import yaml
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..data import make_log_mel_spectrogram, make_mfcc, make_spectrogram


@dataclass
class DataConfig:
    """Config for audio data processing or data dependant parameter"""

    # File Format
    file_format: Literal["pcm", "wav", "flac", "mp3"]
    # Audio Feature Type
    audio_feature_type: Literal["spectrogram", "log-mel-spectrogram", "mfcc"]
    # Audio Sample rate
    sample_rate: int
    # Frame Length for STFT
    frame_length: int
    # Frame Step for STFT
    frame_step: int
    # FFT Length for STFT
    fft_length: int
    # Max audio log mel spectrogram sequence length
    max_audio_length: int
    # Max token sequence length
    max_token_length: int
    # Use delta and deltas accelerate
    use_delta_accelerate: bool
    # Number of Mel bins for mel-spectrogram
    num_mel_bins: Optional[int] = None
    # Number of mfcc feature
    num_mfcc: Optional[int] = None
    # Lowest frequency for mel-spectrogram
    lower_edge_hertz: Optional[float] = None
    # Largest frequency for mel-spectrogram
    upper_edge_hertz: Optional[float] = None

    def __post_init__(self):
        if self.audio_feature_type in ["log-mel-spectrogram", "mfcc"]:
            assert all(
                [self.num_mel_bins, self.lower_edge_hertz, self.upper_edge_hertz]
            ), '"num_mel_bins", "lower_edge_hertz", "upper_edge_hertz" is required'
        if self.audio_feature_type == "mfcc":
            assert self.num_mfcc, '"num_mfcc" is required'

    @property
    def feature_dim(self):
        return 3 if self.use_delta_accelerate else 1

    @property
    def frequency_dim(self):
        if self.audio_feature_type == "spectrogram":
            return self.fft_length // 2 + 1
        if self.audio_feature_type == "log-mel-spectrogram":
            return self.num_mel_bins
        if self.audio_feature_type == "mfcc":
            return self.num_mfcc

    @property
    def audio_feature_fn(self):
        if self.audio_feature_type == "spectrogram":
            return make_spectrogram(self.frame_length, self.frame_step, self.fft_length)
        if self.audio_feature_type == "log-mel-spectrogram":
            return make_log_mel_spectrogram(
                self.sample_rate,
                self.frame_length,
                self.frame_step,
                self.fft_length,
                self.num_mel_bins,
                self.lower_edge_hertz,
                self.upper_edge_hertz,
            )
        if self.audio_feature_type == "mfcc":
            return make_mfcc(
                self.sample_rate,
                self.frame_length,
                self.frame_step,
                self.fft_length,
                self.num_mel_bins,
                self.num_mfcc,
                self.lower_edge_hertz,
                self.upper_edge_hertz,
            )

    @classmethod
    def from_yaml(cls, file_path) -> "DataConfig":
        with open(file_path) as f:
            return cls(**yaml.load(f, yaml.SafeLoader))
