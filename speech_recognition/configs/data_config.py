from typing import Optional

import yaml
from pydantic.dataclasses import dataclass
from typing_extensions import Literal


@dataclass
class DataConfig:
    """Config for audio data processing or data dependant parameter"""

    # File Format ("pcm", "wav", "flac", "mp3")
    file_format: Literal["pcm", "wav", "flac", "mp3"]
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
    # Lowest frequency for mel-spectrogram
    lower_edge_hertz: Optional[float] = None
    # Largest frequency for mel-spectrogram
    upper_edge_hertz: Optional[float] = None

    @property
    def feature_dim(self):
        return 3 if self.use_delta_accelerate else 1

    @classmethod
    def from_yaml(cls, file_path) -> "DataConfig":
        with open(file_path) as f:
            return cls(**yaml.load(f, yaml.SafeLoader))
