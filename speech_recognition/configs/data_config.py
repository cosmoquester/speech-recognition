from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass
class DataConfig:
    """Config for audio data processing or data dependant parameter"""

    # File Format ("pcm", "wav", "flac", "mp3")
    file_format: str
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
