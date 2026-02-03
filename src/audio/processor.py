"""
Audio Processor for SiMBA

Handles audio loading, preprocessing, and augmentation.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


class AudioProcessor:
    """
    Audio processing utilities for training and inference.
    
    Handles:
    - Loading audio files
    - Resampling to target sample rate
    - Normalization
    - Chunking for training
    - Data augmentation (pitch shift, time stretch)
    
    Args:
        sample_rate: Target sample rate
        chunk_length_sec: Length of audio chunks in seconds
        normalize: Whether to normalize audio
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_length_sec: float = 10.0,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.chunk_length_samples = int(sample_rate * chunk_length_sec)
        self.normalize = normalize
        
    def load_audio(
        self,
        path: str,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if needed.
        
        Args:
            path: Path to audio file
            start_sec: Start time in seconds
            duration_sec: Duration in seconds
            
        Returns:
            waveform: (channels, samples)
            sample_rate: Original sample rate
        """
        # Get metadata for smart loading
        info = torchaudio.info(path)
        
        # Calculate frame offset and num_frames if specified
        frame_offset = 0
        num_frames = -1
        
        if start_sec is not None:
            frame_offset = int(start_sec * info.sample_rate)
        if duration_sec is not None:
            num_frames = int(duration_sec * info.sample_rate)
            
        # Load audio
        waveform, sr = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Normalize
        if self.normalize:
            waveform = self.normalize_audio(waveform)
            
        return waveform, self.sample_rate
    
    def normalize_audio(
        self,
        waveform: torch.Tensor,
        target_db: float = -20.0,
    ) -> torch.Tensor:
        """
        Normalize audio to target loudness.
        
        Args:
            waveform: Input audio
            target_db: Target loudness in dB
            
        Returns:
            Normalized waveform
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(waveform ** 2))
        
        # Avoid division by zero
        if rms < 1e-8:
            return waveform
            
        # Calculate gain
        target_rms = 10 ** (target_db / 20)
        gain = target_rms / rms
        
        # Apply gain with soft clipping
        waveform = waveform * gain
        waveform = torch.tanh(waveform)
        
        return waveform
    
    def chunk_audio(
        self,
        waveform: torch.Tensor,
        overlap_ratio: float = 0.5,
    ) -> List[torch.Tensor]:
        """
        Split audio into overlapping chunks for training.
        
        Args:
            waveform: Input audio (1, samples)
            overlap_ratio: Overlap between chunks (0.0-1.0)
            
        Returns:
            List of chunks, each (1, chunk_length_samples)
        """
        waveform = waveform.squeeze(0)  # (samples,)
        total_samples = waveform.shape[0]
        
        # Calculate hop length
        hop_length = int(self.chunk_length_samples * (1 - overlap_ratio))
        
        chunks = []
        start = 0
        
        while start + self.chunk_length_samples <= total_samples:
            chunk = waveform[start:start + self.chunk_length_samples]
            chunks.append(chunk.unsqueeze(0))
            start += hop_length
            
        # Handle last chunk (pad if needed)
        if start < total_samples:
            chunk = waveform[start:]
            padding = self.chunk_length_samples - chunk.shape[0]
            chunk = F.pad(chunk, (0, padding))
            chunks.append(chunk.unsqueeze(0))
            
        return chunks
    
    def augment(
        self,
        waveform: torch.Tensor,
        pitch_shift: bool = True,
        time_stretch: bool = True,
        noise_injection: bool = False,
    ) -> torch.Tensor:
        """
        Apply data augmentation to audio.
        
        Args:
            waveform: Input audio
            pitch_shift: Apply random pitch shift
            time_stretch: Apply random time stretch
            noise_injection: Add background noise
            
        Returns:
            Augmented waveform
        """
        if pitch_shift and torch.rand(1) > 0.5:
            # Random pitch shift (-2 to +2 semitones)
            steps = (torch.rand(1) * 4 - 2).item()
            waveform = self._pitch_shift(waveform, steps)
            
        if time_stretch and torch.rand(1) > 0.5:
            # Random time stretch (0.9x to 1.1x)
            rate = 0.9 + torch.rand(1).item() * 0.2
            waveform = self._time_stretch(waveform, rate)
            
        if noise_injection and torch.rand(1) > 0.7:
            # Add small amount of noise
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
        return waveform
    
    def _pitch_shift(
        self,
        waveform: torch.Tensor,
        semitones: float,
    ) -> torch.Tensor:
        """Apply pitch shift using resampling trick."""
        # Calculate rate for pitch shift
        rate = 2 ** (-semitones / 12)
        
        # Resample to change pitch
        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            int(self.sample_rate * rate),
        )
        stretched = resampler(waveform)
        
        # Resample back to original sample rate
        resampler_back = torchaudio.transforms.Resample(
            int(self.sample_rate * rate),
            self.sample_rate,
        )
        return resampler_back(stretched)
    
    def _time_stretch(
        self,
        waveform: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        """Apply time stretch using interpolation."""
        # Calculate new length
        new_length = int(waveform.shape[-1] / rate)
        
        # Interpolate
        stretched = F.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False,
        ).squeeze(0)
        
        return stretched
    
    def save_audio(
        self,
        waveform: torch.Tensor,
        path: str,
        sample_rate: Optional[int] = None,
    ):
        """
        Save audio to file.
        
        Args:
            waveform: Audio tensor
            path: Output path
            sample_rate: Sample rate (uses default if None)
        """
        sr = sample_rate or self.sample_rate
        
        # Ensure 2D tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Clamp to valid range
        waveform = waveform.clamp(-1, 1)
        
        torchaudio.save(path, waveform, sr)


def get_audio_duration(path: str) -> float:
    """Get duration of audio file in seconds."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def find_audio_files(
    directory: str,
    extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
) -> List[Path]:
    """Find all audio files in directory recursively."""
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
        
    return sorted(files)
