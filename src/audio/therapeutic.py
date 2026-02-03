"""
Therapeutic Audio Optimization for SiMBA

Implements frequency optimization based on scientific research:
1. Bass enhancement (50-60 Hz) for insulin release (ETH Zurich study)
2. High-frequency component preservation (16-32 kHz) for glucose tolerance
3. Binaural beats for brain entrainment (Theta/Beta)

References:
- "Music-triggered release of insulin from designer cells" - Nature 2023
- "Effect of sounds containing high-frequency components on glucose tolerance" - Scientific Reports 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import signal


class TherapeuticOptimizer(nn.Module):
    """
    Optimizes generated audio for therapeutic effects.
    
    Applies frequency-specific enhancements based on research:
    - Bass boost at 50-60 Hz for mechanotransduction
    - HFC preservation for neural modulation
    - Optional binaural beats injection
    
    Args:
        sample_rate: Audio sample rate
        bass_freq_min: Minimum bass frequency (Hz)
        bass_freq_max: Maximum bass frequency (Hz)
        bass_boost_db: Bass boost amount (dB)
        hfc_freq_min: Minimum HFC frequency (Hz)
        hfc_preserve: Whether to preserve HFC
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        bass_freq_min: float = 50.0,
        bass_freq_max: float = 60.0,
        bass_boost_db: float = 6.0,
        hfc_freq_min: float = 16000.0,
        hfc_preserve: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.bass_freq_min = bass_freq_min
        self.bass_freq_max = bass_freq_max
        self.bass_boost_db = bass_boost_db
        self.hfc_freq_min = hfc_freq_min
        self.hfc_preserve = hfc_preserve
        
        # Nyquist frequency
        self.nyquist = sample_rate / 2
        
        # Pre-compute filter coefficients
        self._init_filters()
        
    def _init_filters(self):
        """Initialize filter coefficients for bass and HFC processing."""
        # Bass bandpass filter (50-60 Hz)
        bass_low = self.bass_freq_min / self.nyquist
        bass_high = self.bass_freq_max / self.nyquist
        
        # Ensure valid range
        bass_low = max(0.001, min(bass_low, 0.99))
        bass_high = max(bass_low + 0.001, min(bass_high, 0.99))
        
        self.bass_b, self.bass_a = signal.butter(
            4, [bass_low, bass_high], btype='band'
        )
        
        # HFC highpass filter (> 16 kHz)
        hfc_cutoff = min(self.hfc_freq_min / self.nyquist, 0.95)
        hfc_cutoff = max(0.01, hfc_cutoff)
        
        self.hfc_b, self.hfc_a = signal.butter(
            2, hfc_cutoff, btype='high'
        )
        
    def forward(
        self,
        waveform: torch.Tensor,
        enhance_bass: bool = True,
        preserve_hfc: bool = True,
    ) -> torch.Tensor:
        """
        Apply therapeutic optimization to audio.
        
        Args:
            waveform: Input audio (batch, samples) or (batch, 1, samples)
            enhance_bass: Apply bass enhancement
            preserve_hfc: Preserve high-frequency components
            
        Returns:
            Optimized waveform
        """
        # Handle dimensions
        squeeze_output = False
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            squeeze_output = True
            
        output = waveform.clone()
        
        if enhance_bass:
            output = self._enhance_bass(output)
            
        if preserve_hfc and self.hfc_preserve:
            output = self._preserve_hfc(waveform, output)
            
        if squeeze_output:
            output = output.squeeze(1)
            
        return output
    
    def _enhance_bass(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Enhance bass frequencies (50-60 Hz) for insulin release effect.
        
        Based on ETH Zurich study showing mechanotransduction at these frequencies.
        """
        device = waveform.device
        batch, channels, samples = waveform.shape
        
        # Convert to numpy for filtering
        waveform_np = waveform.cpu().numpy()
        
        enhanced = np.zeros_like(waveform_np)
        
        for b in range(batch):
            for c in range(channels):
                # Extract bass component
                bass = signal.filtfilt(
                    self.bass_b, self.bass_a, waveform_np[b, c]
                )
                
                # Calculate boost gain
                boost_gain = 10 ** (self.bass_boost_db / 20) - 1
                
                # Add boosted bass to original
                enhanced[b, c] = waveform_np[b, c] + bass * boost_gain
                
        return torch.from_numpy(enhanced).to(device).float()
    
    def _preserve_hfc(
        self,
        original: torch.Tensor,
        processed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preserve high-frequency components from original audio.
        
        HFC (>16 kHz) are important for glucose tolerance effects.
        """
        device = original.device
        batch, channels, samples = original.shape
        
        original_np = original.cpu().numpy()
        processed_np = processed.cpu().numpy()
        
        result = np.zeros_like(processed_np)
        
        for b in range(batch):
            for c in range(channels):
                # Extract HFC from original
                hfc = signal.filtfilt(
                    self.hfc_b, self.hfc_a, original_np[b, c]
                )
                
                # Remove HFC from processed (if any)
                processed_low = original_np[b, c] - hfc
                
                # Add original HFC to processed
                result[b, c] = processed_np[b, c] + hfc * 0.5
                
        return torch.from_numpy(result).to(device).float()
    
    def generate_binaural_beats(
        self,
        duration_sec: float,
        beat_frequency: float = 6.0,
        carrier_frequency: float = 200.0,
        volume: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate binaural beats for brain entrainment.
        
        Creates stereo audio with slightly different frequencies in each ear,
        causing perceived beating at the difference frequency.
        
        Therapeutic frequencies:
        - Theta (4-8 Hz): Relaxation, meditation
        - Beta (12-30 Hz): Alertness, anxiety reduction
        
        Args:
            duration_sec: Duration in seconds
            beat_frequency: Desired beat frequency (Hz)
            carrier_frequency: Base carrier frequency (Hz)
            volume: Output volume (0-1)
            
        Returns:
            left_channel: (samples,)
            right_channel: (samples,)
        """
        samples = int(duration_sec * self.sample_rate)
        t = torch.linspace(0, duration_sec, samples)
        
        # Left ear: carrier frequency
        left = torch.sin(2 * np.pi * carrier_frequency * t)
        
        # Right ear: carrier + beat frequency
        right = torch.sin(2 * np.pi * (carrier_frequency + beat_frequency) * t)
        
        # Apply volume
        left = left * volume
        right = right * volume
        
        return left, right
    
    def mix_binaural_with_music(
        self,
        music: torch.Tensor,
        beat_frequency: float = 6.0,
        carrier_frequency: float = 200.0,
        binaural_volume: float = 0.15,
    ) -> torch.Tensor:
        """
        Mix binaural beats with music for therapeutic session.
        
        Args:
            music: Mono music (samples,) or (1, samples)
            beat_frequency: Binaural frequency
            carrier_frequency: Carrier frequency
            binaural_volume: Volume of binaural component
            
        Returns:
            Stereo output (2, samples)
        """
        if music.dim() == 2:
            music = music.squeeze(0)
            
        duration_sec = len(music) / self.sample_rate
        
        # Generate binaural
        left_bin, right_bin = self.generate_binaural_beats(
            duration_sec=duration_sec,
            beat_frequency=beat_frequency,
            carrier_frequency=carrier_frequency,
            volume=binaural_volume,
        )
        
        # Mix with music (centered in stereo)
        music_volume = 1.0 - binaural_volume
        left = music * music_volume + left_bin
        right = music * music_volume + right_bin
        
        # Stack to stereo
        stereo = torch.stack([left, right], dim=0)
        
        # Normalize to prevent clipping
        max_val = stereo.abs().max()
        if max_val > 1.0:
            stereo = stereo / max_val
            
        return stereo


class TherapeuticProtocol:
    """
    Implements the therapeutic protocol from the research.
    
    Protocol (30 minutes):
    - Phase 1 (5 min): Activation - rock/pop with bass 50-60 Hz
    - Phase 2 (15 min): Modulation - instrumental with HFC
    - Phase 3 (10 min): Consolidation - ambient with theta binaural
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        optimizer: Optional[TherapeuticOptimizer] = None,
    ):
        self.sample_rate = sample_rate
        self.optimizer = optimizer or TherapeuticOptimizer(sample_rate)
        
        # Phase durations in seconds
        self.phase_durations = {
            "activation": 5 * 60,      # 5 minutes
            "modulation": 15 * 60,     # 15 minutes  
            "consolidation": 10 * 60,  # 10 minutes
        }
        
        # Phase-specific settings
        self.phase_settings = {
            "activation": {
                "bass_boost_db": 6.0,
                "binaural_freq": None,  # No binaural
            },
            "modulation": {
                "bass_boost_db": 3.0,
                "binaural_freq": 22.0,  # Beta for alertness
            },
            "consolidation": {
                "bass_boost_db": 0.0,
                "binaural_freq": 6.0,   # Theta for relaxation
            },
        }
        
    def get_phase(self, elapsed_seconds: float) -> str:
        """Get current phase based on elapsed time."""
        if elapsed_seconds < self.phase_durations["activation"]:
            return "activation"
        elif elapsed_seconds < (
            self.phase_durations["activation"] + 
            self.phase_durations["modulation"]
        ):
            return "modulation"
        else:
            return "consolidation"
            
    def process_for_phase(
        self,
        audio: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        """
        Process audio according to phase-specific settings.
        
        Args:
            audio: Input audio
            phase: Current phase name
            
        Returns:
            Processed audio
        """
        settings = self.phase_settings.get(phase, {})
        
        # Update optimizer settings
        self.optimizer.bass_boost_db = settings.get("bass_boost_db", 3.0)
        
        # Apply therapeutic optimization
        processed = self.optimizer(audio)
        
        # Add binaural if specified
        binaural_freq = settings.get("binaural_freq")
        if binaural_freq is not None:
            processed = self.optimizer.mix_binaural_with_music(
                processed,
                beat_frequency=binaural_freq,
            )
            
        return processed
