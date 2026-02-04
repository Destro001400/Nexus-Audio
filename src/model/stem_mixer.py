"""
Stem Mixer for Therapeutic Audio

Mixes individual stems with therapeutic optimizations applied per-stem.
Allows precise control over each element for maximum therapeutic effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .multi_track_generator import StemType, StemOutput, StemConfig


@dataclass
class TherapeuticMixSettings:
    """Settings for therapeutic stem mixing."""
    # Bass stem (insulin release)
    bass_boost_db: float = 6.0
    bass_target_freq: float = 55.0  # Hz - optimal for mechanotransduction
    bass_enhance_width: float = 10.0  # Hz bandwidth
    
    # Drums stem (BPM control)
    drums_gain_db: float = 0.0
    
    # Melody stem (HFC preservation)
    melody_gain_db: float = 0.0
    melody_hfc_boost_db: float = 3.0  # Boost >8kHz for glucose tolerance
    melody_hfc_freq: float = 8000.0   # Hz
    
    # Ambient stem (binaural beats)
    ambient_gain_db: float = -6.0
    binaural_enabled: bool = True
    binaural_frequency: float = 6.0   # Hz (theta for relaxation)
    binaural_carrier: float = 200.0   # Hz carrier
    binaural_amplitude: float = 0.3   # Relative to ambient
    
    # Master
    master_limiter: bool = True
    master_ceiling_db: float = -1.0
    
    @classmethod
    def for_high_glucose(cls) -> "TherapeuticMixSettings":
        """Settings optimized for high glucose (>180 mg/dL)."""
        return cls(
            bass_boost_db=8.0,
            bass_target_freq=55.0,
            binaural_enabled=True,
            binaural_frequency=6.0,  # Theta
            ambient_gain_db=-3.0,
        )
    
    @classmethod
    def for_low_glucose(cls) -> "TherapeuticMixSettings":
        """Settings for low glucose (<70 mg/dL) - more energetic."""
        return cls(
            bass_boost_db=2.0,
            drums_gain_db=3.0,
            melody_gain_db=2.0,
            binaural_enabled=True,
            binaural_frequency=22.0,  # Beta for alertness
            ambient_gain_db=-9.0,
        )
    
    @classmethod
    def for_stress_reduction(cls) -> "TherapeuticMixSettings":
        """Settings for stress/anxiety reduction."""
        return cls(
            bass_boost_db=4.0,
            drums_gain_db=-3.0,  # Less drums
            melody_gain_db=0.0,
            binaural_enabled=True,
            binaural_frequency=6.0,  # Theta
            ambient_gain_db=0.0,  # More ambient
        )


class StemProcessor(nn.Module):
    """Processes individual stems with therapeutic optimizations."""
    
    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
    
    def apply_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain in dB."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def apply_bass_enhancement(
        self,
        audio: torch.Tensor,
        target_freq: float,
        bandwidth: float,
        boost_db: float,
    ) -> torch.Tensor:
        """
        Enhance bass frequencies around target using parametric EQ.
        
        Args:
            audio: Input waveform
            target_freq: Center frequency (e.g., 55 Hz)
            bandwidth: Width of boost
            boost_db: Amount of boost
        """
        if not SCIPY_AVAILABLE:
            # Fallback: simple low shelf
            return self.apply_gain(audio, boost_db * 0.5)
        
        # Design parametric EQ filter
        nyq = self.sample_rate / 2
        low = max((target_freq - bandwidth / 2) / nyq, 0.001)
        high = min((target_freq + bandwidth / 2) / nyq, 0.999)
        
        # Bandpass to isolate frequencies
        b, a = signal.butter(2, [low, high], btype='band')
        
        # Process each batch
        audio_np = audio.cpu().numpy()
        enhanced = signal.filtfilt(b, a, audio_np)
        
        # Add boosted band to original
        boost_linear = 10 ** (boost_db / 20) - 1
        result = audio_np + enhanced * boost_linear
        
        return torch.from_numpy(result).to(audio.device)
    
    def apply_hfc_boost(
        self,
        audio: torch.Tensor,
        cutoff_freq: float,
        boost_db: float,
    ) -> torch.Tensor:
        """
        Boost high frequency components above cutoff.
        
        HFC preservation important for glucose tolerance effect.
        """
        if not SCIPY_AVAILABLE:
            return audio
        
        nyq = self.sample_rate / 2
        if cutoff_freq >= nyq:
            return audio
        
        # Highpass filter
        normalized_cutoff = cutoff_freq / nyq
        b, a = signal.butter(2, normalized_cutoff, btype='high')
        
        audio_np = audio.cpu().numpy()
        hfc = signal.filtfilt(b, a, audio_np)
        
        # Add boosted HFC
        boost_linear = 10 ** (boost_db / 20) - 1
        result = audio_np + hfc * boost_linear
        
        return torch.from_numpy(result).to(audio.device)
    
    def generate_binaural_beat(
        self,
        duration_samples: int,
        beat_frequency: float,
        carrier_frequency: float,
        amplitude: float,
    ) -> torch.Tensor:
        """
        Generate binaural beat (requires stereo playback with headphones).
        
        Left and right channels have slightly different frequencies,
        the brain perceives the difference as a "beat."
        
        Args:
            duration_samples: Number of samples
            beat_frequency: Desired beat frequency (e.g., 6 Hz for theta)
            carrier_frequency: Base frequency (e.g., 200 Hz)
            amplitude: Volume (0-1)
        """
        t = torch.arange(duration_samples) / self.sample_rate
        
        # Left channel: carrier frequency
        left_freq = carrier_frequency
        left = torch.sin(2 * math.pi * left_freq * t) * amplitude
        
        # Right channel: carrier + beat frequency
        right_freq = carrier_frequency + beat_frequency
        right = torch.sin(2 * math.pi * right_freq * t) * amplitude
        
        # Stack as stereo (2, samples)
        return torch.stack([left, right], dim=0)


class TherapeuticStemMixer(nn.Module):
    """
    Mixes stems with therapeutic processing applied to each.
    
    Processing order:
    1. Apply stem-specific processing (bass boost, HFC, etc.)
    2. Apply stem gains
    3. Mix all stems
    4. Add binaural beats (stereo)
    5. Apply master limiter
    """
    
    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.processor = StemProcessor(sample_rate)
    
    def mix(
        self,
        stem_output: StemOutput,
        settings: Optional[TherapeuticMixSettings] = None,
    ) -> torch.Tensor:
        """
        Mix stems with therapeutic processing.
        
        Args:
            stem_output: Output from MultiTrackGenerator
            settings: Therapeutic mix settings
            
        Returns:
            Mixed stereo waveform (2, samples)
        """
        if settings is None:
            settings = TherapeuticMixSettings()
        
        processed_stems = {}
        max_length = 0
        
        # Process each stem
        for stem_type, waveform in stem_output.stems.items():
            processed = self._process_stem(waveform, stem_type, settings)
            processed_stems[stem_type] = processed
            max_length = max(max_length, processed.shape[-1])
        
        # Pad all to same length
        for stem_type in processed_stems:
            current_length = processed_stems[stem_type].shape[-1]
            if current_length < max_length:
                pad_amount = max_length - current_length
                processed_stems[stem_type] = F.pad(
                    processed_stems[stem_type], (0, pad_amount)
                )
        
        # Mix all stems
        mixed = None
        for stem_waveform in processed_stems.values():
            if mixed is None:
                mixed = stem_waveform
            else:
                mixed = mixed + stem_waveform
        
        # Ensure stereo
        if mixed.dim() == 2 and mixed.shape[0] == 1:
            mixed = mixed.repeat(2, 1)
        elif mixed.dim() == 1:
            mixed = mixed.unsqueeze(0).repeat(2, 1)
        
        # Add binaural beats if enabled
        if settings.binaural_enabled:
            binaural = self.processor.generate_binaural_beat(
                duration_samples=mixed.shape[-1],
                beat_frequency=settings.binaural_frequency,
                carrier_frequency=settings.binaural_carrier,
                amplitude=settings.binaural_amplitude,
            )
            binaural = binaural.to(mixed.device)
            mixed = mixed + binaural
        
        # Apply master limiter
        if settings.master_limiter:
            mixed = self._apply_limiter(mixed, settings.master_ceiling_db)
        
        return mixed
    
    def _process_stem(
        self,
        waveform: torch.Tensor,
        stem_type: StemType,
        settings: TherapeuticMixSettings,
    ) -> torch.Tensor:
        """Apply stem-specific processing."""
        
        # Ensure correct shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2 and waveform.shape[0] > 2:
            waveform = waveform[0:1]  # Take first channel
        
        if stem_type == StemType.BASS:
            # Bass enhancement for insulin release
            waveform = self.processor.apply_bass_enhancement(
                waveform,
                target_freq=settings.bass_target_freq,
                bandwidth=settings.bass_enhance_width,
                boost_db=settings.bass_boost_db,
            )
            
        elif stem_type == StemType.DRUMS:
            # Just gain for drums
            waveform = self.processor.apply_gain(waveform, settings.drums_gain_db)
            
        elif stem_type == StemType.MELODY:
            # HFC boost for glucose tolerance
            waveform = self.processor.apply_hfc_boost(
                waveform,
                cutoff_freq=settings.melody_hfc_freq,
                boost_db=settings.melody_hfc_boost_db,
            )
            waveform = self.processor.apply_gain(waveform, settings.melody_gain_db)
            
        elif stem_type == StemType.AMBIENT:
            # Ambient stem - carrier for binaural
            waveform = self.processor.apply_gain(waveform, settings.ambient_gain_db)
        
        else:
            # Unknown stem, pass through
            pass
        
        return waveform
    
    def _apply_limiter(
        self,
        audio: torch.Tensor,
        ceiling_db: float,
    ) -> torch.Tensor:
        """
        Simple brickwall limiter to prevent clipping.
        """
        ceiling_linear = 10 ** (ceiling_db / 20)
        
        # Find peaks
        max_val = audio.abs().max()
        
        if max_val > ceiling_linear:
            # Apply compression
            audio = audio * (ceiling_linear / max_val)
        
        return audio


def get_mix_settings_from_biometrics(
    glucose_mg_dl: Optional[float] = None,
    hrv_ms: Optional[float] = None,
    stress_level: Optional[str] = None,
) -> TherapeuticMixSettings:
    """
    Get optimal mix settings based on biometric data.
    
    Args:
        glucose_mg_dl: Blood glucose level
        hrv_ms: Heart rate variability
        stress_level: "relaxed", "normal", "stressed", "anxious"
    """
    settings = TherapeuticMixSettings()
    
    # Glucose-based adjustments
    if glucose_mg_dl is not None:
        if glucose_mg_dl > 180:
            # High glucose - maximize bass for insulin
            settings = TherapeuticMixSettings.for_high_glucose()
        elif glucose_mg_dl < 70:
            # Low glucose - energetic mix
            settings = TherapeuticMixSettings.for_low_glucose()
    
    # Override with stress settings if highly stressed
    if stress_level in ["stressed", "anxious"]:
        stress_settings = TherapeuticMixSettings.for_stress_reduction()
        # Merge: keep glucose bass boost but add stress binaural
        settings.binaural_enabled = True
        settings.binaural_frequency = stress_settings.binaural_frequency
        settings.drums_gain_db = stress_settings.drums_gain_db
        settings.ambient_gain_db = stress_settings.ambient_gain_db
    
    return settings
