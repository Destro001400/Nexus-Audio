"""
Biofeedback Module for Therapeutic Music

Adapts music generation parameters based on real-time health data:
- Glucose levels (from CGM or manual input)
- Heart Rate Variability (HRV) from wearables
- Stress/anxiety indicators

Based on SnaX research: Triple Convergence Hypothesis
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import math


class TherapeuticPhase(Enum):
    """Therapeutic protocol phases."""
    ACTIVATION = "activation"      # Fase 1: 5 min, bass 50-60Hz
    MODULATION = "modulation"      # Fase 2: 15 min, HFC + binaural
    CONSOLIDATION = "consolidation"  # Fase 3: 10 min, theta waves
    EMERGENCY = "emergency"        # When glucose is critical


class GlucoseState(Enum):
    """Glucose level states."""
    CRITICAL_LOW = "critical_low"   # <54 mg/dL
    LOW = "low"                     # 54-70 mg/dL
    NORMAL = "normal"               # 70-180 mg/dL
    HIGH = "high"                   # 180-250 mg/dL
    CRITICAL_HIGH = "critical_high" # >250 mg/dL


class StressLevel(Enum):
    """Stress levels based on HRV."""
    RELAXED = "relaxed"       # High HRV
    NORMAL = "normal"         # Normal HRV
    STRESSED = "stressed"     # Low HRV
    ANXIOUS = "anxious"       # Very low HRV


@dataclass
class BiometricData:
    """Container for biometric readings."""
    glucose_mg_dl: Optional[float] = None
    hrv_ms: Optional[float] = None  # Heart Rate Variability in ms (RMSSD)
    heart_rate_bpm: Optional[float] = None
    anxiety_score: Optional[float] = None  # 0-10 scale
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening", "night"
    
    def get_glucose_state(self) -> GlucoseState:
        """Classify glucose level."""
        if self.glucose_mg_dl is None:
            return GlucoseState.NORMAL
        
        g = self.glucose_mg_dl
        if g < 54:
            return GlucoseState.CRITICAL_LOW
        elif g < 70:
            return GlucoseState.LOW
        elif g <= 180:
            return GlucoseState.NORMAL
        elif g <= 250:
            return GlucoseState.HIGH
        else:
            return GlucoseState.CRITICAL_HIGH
    
    def get_stress_level(self) -> StressLevel:
        """Classify stress based on HRV."""
        if self.hrv_ms is None:
            if self.anxiety_score is not None:
                # Use anxiety score as fallback
                if self.anxiety_score < 3:
                    return StressLevel.RELAXED
                elif self.anxiety_score < 6:
                    return StressLevel.NORMAL
                elif self.anxiety_score < 8:
                    return StressLevel.STRESSED
                else:
                    return StressLevel.ANXIOUS
            return StressLevel.NORMAL
        
        # HRV classification (RMSSD in ms)
        # Higher HRV = more relaxed
        hrv = self.hrv_ms
        if hrv > 50:
            return StressLevel.RELAXED
        elif hrv > 30:
            return StressLevel.NORMAL
        elif hrv > 20:
            return StressLevel.STRESSED
        else:
            return StressLevel.ANXIOUS


@dataclass
class MusicParameters:
    """
    Parameters for music generation based on therapeutic goals.
    
    Derived from SnaX research on music-glucose relationship.
    """
    # Tempo
    bpm: float = 80.0
    bpm_range: Tuple[float, float] = (60.0, 120.0)
    
    # Bass enhancement (50-60 Hz for insulin release)
    bass_boost_db: float = 0.0
    bass_frequency_hz: float = 55.0  # Target bass frequency
    
    # High Frequency Components (>16kHz for glucose tolerance)
    hfc_preserve: bool = True
    hfc_boost_db: float = 0.0
    
    # Binaural beats
    binaural_enabled: bool = False
    binaural_frequency_hz: float = 6.0  # Theta (4-8Hz) or Beta (12-30Hz)
    binaural_carrier_hz: float = 200.0
    
    # Musical characteristics
    key: str = "C"
    mode: str = "major"  # "major", "minor", "dorian", etc.
    energy: float = 0.5  # 0-1 scale
    
    # Instrumentation hints
    instruments: List[str] = field(default_factory=lambda: ["piano", "strings"])
    add_nature_sounds: bool = False
    
    # Duration
    duration_seconds: float = 300.0  # 5 minutes default
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for model conditioning."""
        return {
            "bpm": self.bpm,
            "bass_boost_db": self.bass_boost_db,
            "bass_frequency_hz": self.bass_frequency_hz,
            "hfc_preserve": self.hfc_preserve,
            "hfc_boost_db": self.hfc_boost_db,
            "binaural_enabled": self.binaural_enabled,
            "binaural_frequency_hz": self.binaural_frequency_hz,
            "key": self.key,
            "mode": self.mode,
            "energy": self.energy,
            "instruments": self.instruments,
            "duration_seconds": self.duration_seconds,
        }


class BiofeedbackController:
    """
    Controls music parameters based on biometric data.
    
    Implements the therapeutic protocol from SnaX research:
    - Glucose high -> Bass 50-60Hz (insulin release via mechanotransduction)
    - Anxiety/stress -> Theta binaural (6Hz) + slow tempo
    - Low energy -> Beta binaural (22Hz) + faster tempo
    
    Reference: "Music-triggered release of insulin from designer cells" (Nature, 2023)
    """
    
    def __init__(
        self,
        protocol_duration_minutes: float = 30.0,
        personalization_enabled: bool = True,
    ):
        self.protocol_duration = protocol_duration_minutes * 60  # Convert to seconds
        self.personalization_enabled = personalization_enabled
        
        # Default phase durations (total = 30 min)
        self.phase_durations = {
            TherapeuticPhase.ACTIVATION: 5 * 60,      # 5 min
            TherapeuticPhase.MODULATION: 15 * 60,     # 15 min
            TherapeuticPhase.CONSOLIDATION: 10 * 60,  # 10 min
        }
    
    def compute_parameters(
        self,
        biometrics: BiometricData,
        current_phase: Optional[TherapeuticPhase] = None,
    ) -> MusicParameters:
        """
        Compute optimal music parameters from biometric data.
        
        Args:
            biometrics: Current biometric readings
            current_phase: Override therapeutic phase
            
        Returns:
            MusicParameters optimized for user's state
        """
        glucose_state = biometrics.get_glucose_state()
        stress_level = biometrics.get_stress_level()
        
        # Start with default parameters
        params = MusicParameters()
        
        # Handle critical states first
        if glucose_state == GlucoseState.CRITICAL_LOW:
            return self._emergency_hypoglycemia_params()
        elif glucose_state == GlucoseState.CRITICAL_HIGH:
            return self._emergency_hyperglycemia_params()
        
        # Glucose-based adjustments
        if glucose_state == GlucoseState.HIGH:
            # Enhance bass for insulin release
            params.bass_boost_db = 6.0
            params.bass_frequency_hz = 55.0
            params.bpm = 70.0  # Slower for relaxation (reduces cortisol)
            params.binaural_enabled = True
            params.binaural_frequency_hz = 6.0  # Theta for parasympathetic activation
            
        elif glucose_state == GlucoseState.LOW:
            # Stimulating music to raise alertness
            params.bpm = 100.0
            params.energy = 0.7
            params.binaural_enabled = True
            params.binaural_frequency_hz = 22.0  # Beta for alertness
            params.mode = "major"
        
        # Stress-based adjustments
        if stress_level == StressLevel.ANXIOUS:
            params.bpm = min(params.bpm, 65.0)
            params.binaural_enabled = True
            params.binaural_frequency_hz = 6.0  # Theta
            params.add_nature_sounds = True
            params.instruments = ["piano", "strings", "ambient"]
            params.energy = 0.3
            
        elif stress_level == StressLevel.STRESSED:
            params.bpm = min(params.bpm, 75.0)
            params.energy = 0.4
            params.mode = "dorian"  # Minor-ish but not sad
        
        elif stress_level == StressLevel.RELAXED:
            # User is already relaxed, can be more energetic if needed
            if glucose_state == GlucoseState.NORMAL:
                params.bpm = 85.0
                params.energy = 0.6
        
        # Time of day adjustments
        if biometrics.time_of_day == "morning":
            params.bpm = max(params.bpm, 80.0)
            params.energy = max(params.energy, 0.5)
            params.binaural_frequency_hz = 15.0  # Low beta for focus
            
        elif biometrics.time_of_day == "night":
            params.bpm = min(params.bpm, 65.0)
            params.energy = min(params.energy, 0.3)
            params.binaural_frequency_hz = 4.0  # Low theta for sleep prep
        
        # Apply phase-specific modifications
        if current_phase:
            params = self._apply_phase_modifications(params, current_phase)
        
        return params
    
    def _emergency_hypoglycemia_params(self) -> MusicParameters:
        """Parameters for critical low glucose."""
        return MusicParameters(
            bpm=110.0,
            energy=0.8,
            binaural_enabled=True,
            binaural_frequency_hz=25.0,  # Beta for alertness
            mode="major",
            instruments=["drums", "brass", "synth"],
            bass_boost_db=0.0,  # Don't enhance bass (could trigger more insulin)
            duration_seconds=60.0,  # Short alert
        )
    
    def _emergency_hyperglycemia_params(self) -> MusicParameters:
        """Parameters for critical high glucose."""
        return MusicParameters(
            bpm=65.0,
            energy=0.3,
            binaural_enabled=True,
            binaural_frequency_hz=6.0,  # Theta for deep relaxation
            bass_boost_db=8.0,  # Strong bass enhancement
            bass_frequency_hz=55.0,
            hfc_preserve=True,
            hfc_boost_db=3.0,
            mode="dorian",
            instruments=["bass", "piano", "strings", "ambient"],
            add_nature_sounds=True,
            duration_seconds=1800.0,  # 30 min protocol
        )
    
    def _apply_phase_modifications(
        self,
        params: MusicParameters,
        phase: TherapeuticPhase,
    ) -> MusicParameters:
        """Apply phase-specific modifications to parameters."""
        
        if phase == TherapeuticPhase.ACTIVATION:
            # Phase 1: Bass enhancement for insulin
            params.bass_boost_db = max(params.bass_boost_db, 6.0)
            params.duration_seconds = self.phase_durations[phase]
            
        elif phase == TherapeuticPhase.MODULATION:
            # Phase 2: HFC + binaural
            params.hfc_preserve = True
            params.hfc_boost_db = 3.0
            params.binaural_enabled = True
            params.instruments = ["piano", "strings", "nature"]
            params.duration_seconds = self.phase_durations[phase]
            
        elif phase == TherapeuticPhase.CONSOLIDATION:
            # Phase 3: Deep relaxation
            params.bpm = min(params.bpm, 60.0)
            params.energy = 0.2
            params.binaural_enabled = True
            params.binaural_frequency_hz = 6.0  # Theta
            params.add_nature_sounds = True
            params.duration_seconds = self.phase_durations[phase]
        
        return params
    
    def get_full_protocol_params(
        self,
        biometrics: BiometricData,
    ) -> List[Tuple[TherapeuticPhase, MusicParameters]]:
        """
        Get parameters for the full 30-minute protocol.
        
        Returns list of (phase, parameters) tuples.
        """
        phases = [
            TherapeuticPhase.ACTIVATION,
            TherapeuticPhase.MODULATION,
            TherapeuticPhase.CONSOLIDATION,
        ]
        
        protocol = []
        for phase in phases:
            params = self.compute_parameters(biometrics, current_phase=phase)
            protocol.append((phase, params))
        
        return protocol


class BiofeedbackEmbedding(nn.Module):
    """
    Converts biometric data to conditioning embeddings for the model.
    
    These embeddings are added to the audio token embeddings to
    condition generation on the user's physiological state.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Glucose embedding
        self.glucose_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        
        # HRV embedding
        self.hrv_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        
        # Stress level embedding (categorical)
        self.stress_embed = nn.Embedding(len(StressLevel), d_model)
        
        # Time of day embedding
        self.time_embed = nn.Embedding(4, d_model)  # morning, afternoon, evening, night
        
        # Combined projection
        self.combine = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
    
    def forward(
        self,
        glucose: Optional[torch.Tensor] = None,
        hrv: Optional[torch.Tensor] = None,
        stress_level: Optional[torch.Tensor] = None,
        time_of_day: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute biofeedback conditioning embedding.
        
        Args:
            glucose: Glucose values (batch,) normalized to 0-1
            hrv: HRV values (batch,) normalized to 0-1
            stress_level: Stress level indices (batch,)
            time_of_day: Time of day indices (batch,)
            
        Returns:
            embedding: (batch, d_model) conditioning vector
        """
        batch_size = 1
        if glucose is not None:
            batch_size = glucose.shape[0]
        elif hrv is not None:
            batch_size = hrv.shape[0]
        
        device = next(self.parameters()).device
        
        # Get individual embeddings
        if glucose is not None:
            glucose_emb = self.glucose_embed(glucose.unsqueeze(-1))
        else:
            glucose_emb = torch.zeros(batch_size, self.d_model, device=device)
        
        if hrv is not None:
            hrv_emb = self.hrv_embed(hrv.unsqueeze(-1))
        else:
            hrv_emb = torch.zeros(batch_size, self.d_model, device=device)
        
        if stress_level is not None:
            stress_emb = self.stress_embed(stress_level)
        else:
            stress_emb = torch.zeros(batch_size, self.d_model, device=device)
        
        if time_of_day is not None:
            time_emb = self.time_embed(time_of_day)
        else:
            time_emb = torch.zeros(batch_size, self.d_model, device=device)
        
        # Combine
        combined = torch.cat([glucose_emb, hrv_emb, stress_emb, time_emb], dim=-1)
        embedding = self.combine(combined)
        
        return embedding
    
    @staticmethod
    def normalize_glucose(glucose_mg_dl: float) -> float:
        """Normalize glucose to 0-1 range (0-400 mg/dL)."""
        return min(max(glucose_mg_dl / 400.0, 0.0), 1.0)
    
    @staticmethod
    def normalize_hrv(hrv_ms: float) -> float:
        """Normalize HRV to 0-1 range (0-100 ms)."""
        return min(max(hrv_ms / 100.0, 0.0), 1.0)
