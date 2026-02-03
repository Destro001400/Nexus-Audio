#!/usr/bin/env python3
"""
Therapeutic Music Generation Script

Generates personalized therapeutic music based on biometric data.
Uses SiMBATherapeutic model with EnCodec tokenization.

Usage:
    # Generate with biometric data
    python generate_therapeutic.py --glucose 150 --hrv 35 --output session.wav
    
    # Full 30-minute protocol
    python generate_therapeutic.py --protocol full --glucose 200 --output therapy.wav
    
    # Specific phase
    python generate_therapeutic.py --phase activation --glucose 180 --duration 300
"""

import argparse
import os
import sys
import yaml
import torch
import torchaudio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SiMBATherapeutic
from src.therapeutic.biofeedback import (
    BiofeedbackController,
    BiometricData,
    MusicParameters,
    TherapeuticPhase,
)
from src.audio.therapeutic import TherapeuticOptimizer


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: str) -> SiMBATherapeutic:
    """Load trained model from checkpoint."""
    model = SiMBATherapeutic.from_config(config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model (random weights)")
    
    model.to(device)
    model.eval()
    return model


def create_biometrics(args) -> BiometricData:
    """Create BiometricData from CLI args."""
    # Determine time of day
    hour = datetime.now().hour
    if hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    elif hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    return BiometricData(
        glucose_mg_dl=args.glucose,
        hrv_ms=args.hrv,
        heart_rate_bpm=args.heart_rate,
        anxiety_score=args.anxiety,
        time_of_day=time_of_day,
    )


def apply_therapeutic_optimizations(
    waveform: torch.Tensor,
    params: MusicParameters,
    sample_rate: int,
    config: dict,
) -> torch.Tensor:
    """Apply post-processing therapeutic optimizations."""
    therapeutic_cfg = config.get("therapeutic", {})
    
    optimizer = TherapeuticOptimizer(
        sample_rate=sample_rate,
        bass_freq_min=therapeutic_cfg.get("bass_freq_min", 50),
        bass_freq_max=therapeutic_cfg.get("bass_freq_max", 60),
        bass_boost_db=params.bass_boost_db,
        hfc_preserve=params.hfc_preserve,
    )
    
    # Apply optimizations
    optimized = optimizer.apply(waveform)
    
    # Add binaural beats if enabled
    if params.binaural_enabled:
        binaural_cfg = therapeutic_cfg.get("binaural", {})
        optimized = optimizer.add_binaural_beats(
            optimized,
            beat_frequency=params.binaural_frequency_hz,
            carrier_frequency=binaural_cfg.get("base_freq", 200),
        )
    
    return optimized


def generate_single_phase(
    model: SiMBATherapeutic,
    biometrics: BiometricData,
    params: MusicParameters,
    device: str,
) -> torch.Tensor:
    """Generate audio for a single phase."""
    print(f"  Generating {params.duration_seconds}s of audio...")
    print(f"  Parameters: BPM={params.bpm}, Bass Boost={params.bass_boost_db}dB, "
          f"Binaural={params.binaural_frequency_hz}Hz")
    
    waveform = model.generate(
        duration_seconds=params.duration_seconds,
        biometrics=biometrics,
        music_params=params,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )
    
    return waveform


def generate_full_protocol(
    model: SiMBATherapeutic,
    biometrics: BiometricData,
    config: dict,
    device: str,
) -> torch.Tensor:
    """Generate the full 30-minute therapeutic protocol."""
    controller = BiofeedbackController()
    protocol = controller.get_full_protocol_params(biometrics)
    
    waveforms = []
    
    for phase, params in protocol:
        print(f"\n[Phase: {phase.value}]")
        waveform = generate_single_phase(model, biometrics, params, device)
        
        # Apply therapeutic optimizations
        optimized = apply_therapeutic_optimizations(
            waveform, params, model.sample_rate, config
        )
        waveforms.append(optimized)
    
    # Concatenate all phases
    full_audio = torch.cat(waveforms, dim=-1)
    return full_audio


def main():
    parser = argparse.ArgumentParser(
        description="Generate therapeutic music based on biometric data"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simba_therapy.yaml",
        help="Path to config file"
    )
    
    # Biometric arguments
    parser.add_argument(
        "--glucose",
        type=float,
        default=None,
        help="Glucose level in mg/dL"
    )
    parser.add_argument(
        "--hrv",
        type=float,
        default=None,
        help="Heart Rate Variability (RMSSD) in ms"
    )
    parser.add_argument(
        "--heart-rate",
        type=float,
        default=None,
        help="Heart rate in BPM"
    )
    parser.add_argument(
        "--anxiety",
        type=float,
        default=None,
        help="Anxiety score (0-10)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["full", "custom"],
        default="custom",
        help="'full' for 30-min protocol, 'custom' for single generation"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["activation", "modulation", "consolidation"],
        default=None,
        help="Specific therapeutic phase"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300,
        help="Duration in seconds (for custom generation)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="therapeutic_session.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, args.device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create biometrics
    biometrics = create_biometrics(args)
    print(f"\nBiometric data:")
    print(f"  Glucose: {biometrics.glucose_mg_dl or 'N/A'} mg/dL")
    print(f"  HRV: {biometrics.hrv_ms or 'N/A'} ms")
    print(f"  Glucose state: {biometrics.get_glucose_state().value}")
    print(f"  Stress level: {biometrics.get_stress_level().value}")
    print(f"  Time of day: {biometrics.time_of_day}")
    
    # Generate
    print("\nGenerating therapeutic music...")
    
    if args.protocol == "full":
        waveform = generate_full_protocol(model, biometrics, config, args.device)
    else:
        # Get controller for parameters
        controller = BiofeedbackController()
        
        phase = None
        if args.phase:
            phase = TherapeuticPhase(args.phase)
        
        params = controller.compute_parameters(biometrics, current_phase=phase)
        params.duration_seconds = args.duration
        
        waveform = generate_single_phase(model, biometrics, params, args.device)
        
        # Apply optimizations
        waveform = apply_therapeutic_optimizations(
            waveform, params, model.sample_rate, config
        )
    
    # Save output
    print(f"\nSaving to {args.output}...")
    
    # Ensure waveform is 2D (channels, samples)
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    
    torchaudio.save(
        args.output,
        waveform.cpu(),
        model.sample_rate,
    )
    
    duration = waveform.shape[-1] / model.sample_rate
    print(f"Generated {duration:.1f}s of therapeutic audio")
    print("Done!")


if __name__ == "__main__":
    main()
