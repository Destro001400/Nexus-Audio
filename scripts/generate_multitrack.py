#!/usr/bin/env python3
"""
Multi-Track Therapeutic Music Generation Script

Generates music as separate stems (bass, drums, melody, ambient)
and mixes them with therapeutic optimizations.

Usage:
    # Generate with all stems
    python generate_multitrack.py --glucose 180 --output session.wav
    
    # Generate only specific stems
    python generate_multitrack.py --stems bass,melody --output bass_melody.wav
    
    # Export individual stems
    python generate_multitrack.py --export-stems --output-dir ./stems/
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

import torch
import torchaudio

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    SiMBATherapeutic,
    MultiTrackGenerator,
    TherapeuticStemMixer,
    TherapeuticMixSettings,
    StemType,
    get_mix_settings_from_biometrics,
)
from src.therapeutic.biofeedback import (
    BiometricData,
    BiofeedbackController,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: str,
    config: dict,
    device: str,
) -> tuple:
    """Load SiMBA model and multi-track generator."""
    # Load backbone model
    model = SiMBATherapeutic.from_config(config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint at {checkpoint_path}, using random weights")
    
    model.to(device)
    model.eval()
    
    # Create multi-track generator
    multi_track = MultiTrackGenerator(
        d_model=config.get("model", {}).get("d_model", 512),
        vocab_size=model.vocab_size,
        n_codebooks=model.n_codebooks,
        sample_rate=model.sample_rate,
    )
    multi_track.to(device)
    
    return model, multi_track


def create_biometrics(args) -> BiometricData:
    """Create BiometricData from CLI arguments."""
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
        anxiety_score=args.anxiety,
        time_of_day=time_of_day,
    )


def parse_stems(stems_str: str) -> list:
    """Parse comma-separated stem names."""
    if not stems_str:
        return [StemType.BASS, StemType.DRUMS, StemType.MELODY, StemType.AMBIENT]
    
    stems = []
    for name in stems_str.split(","):
        name = name.strip().lower()
        try:
            stems.append(StemType(name))
        except ValueError:
            print(f"Warning: Unknown stem type '{name}', skipping")
    
    return stems


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-track therapeutic music"
    )
    
    # Model args
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
    
    # Biometrics
    parser.add_argument("--glucose", type=float, help="Glucose in mg/dL")
    parser.add_argument("--hrv", type=float, help="HRV (RMSSD) in ms")
    parser.add_argument("--anxiety", type=float, help="Anxiety score 0-10")
    
    # Generation
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration in seconds"
    )
    parser.add_argument(
        "--stems",
        type=str,
        default="",
        help="Comma-separated stems to generate (bass,drums,melody,ambient)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="therapeutic_multitrack.wav",
        help="Output mixed audio file"
    )
    parser.add_argument(
        "--export-stems",
        action="store_true",
        help="Export individual stems"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./stems",
        help="Directory for exported stems"
    )
    
    # Mix settings
    parser.add_argument(
        "--bass-boost",
        type=float,
        default=None,
        help="Bass boost in dB (auto if not set)"
    )
    parser.add_argument(
        "--binaural-freq",
        type=float,
        default=None,
        help="Binaural beat frequency in Hz"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    args = parser.parse_args()
    
    # Load config and model
    print("Loading configuration...")
    config = load_config(args.config)
    
    print("Loading model...")
    model, multi_track = load_model(args.checkpoint, config, args.device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create biometric data
    biometrics = create_biometrics(args)
    print(f"\nBiometric Data:")
    print(f"  Glucose: {biometrics.glucose_mg_dl or 'N/A'} mg/dL")
    print(f"  State: {biometrics.get_glucose_state().value}")
    print(f"  Stress: {biometrics.get_stress_level().value}")
    
    # Parse stems
    target_stems = parse_stems(args.stems)
    print(f"\nGenerating stems: {[s.value for s in target_stems]}")
    
    # Generate stems
    print(f"\nGenerating {args.duration}s of music...")
    stem_output = multi_track.generate_stems(
        backbone_model=model,
        tokenizer=model.tokenizer,
        duration_seconds=args.duration,
        temperature=args.temperature,
        target_stems=target_stems,
    )
    
    # Export individual stems if requested
    if args.export_stems:
        os.makedirs(args.output_dir, exist_ok=True)
        for stem_type, waveform in stem_output.stems.items():
            stem_path = os.path.join(args.output_dir, f"{stem_type.value}.wav")
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            torchaudio.save(stem_path, waveform.cpu(), stem_output.sample_rate)
            print(f"  Exported: {stem_path}")
    
    # Get mix settings from biometrics
    mix_settings = get_mix_settings_from_biometrics(
        glucose_mg_dl=biometrics.glucose_mg_dl,
        stress_level=biometrics.get_stress_level().value,
    )
    
    # Override with CLI args
    if args.bass_boost is not None:
        mix_settings.bass_boost_db = args.bass_boost
    if args.binaural_freq is not None:
        mix_settings.binaural_frequency = args.binaural_freq
        mix_settings.binaural_enabled = True
    
    print(f"\nMix Settings:")
    print(f"  Bass Boost: {mix_settings.bass_boost_db} dB")
    print(f"  Binaural: {mix_settings.binaural_frequency} Hz")
    
    # Mix stems
    print("\nMixing stems...")
    mixer = TherapeuticStemMixer(sample_rate=stem_output.sample_rate)
    mixed = mixer.mix(stem_output, mix_settings)
    
    # Save output
    print(f"\nSaving to {args.output}...")
    torchaudio.save(args.output, mixed.cpu(), stem_output.sample_rate)
    
    duration = mixed.shape[-1] / stem_output.sample_rate
    print(f"Generated {duration:.1f}s of therapeutic multi-track audio")
    print("Done!")


if __name__ == "__main__":
    main()
