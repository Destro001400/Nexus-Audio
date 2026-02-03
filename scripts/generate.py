#!/usr/bin/env python3
"""
Generate Therapeutic Music with SiMBA

Generate music optimized for therapeutic effects:
- Insulin release (bass 50-60 Hz)
- Glucose tolerance (HFC)
- Brain entrainment (binaural beats)

Usage:
    python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --output generated.wav
    python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --phase activation --duration 300
"""

import argparse
from pathlib import Path
import torch
import torchaudio

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SiMBAMusic
from src.audio.therapeutic import TherapeuticOptimizer, TherapeuticProtocol
from src.audio.processor import AudioProcessor


def load_model(checkpoint_path: str, device: str = "cuda") -> SiMBAMusic:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", {})
    
    model = SiMBAMusic.from_config({"model": config, "audio": config})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def generate_therapeutic_session(
    model: SiMBAMusic,
    output_path: str,
    duration_sec: float = 1800,  # 30 minutes
    phase: str = "full",
    sample_rate: int = 44100,
    device: str = "cuda",
):
    """
    Generate a full therapeutic music session.
    
    Phases:
    - activation: 5 min, rock/bass heavy
    - modulation: 15 min, instrumental with HFC
    - consolidation: 10 min, ambient with theta binaural
    - full: All three phases
    
    Args:
        model: Trained SiMBA model
        output_path: Path to save generated audio
        duration_sec: Duration in seconds
        phase: Which phase to generate
        sample_rate: Audio sample rate
        device: Device (cuda/cpu)
    """
    print(f"\nGenerating therapeutic music session...")
    print(f"Duration: {duration_sec / 60:.1f} minutes")
    print(f"Phase: {phase}")
    
    # Initialize therapeutic components
    optimizer = TherapeuticOptimizer(sample_rate=sample_rate)
    protocol = TherapeuticProtocol(sample_rate=sample_rate, optimizer=optimizer)
    processor = AudioProcessor(sample_rate=sample_rate)
    
    # Calculate tokens needed
    tokens_per_second = sample_rate / 512  # Based on hop_length
    n_tokens = int(duration_sec * tokens_per_second)
    
    print(f"Generating {n_tokens:,} tokens...")
    
    # Generate tokens
    with torch.no_grad():
        generated_ids = model.generate(
            max_length=min(n_tokens, model.max_seq_len),
            temperature=0.95,
            top_k=50,
            top_p=0.95,
        )
        
    print(f"Generated {generated_ids.shape[1]} tokens")
    
    # Convert to mel-spectrogram
    mel = model.tokens_to_audio(generated_ids)
    
    # Convert mel to waveform (simple inverse using Griffin-Lim)
    # For production, use a proper vocoder like HiFi-GAN
    mel = mel.squeeze(0).cpu()
    
    # Inverse mel-spectrogram (simplified)
    n_fft = 2048
    hop_length = 512
    n_mels = mel.shape[0]
    
    # Use torchaudio's inverse mel scale
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    # Convert log mel to linear
    mel_linear = torch.exp(mel)
    
    # Inverse mel scale
    spec = inverse_mel(mel_linear)
    
    # Griffin-Lim to get waveform
    waveform = griffin_lim(spec)
    waveform = waveform.unsqueeze(0)  # Add channel dim
    
    print(f"Generated {waveform.shape[-1] / sample_rate:.1f} seconds of audio")
    
    # Apply therapeutic optimization
    print("Applying therapeutic optimization...")
    
    if phase == "full":
        # Apply full protocol with phase transitions
        # For now, apply average settings
        waveform = optimizer(waveform, enhance_bass=True, preserve_hfc=True)
        # Add binaural beats (theta for relaxation)
        waveform = optimizer.mix_binaural_with_music(
            waveform.squeeze(),
            beat_frequency=6.0,  # Theta
            binaural_volume=0.1,
        )
    elif phase == "activation":
        optimizer.bass_boost_db = 6.0
        waveform = optimizer(waveform, enhance_bass=True)
    elif phase == "modulation":
        optimizer.bass_boost_db = 3.0
        waveform = optimizer(waveform, enhance_bass=True, preserve_hfc=True)
    elif phase == "consolidation":
        waveform = optimizer.mix_binaural_with_music(
            waveform.squeeze(),
            beat_frequency=6.0,
            binaural_volume=0.15,
        )
        
    # Normalize
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform / waveform.abs().max() * 0.95
    
    # Save
    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, waveform, sample_rate)
    
    print(f"\n✓ Generated therapeutic music saved to {output_path}")
    print(f"  Duration: {waveform.shape[-1] / sample_rate:.1f} seconds")
    print(f"  Channels: {waveform.shape[0]}")
    print(f"  Sample rate: {sample_rate} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Generate therapeutic music with SiMBA"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="generated_therapy.wav",
        help="Output audio path"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="Duration in seconds"
    )
    parser.add_argument(
        "--phase", "-p",
        type=str,
        choices=["full", "activation", "modulation", "consolidation"],
        default="full",
        help="Therapeutic phase to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SiMBA Therapeutic Music Generator")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {args.device}")
    
    # Generate
    generate_therapeutic_session(
        model=model,
        output_path=args.output,
        duration_sec=args.duration,
        phase=args.phase,
        device=args.device,
    )


if __name__ == "__main__":
    main()
