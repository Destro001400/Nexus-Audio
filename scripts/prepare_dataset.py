#!/usr/bin/env python3
"""
Prepare Dataset for SiMBA Training

This script processes raw audio files and prepares them for training:
1. Finds all audio files in input directory
2. Resamples to target sample rate
3. Chunks into training segments
4. Splits into train/val sets
5. Saves to output directory

Usage:
    python scripts/prepare_dataset.py --input ./raw_audio --output ./data
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.processor import AudioProcessor, find_audio_files, get_audio_duration


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 44100,
    chunk_length_sec: float = 10.0,
    val_split: float = 0.1,
    min_duration_sec: float = 30.0,
    seed: int = 42,
):
    """
    Prepare audio dataset for training.
    
    Args:
        input_dir: Directory with raw audio files
        output_dir: Output directory for processed data
        sample_rate: Target sample rate
        chunk_length_sec: Length of audio chunks
        val_split: Fraction for validation set
        min_duration_sec: Minimum file duration to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = AudioProcessor(
        sample_rate=sample_rate,
        chunk_length_sec=chunk_length_sec,
    )
    
    # Find all audio files
    print(f"Scanning {input_dir} for audio files...")
    audio_files = find_audio_files(input_dir)
    print(f"Found {len(audio_files)} audio files")
    
    # Filter by duration
    valid_files = []
    total_duration = 0.0
    
    print("Checking file durations...")
    for path in tqdm(audio_files, desc="Scanning"):
        try:
            duration = get_audio_duration(str(path))
            if duration >= min_duration_sec:
                valid_files.append((path, duration))
                total_duration += duration
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            
    print(f"\nValid files: {len(valid_files)}")
    print(f"Total duration: {total_duration / 3600:.2f} hours")
    
    if len(valid_files) == 0:
        print("No valid audio files found!")
        return
        
    # Shuffle and split
    random.shuffle(valid_files)
    n_val = max(1, int(len(valid_files) * val_split))
    n_train = len(valid_files) - n_val
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:]
    
    print(f"\nTrain files: {n_train}")
    print(f"Val files: {n_val}")
    
    # Process files
    def process_split(files, output_dir, split_name):
        chunk_count = 0
        
        for path, duration in tqdm(files, desc=f"Processing {split_name}"):
            try:
                # Load full audio
                waveform, _ = processor.load_audio(str(path))
                
                # Chunk audio
                chunks = processor.chunk_audio(waveform, overlap_ratio=0.0)
                
                # Save chunks
                for i, chunk in enumerate(chunks):
                    chunk_path = output_dir / f"{path.stem}_{i:04d}.wav"
                    processor.save_audio(chunk, str(chunk_path))
                    chunk_count += 1
                    
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                continue
                
        return chunk_count
    
    # Process train and val
    train_chunks = process_split(train_files, train_dir, "train")
    val_chunks = process_split(val_files, val_dir, "val")
    
    # Summary
    print("\n" + "=" * 50)
    print("Dataset Preparation Complete!")
    print("=" * 50)
    print(f"Train chunks: {train_chunks}")
    print(f"Val chunks: {val_chunks}")
    print(f"Output directory: {output_path}")
    print(f"\nEstimated training duration: ~{total_duration * 0.9 / chunk_length_sec * 50 / 3600:.1f} GPU hours")
    print("(Assuming 50ms per step with batch size 8)")
    

def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset for SiMBA training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with raw audio files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=44100,
        help="Target sample rate (default: 44100)"
    )
    parser.add_argument(
        "--chunk-length", "-c",
        type=float,
        default=10.0,
        help="Chunk length in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--val-split", "-v",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--min-duration", "-m",
        type=float,
        default=30.0,
        help="Minimum file duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        chunk_length_sec=args.chunk_length,
        val_split=args.val_split,
        min_duration_sec=args.min_duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
