#!/usr/bin/env python3
"""
Train SiMBA Music Generation Model

Training script with:
- Configuration loading from YAML
- Mixed precision training
- Checkpointing and resuming
- Logging to TensorBoard/W&B

Usage:
    python scripts/train.py --config configs/simba_therapy.yaml
    python scripts/train.py --config configs/simba_therapy.yaml --resume outputs/checkpoints/step_10000.pt
    python scripts/train.py --debug  # Quick test with small model
"""

import argparse
from pathlib import Path
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import SiMBAMusic
from src.data.dataset import AudioDataModule
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_debug_config() -> dict:
    """Create minimal config for debugging."""
    return {
        "model": {
            "d_model": 128,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "vocab_size": 1024,
            "max_seq_len": 1024,
        },
        "audio": {
            "sample_rate": 22050,
            "n_mels": 64,
            "n_fft": 1024,
            "hop_length": 256,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "warmup_steps": 10,
            "max_steps": 100,
            "fp16": False,
            "log_every_n_steps": 10,
            "save_every_n_steps": 50,
            "eval_every_n_steps": 50,
        },
        "data": {
            "train_path": "./data/train",
            "val_path": "./data/val",
            "chunk_length_sec": 5.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SiMBA music generation model"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/simba_therapy.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with small model"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.debug:
        print("Running in DEBUG mode with minimal configuration")
        config = create_debug_config()
    else:
        config = load_config(args.config)
        
    print("\n" + "=" * 50)
    print("SiMBA Music Generation Training")
    print("=" * 50)
    
    # Print config summary
    model_config = config.get("model", {})
    print(f"\nModel Config:")
    print(f"  d_model: {model_config.get('d_model', 512)}")
    print(f"  n_layers: {model_config.get('n_layers', 8)}")
    print(f"  vocab_size: {model_config.get('vocab_size', 4096)}")
    
    training_config = config.get("training", {})
    print(f"\nTraining Config:")
    print(f"  batch_size: {training_config.get('batch_size', 8)}")
    print(f"  learning_rate: {training_config.get('learning_rate', 3e-4)}")
    print(f"  max_steps: {training_config.get('max_steps', 100000)}")
    print(f"  fp16: {training_config.get('fp16', True)}")
    
    # Create model
    print("\nCreating model...")
    model = SiMBAMusic.from_config(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create data module
    print("\nLoading data...")
    data_config = config.get("data", {})
    audio_config = config.get("audio", {})
    
    data_module = AudioDataModule(
        train_dir=data_config.get("train_path", "./data/train"),
        val_dir=data_config.get("val_path", "./data/val"),
        batch_size=training_config.get("batch_size", 8),
        sample_rate=audio_config.get("sample_rate", 44100),
        chunk_length_sec=data_config.get("chunk_length_sec", 10.0),
    )
    
    try:
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        print(f"Train batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"Val batches: {len(val_dataloader)}")
    except Exception as e:
        print(f"\nWarning: Could not load data: {e}")
        print("Creating dummy dataloader for testing...")
        
        # Create dummy dataloader for testing
        from torch.utils.data import DataLoader, TensorDataset
        
        dummy_waveforms = torch.randn(16, 1, 220500)  # 5 seconds at 44.1kHz
        dummy_dataset = TensorDataset(dummy_waveforms)
        
        class DummyDataLoader:
            def __init__(self, data):
                self.data = data
                
            def __iter__(self):
                for (waveform,) in self.data:
                    yield {"waveform": waveform}
                    
            def __len__(self):
                return len(self.data)
                
        train_dataloader = DummyDataLoader(
            DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        )
        val_dataloader = None
        
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Train!
    print("\nStarting training...")
    print("=" * 50 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted")
        print("Checkpoint saved as 'interrupted.pt'")
        
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
