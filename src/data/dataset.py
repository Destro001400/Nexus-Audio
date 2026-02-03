"""
Audio Dataset for SiMBA Training

Handles audio loading, preprocessing, and batching for efficient training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import random

from ..audio.processor import AudioProcessor, find_audio_files


class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio files.
    
    Loads and preprocesses audio chunks for training.
    
    Args:
        data_dir: Directory containing audio files
        sample_rate: Target sample rate
        chunk_length_sec: Length of audio chunks
        augment: Apply data augmentation
        cache_metadata: Cache file metadata
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        chunk_length_sec: float = 10.0,
        augment: bool = True,
        cache_metadata: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.augment = augment
        
        # Initialize processor
        self.processor = AudioProcessor(
            sample_rate=sample_rate,
            chunk_length_sec=chunk_length_sec,
        )
        
        # Find all audio files
        self.audio_files = find_audio_files(data_dir)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
            
        # Build index of chunks
        self.chunk_index = self._build_chunk_index(cache_metadata)
        
    def _build_chunk_index(self, cache: bool = True) -> List[Dict[str, Any]]:
        """
        Build index of all available audio chunks.
        
        Returns list of dicts with:
            - file_path: Path to audio file
            - start_sec: Start time of chunk
            - duration_sec: Duration of chunk
        """
        cache_path = self.data_dir / ".chunk_index.json"
        
        # Try to load from cache
        if cache and cache_path.exists():
            with open(cache_path, 'r') as f:
                index = json.load(f)
                # Convert paths back to Path objects
                for item in index:
                    item['file_path'] = Path(item['file_path'])
                return index
        
        index = []
        
        for audio_path in self.audio_files:
            try:
                from ..audio.processor import get_audio_duration
                duration = get_audio_duration(str(audio_path))
                
                # Calculate number of chunks
                n_chunks = int(duration / self.chunk_length_sec)
                
                for i in range(n_chunks):
                    index.append({
                        'file_path': audio_path,
                        'start_sec': i * self.chunk_length_sec,
                        'duration_sec': self.chunk_length_sec,
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {audio_path}: {e}")
                continue
                
        # Cache index
        if cache:
            cache_data = [
                {**item, 'file_path': str(item['file_path'])}
                for item in index
            ]
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
        return index
    
    def __len__(self) -> int:
        return len(self.chunk_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio chunk.
        
        Returns:
            Dict with:
                - waveform: (1, samples)
                - input_ids: Token IDs (if tokenizer available)
        """
        chunk_info = self.chunk_index[idx]
        
        # Load audio chunk
        waveform, _ = self.processor.load_audio(
            str(chunk_info['file_path']),
            start_sec=chunk_info['start_sec'],
            duration_sec=chunk_info['duration_sec'],
        )
        
        # Apply augmentation
        if self.augment:
            waveform = self.processor.augment(
                waveform,
                pitch_shift=True,
                time_stretch=False,  # Disabled to maintain chunk length
            )
            
        return {
            'waveform': waveform,
        }


class AudioDataModule:
    """
    Data module for managing train/val datasets and dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: DataLoader workers
        **kwargs: Additional dataset arguments
    """
    
    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        sample_rate: int = 44100,
        chunk_length_sec: float = 10.0,
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self):
        """Initialize datasets."""
        self.train_dataset = AudioDataset(
            self.train_dir,
            sample_rate=self.sample_rate,
            chunk_length_sec=self.chunk_length_sec,
            augment=True,
        )
        
        if self.val_dir:
            self.val_dataset = AudioDataset(
                self.val_dir,
                sample_rate=self.sample_rate,
                chunk_length_sec=self.chunk_length_sec,
                augment=False,
            )
            
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
