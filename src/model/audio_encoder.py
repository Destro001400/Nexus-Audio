"""
Audio Encoder for SiMBA

Converts raw audio waveforms into discrete tokens using:
1. Mel-spectrogram extraction
2. Convolutional encoder
3. Vector quantization (simplified RVQ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Tuple, Optional


class AudioEncoder(nn.Module):
    """
    Encodes audio waveforms into discrete tokens for language modeling.
    
    Pipeline:
    1. Waveform -> Mel-spectrogram
    2. Mel-spectrogram -> Convolutional features
    3. Features -> Discrete tokens (via codebook lookup)
    
    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel filterbanks
        n_fft: FFT size
        hop_length: Hop length for STFT
        d_model: Output embedding dimension
        vocab_size: Number of discrete tokens
        n_codebooks: Number of RVQ levels (simplified to 1 for now)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        d_model: int = 512,
        vocab_size: int = 4096,
        n_codebooks: int = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Mel-spectrogram extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Codebook for vector quantization
        self.codebook = nn.Embedding(vocab_size, d_model)
        nn.init.uniform_(self.codebook.weight, -1/vocab_size, 1/vocab_size)
        
        # Projection for continuous output
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        waveform: torch.Tensor,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode waveform to embeddings or tokens.
        
        Args:
            waveform: Input audio (batch, samples) or (batch, 1, samples)
            return_tokens: If True, also return discrete tokens
            
        Returns:
            embeddings: Continuous embeddings (batch, seq_len, d_model)
            tokens: Discrete tokens if return_tokens=True (batch, seq_len)
        """
        # Handle mono/stereo
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)  # Mix to mono
            
        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)  # (B, n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))  # Log compression
        
        # Encode
        features = self.encoder(mel)  # (B, d_model, T//2)
        features = features.transpose(1, 2)  # (B, T//2, d_model)
        
        if return_tokens:
            # Vector quantization
            tokens = self.quantize(features)
            embeddings = self.codebook(tokens)
            return embeddings, tokens
        else:
            embeddings = self.proj(features)
            return embeddings, None
    
    def quantize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous features to discrete tokens.
        
        Uses nearest neighbor lookup in codebook.
        
        Args:
            features: (batch, seq_len, d_model)
            
        Returns:
            tokens: (batch, seq_len)
        """
        # Flatten for distance computation
        B, L, D = features.shape
        flat_features = features.reshape(-1, D)  # (B*L, D)
        
        # Compute distances to codebook
        codebook = self.codebook.weight  # (vocab_size, D)
        
        # Efficient distance computation: ||x - c||^2 = ||x||^2 - 2*x*c + ||c||^2
        distances = (
            flat_features.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat_features @ codebook.T
            + codebook.pow(2).sum(dim=-1)
        )
        
        # Get nearest tokens
        tokens = distances.argmin(dim=-1)
        tokens = tokens.reshape(B, L)
        
        return tokens
    
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens back to embeddings.
        
        Args:
            tokens: (batch, seq_len)
            
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        return self.codebook(tokens)


class AudioDecoder(nn.Module):
    """
    Decodes embeddings back to mel-spectrogram (for vocoder input).
    
    Args:
        d_model: Input embedding dimension
        n_mels: Number of mel filterbanks
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, n_mels),
        )
        
        # Upsample to match encoder downsampling
        self.upsample = nn.ConvTranspose1d(
            n_mels, n_mels, kernel_size=4, stride=2, padding=1
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to mel-spectrogram.
        
        Args:
            embeddings: (batch, seq_len, d_model)
            
        Returns:
            mel: (batch, n_mels, seq_len * 2)
        """
        mel = self.decoder(embeddings)  # (B, L, n_mels)
        mel = mel.transpose(1, 2)  # (B, n_mels, L)
        mel = self.upsample(mel)  # (B, n_mels, L*2)
        return mel
