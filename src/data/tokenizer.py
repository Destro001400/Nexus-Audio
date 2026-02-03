"""
Audio Tokenizer for SiMBA

Converts audio to discrete tokens using Residual Vector Quantization (RVQ).
This is a simplified implementation - for production, consider using EnCodec or DAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchaudio.transforms as T


class AudioTokenizer(nn.Module):
    """
    Tokenizes audio waveforms into discrete indices.
    
    Uses a simplified RVQ approach:
    1. Extract mel-spectrogram
    2. Encode with CNN
    3. Quantize with learned codebooks
    
    For production, consider using:
    - EnCodec (Meta)
    - DAC (Descript Audio Codec)
    - SoundStream (Google)
    
    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel filterbanks
        vocab_size: Size of each codebook
        n_codebooks: Number of RVQ levels
        d_model: Encoder hidden dimension
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        vocab_size: int = 4096,
        n_codebooks: int = 4,
        d_model: int = 512,
        hop_length: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.d_model = d_model
        self.hop_length = hop_length
        
        # Mel-spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # RVQ codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])
        
        # Initialize codebooks
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1/vocab_size, 1/vocab_size)
            
        # Decoder (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, n_mels, kernel_size=3, padding=1),
        )
        
    def encode(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode waveform to tokens.
        
        Args:
            waveform: (batch, samples) or (batch, 1, samples)
            
        Returns:
            tokens: (batch, n_codebooks, seq_len)
            embeddings: (batch, seq_len, d_model)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)  # (B, n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))
        
        # Encode
        features = self.encoder(mel)  # (B, d_model, T)
        features = features.transpose(1, 2)  # (B, T, d_model)
        
        # RVQ encoding
        tokens, embeddings = self._rvq_encode(features)
        
        return tokens, embeddings
    
    def _rvq_encode(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Residual Vector Quantization.
        
        Progressively quantizes the residual with multiple codebooks.
        
        Args:
            features: (batch, seq_len, d_model)
            
        Returns:
            tokens: (batch, n_codebooks, seq_len)
            quantized: (batch, seq_len, d_model)
        """
        B, L, D = features.shape
        
        residual = features.clone()
        all_tokens = []
        quantized_sum = torch.zeros_like(features)
        
        for codebook in self.codebooks:
            # Find nearest codebook entry
            tokens = self._quantize_to_codebook(residual, codebook)
            all_tokens.append(tokens)
            
            # Get quantized values
            quantized = codebook(tokens)  # (B, L, D)
            
            # Update residual
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            
        # Stack tokens: (B, n_codebooks, L)
        tokens = torch.stack(all_tokens, dim=1)
        
        return tokens, quantized_sum
    
    def _quantize_to_codebook(
        self,
        features: torch.Tensor,
        codebook: nn.Embedding,
    ) -> torch.Tensor:
        """
        Find nearest codebook entry for each position.
        
        Args:
            features: (batch, seq_len, d_model)
            codebook: Embedding layer
            
        Returns:
            tokens: (batch, seq_len)
        """
        B, L, D = features.shape
        
        # Flatten
        flat = features.reshape(-1, D)  # (B*L, D)
        weights = codebook.weight  # (vocab_size, D)
        
        # Compute distances
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ weights.T
            + weights.pow(2).sum(dim=-1)
        )
        
        # Get nearest
        tokens = distances.argmin(dim=-1)
        return tokens.reshape(B, L)
    
    def decode(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode tokens to mel-spectrogram.
        
        Args:
            tokens: (batch, n_codebooks, seq_len)
            
        Returns:
            mel: (batch, n_mels, seq_len)
        """
        B, N, L = tokens.shape
        
        # Sum embeddings from all codebooks
        embeddings = torch.zeros(B, L, self.d_model, device=tokens.device)
        
        for i, codebook in enumerate(self.codebooks):
            embeddings = embeddings + codebook(tokens[:, i])
            
        # Decode to mel
        embeddings = embeddings.transpose(1, 2)  # (B, d_model, L)
        mel = self.decoder(embeddings)  # (B, n_mels, L)
        
        return mel
    
    def flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Flatten multi-codebook tokens for language model input.
        
        Interleaves tokens from different codebooks.
        
        Args:
            tokens: (batch, n_codebooks, seq_len)
            
        Returns:
            flat_tokens: (batch, n_codebooks * seq_len)
        """
        B, N, L = tokens.shape
        
        # Interleave: [c0_t0, c1_t0, ..., c0_t1, c1_t1, ...]
        tokens = tokens.transpose(1, 2)  # (B, L, N)
        return tokens.reshape(B, -1)
    
    def unflatten_tokens(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        """
        Unflatten language model output to multi-codebook tokens.
        
        Args:
            flat_tokens: (batch, n_codebooks * seq_len)
            
        Returns:
            tokens: (batch, n_codebooks, seq_len)
        """
        B, S = flat_tokens.shape
        L = S // self.n_codebooks
        
        tokens = flat_tokens.reshape(B, L, self.n_codebooks)
        return tokens.transpose(1, 2)
    
    @property
    def tokens_per_second(self) -> float:
        """Number of token frames per second of audio."""
        return self.sample_rate / self.hop_length
