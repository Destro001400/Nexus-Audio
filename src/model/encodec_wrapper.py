"""
EnCodec Wrapper for SiMBA

Uses Meta's EnCodec for proper neural audio tokenization.
EnCodec compresses audio ~40x while maintaining quality.

Paper: "High Fidelity Neural Audio Compression" (Meta, 2022)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import warnings

# Try to import encodec, provide fallback
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    warnings.warn(
        "EnCodec not installed. Install with: pip install encodec\n"
        "Falling back to simplified tokenizer."
    )


class EnCodecWrapper(nn.Module):
    """
    Wrapper around Meta's EnCodec for audio tokenization.
    
    EnCodec uses Residual Vector Quantization (RVQ) to create
    discrete tokens from audio. Each codebook adds refinement.
    
    Args:
        model_type: "24k" (24kHz) or "48k" (48kHz)
        bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
        device: "cuda" or "cpu"
        
    Output Characteristics (24kHz, 6.0 kbps):
        - ~75 tokens per second of audio
        - 4 codebooks (can be configured)
        - Vocab size: 1024 per codebook
    """
    
    def __init__(
        self,
        model_type: str = "24k",
        bandwidth: float = 6.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        self.device = device
        self.bandwidth = bandwidth
        
        if not ENCODEC_AVAILABLE:
            raise ImportError(
                "EnCodec is required. Install with: pip install encodec"
            )
        
        # Load pre-trained EnCodec model
        if model_type == "24k":
            self.model = EncodecModel.encodec_model_24khz()
            self.sample_rate = 24000
        elif model_type == "48k":
            self.model = EncodecModel.encodec_model_48khz()
            self.sample_rate = 48000
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self.model.to(device)
        
        # Freeze EnCodec parameters (we don't train it)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Token characteristics
        self.vocab_size = 1024  # EnCodec uses 1024 tokens per codebook
        self.n_codebooks = self._get_n_codebooks()
        self.tokens_per_second = int(self.sample_rate / self.model.encoder.hop_length)
        
    def _get_n_codebooks(self) -> int:
        """Get number of codebooks based on bandwidth."""
        # EnCodec bandwidth -> codebook mapping
        bandwidth_to_codebooks = {
            1.5: 2,
            3.0: 4,
            6.0: 8,
            12.0: 16,
            24.0: 32,
        }
        return bandwidth_to_codebooks.get(self.bandwidth, 8)
    
    @torch.no_grad()
    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode audio waveform to discrete tokens.
        
        Args:
            waveform: Audio tensor (batch, channels, samples) or (batch, samples)
            sample_rate: Input sample rate (resampled if different from model)
            
        Returns:
            tokens: Discrete tokens (batch, n_codebooks, n_frames)
        """
        # Ensure correct shape
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # Add channel dim
        
        # Convert sample rate if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = convert_audio(
                waveform, sample_rate, self.sample_rate, self.model.channels
            )
        
        waveform = waveform.to(self.device)
        
        # Encode to tokens
        encoded_frames = self.model.encode(waveform)
        
        # Extract codes (list of (codes, scale) tuples)
        codes = encoded_frames[0][0]  # (batch, n_codebooks, n_frames)
        
        return codes
    
    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens back to audio waveform.
        
        Args:
            tokens: Discrete tokens (batch, n_codebooks, n_frames)
            
        Returns:
            waveform: Audio tensor (batch, channels, samples)
        """
        tokens = tokens.to(self.device)
        
        # Create encoded frames format expected by decoder
        encoded_frames = [(tokens, None)]
        
        # Decode
        waveform = self.model.decode(encoded_frames)
        
        return waveform
    
    def tokens_to_embeddings(self, tokens: torch.Tensor, d_model: int = 512) -> torch.Tensor:
        """
        Convert tokens to continuous embeddings for the model.
        
        Flattens multi-codebook tokens and projects to d_model.
        
        Args:
            tokens: (batch, n_codebooks, n_frames)
            d_model: Target embedding dimension
            
        Returns:
            embeddings: (batch, n_frames, d_model)
        """
        batch, n_codebooks, n_frames = tokens.shape
        
        # Create embedding layers if not exists
        if not hasattr(self, 'token_embeddings'):
            self.token_embeddings = nn.ModuleList([
                nn.Embedding(self.vocab_size, d_model // n_codebooks)
                for _ in range(n_codebooks)
            ]).to(self.device)
            
            self.embedding_proj = nn.Linear(
                d_model // n_codebooks * n_codebooks, d_model
            ).to(self.device)
        
        # Embed each codebook
        embeddings = []
        for i in range(n_codebooks):
            emb = self.token_embeddings[i](tokens[:, i, :])  # (B, T, D/n_cb)
            embeddings.append(emb)
        
        # Concatenate and project
        combined = torch.cat(embeddings, dim=-1)  # (B, T, D)
        projected = self.embedding_proj(combined)  # (B, T, d_model)
        
        return projected
    
    def get_audio_duration(self, n_tokens: int) -> float:
        """Calculate audio duration from number of tokens."""
        return n_tokens / self.tokens_per_second
    
    def get_n_tokens(self, duration_seconds: float) -> int:
        """Calculate number of tokens for a given duration."""
        return int(duration_seconds * self.tokens_per_second)


class SimplifiedTokenizer(nn.Module):
    """
    Fallback tokenizer when EnCodec is not available.
    Uses mel-spectrogram + learned codebook (less quality than EnCodec).
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: int = 128,
        vocab_size: int = 1024,
        n_codebooks: int = 4,
        d_model: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        
        # Mel-spectrogram
        import torchaudio.transforms as T
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,  # ~75 tokens/sec at 24kHz
            n_mels=n_mels,
        )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, d_model, 3, padding=1),
        )
        
        # Codebooks (simplified RVQ)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(vocab_size, d_model // n_codebooks))
            for _ in range(n_codebooks)
        ])
        
        self.tokens_per_second = sample_rate // 320
    
    def encode(self, waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode waveform to tokens."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        mel = self.mel(waveform)
        mel = torch.log(mel.clamp(min=1e-5))
        
        features = self.encoder(mel)  # (B, D, T)
        features = features.transpose(1, 2)  # (B, T, D)
        
        # Quantize to each codebook
        B, T, D = features.shape
        tokens = []
        
        for i, codebook in enumerate(self.codebooks):
            # Select feature slice for this codebook
            start = i * (D // self.n_codebooks)
            end = (i + 1) * (D // self.n_codebooks)
            feat_slice = features[:, :, start:end]
            
            # Find nearest codebook entry
            distances = torch.cdist(
                feat_slice.reshape(-1, feat_slice.shape[-1]),
                codebook
            )
            token_ids = distances.argmin(dim=-1).reshape(B, T)
            tokens.append(token_ids)
        
        return torch.stack(tokens, dim=1)  # (B, n_codebooks, T)
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Simplified decode (returns zeros as placeholder)."""
        warnings.warn(
            "SimplifiedTokenizer.decode() is not fully implemented. "
            "Use EnCodec for proper audio reconstruction."
        )
        B, n_cb, T = tokens.shape
        # Return placeholder audio
        return torch.zeros(B, 1, T * 320)


def get_tokenizer(use_encodec: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to get the appropriate tokenizer.
    
    Args:
        use_encodec: Whether to use EnCodec (recommended)
        **kwargs: Arguments passed to tokenizer
        
    Returns:
        Tokenizer module
    """
    if use_encodec and ENCODEC_AVAILABLE:
        return EnCodecWrapper(**kwargs)
    else:
        if use_encodec:
            warnings.warn("EnCodec not available, using simplified tokenizer.")
        return SimplifiedTokenizer(**kwargs)
