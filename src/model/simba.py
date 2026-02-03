"""
SiMBA Music Generation Model

Main architecture combining Mamba blocks for efficient music generation.
Based on: "Exploring State-Space-Model based Language Model in Music Generation"

Key advantages:
- O(L) complexity vs O(L²) for Transformers
- 98% less training data required
- 12x cheaper to train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .mamba_block import ResidualMambaBlock, RMSNorm
from .audio_encoder import AudioEncoder, AudioDecoder


class SiMBAMusic(nn.Module):
    """
    SiMBA-based music generation model.
    
    Architecture:
    1. Audio Encoder (waveform -> tokens/embeddings)
    2. Token Embedding + Positional encoding
    3. Stack of Residual Mamba Blocks
    4. Output projection (-> vocabulary)
    5. Audio Decoder (embeddings -> mel-spectrogram)
    
    Args:
        d_model: Model hidden dimension
        n_layers: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Block expansion factor
        vocab_size: Audio token vocabulary size
        max_seq_len: Maximum sequence length
        sample_rate: Audio sample rate
        n_mels: Number of mel filterbanks
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        vocab_size: int = 4096,
        max_seq_len: int = 8192,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            d_model=d_model,
            vocab_size=vocab_size,
        )
        
        # Token embedding (for teacher forcing)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            ResidualMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # Output normalization
        self.norm = RMSNorm(d_model)
        
        # Language model head (for next token prediction)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights with token embedding
        self.lm_head.weight = self.token_embedding.weight
        
        # Audio decoder (for inference)
        self.audio_decoder = AudioDecoder(
            d_model=d_model,
            n_mels=n_mels,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with small std for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Token ids (batch, seq_len) - for teacher forcing
            waveform: Raw audio (batch, samples) - for encoding
            labels: Target token ids for loss computation
            return_dict: Return dictionary of outputs
            
        Returns:
            Dictionary with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if labels provided
                - hidden_states: (batch, seq_len, d_model)
        """
        # Get embeddings from tokens or waveform
        if input_ids is not None:
            embeddings = self.token_embedding(input_ids)
        elif waveform is not None:
            embeddings, _ = self.audio_encoder(waveform)
        else:
            raise ValueError("Must provide either input_ids or waveform")
            
        batch_size, seq_len, _ = embeddings.shape
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=embeddings.device)
        embeddings = embeddings + self.pos_embedding(positions)
        
        # Pass through Mamba layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": hidden_states,
            }
        else:
            return logits, loss, hidden_states
            
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_waveform: Optional[torch.Tensor] = None,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate audio tokens autoregressively.
        
        Args:
            prompt_ids: Starting token ids (batch, prompt_len)
            prompt_waveform: Starting audio waveform
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            generated_ids: (batch, generated_length)
        """
        self.eval()
        
        # Get initial tokens
        if prompt_ids is not None:
            current_ids = prompt_ids
        elif prompt_waveform is not None:
            _, current_ids = self.audio_encoder(prompt_waveform, return_tokens=True)
        else:
            # Start with random token
            current_ids = torch.zeros(1, 1, dtype=torch.long, device=next(self.parameters()).device)
            
        # Generate tokens
        for _ in range(max_length - current_ids.shape[1]):
            # Get logits
            outputs = self(input_ids=current_ids)
            next_logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
                
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
                
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
        return current_ids
    
    def tokens_to_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert generated tokens to mel-spectrogram.
        
        Args:
            tokens: (batch, seq_len)
            
        Returns:
            mel: (batch, n_mels, time)
        """
        embeddings = self.audio_encoder.decode_tokens(tokens)
        mel = self.audio_decoder(embeddings)
        return mel
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SiMBAMusic":
        """Create model from configuration dictionary."""
        model_config = config.get("model", {})
        audio_config = config.get("audio", {})
        
        return cls(
            d_model=model_config.get("d_model", 512),
            n_layers=model_config.get("n_layers", 8),
            d_state=model_config.get("d_state", 16),
            d_conv=model_config.get("d_conv", 4),
            expand=model_config.get("expand", 2),
            vocab_size=model_config.get("vocab_size", 4096),
            max_seq_len=model_config.get("max_seq_len", 8192),
            sample_rate=audio_config.get("sample_rate", 44100),
            n_mels=audio_config.get("n_mels", 128),
            n_fft=audio_config.get("n_fft", 2048),
            hop_length=audio_config.get("hop_length", 512),
        )
        
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
