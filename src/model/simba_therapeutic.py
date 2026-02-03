"""
SiMBA Therapeutic Music Generation Model

Enhanced version of SiMBA with:
- EnCodec integration for high-quality tokenization
- Biofeedback conditioning (glucose, HRV, stress)
- Therapeutic protocol support

This model generates music that adapts to the user's physiological state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import warnings

from .mamba_block import ResidualMambaBlock, RMSNorm
from .encodec_wrapper import EnCodecWrapper, get_tokenizer, ENCODEC_AVAILABLE

# Try to import biofeedback (might not exist yet)
try:
    from src.therapeutic.biofeedback import (
        BiofeedbackEmbedding,
        BiofeedbackController,
        BiometricData,
        MusicParameters,
    )
    BIOFEEDBACK_AVAILABLE = True
except ImportError:
    BIOFEEDBACK_AVAILABLE = False
    warnings.warn("Biofeedback module not available.")


class SiMBATherapeutic(nn.Module):
    """
    SiMBA model optimized for therapeutic music generation.
    
    Key differences from base SiMBA:
    1. Uses EnCodec for proper RVQ tokenization
    2. Accepts biofeedback conditioning (glucose, HRV, etc.)
    3. Generates music adapted to user's physiological state
    
    Architecture:
        EnCodec Tokens → Token Embedding + Biofeedback Embedding
        → Mamba Blocks × N → Output Projection → Token Prediction
        → EnCodec Decoder → Audio Waveform
    
    Args:
        d_model: Hidden dimension
        n_layers: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Conv kernel size in Mamba
        expand: Block expansion factor
        encodec_bandwidth: EnCodec bandwidth (6.0 kbps recommended)
        sample_rate: Target sample rate (24000 for EnCodec)
        max_seq_len: Maximum sequence length
        use_biofeedback: Enable biofeedback conditioning
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        encodec_bandwidth: float = 6.0,
        sample_rate: int = 24000,
        max_seq_len: int = 8192,
        use_biofeedback: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.max_seq_len = max_seq_len
        self.use_biofeedback = use_biofeedback and BIOFEEDBACK_AVAILABLE
        self.device = device
        
        # Initialize EnCodec tokenizer
        self.tokenizer = get_tokenizer(
            use_encodec=ENCODEC_AVAILABLE,
            bandwidth=encodec_bandwidth,
            device=device,
        )
        
        self.vocab_size = self.tokenizer.vocab_size
        self.n_codebooks = self.tokenizer.n_codebooks
        
        # Token embeddings (one per codebook)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model // self.n_codebooks)
            for _ in range(self.n_codebooks)
        ])
        
        # Combine codebook embeddings
        self.embedding_combine = nn.Linear(
            d_model // self.n_codebooks * self.n_codebooks,
            d_model
        )
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Biofeedback conditioning
        if self.use_biofeedback:
            self.biofeedback_embedding = BiofeedbackEmbedding(d_model)
            self.biofeedback_proj = nn.Linear(d_model * 2, d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            ResidualMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(d_model)
        
        # Prediction heads (one per codebook)
        self.lm_heads = nn.ModuleList([
            nn.Linear(d_model, self.vocab_size)
            for _ in range(self.n_codebooks)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed multi-codebook tokens.
        
        Args:
            tokens: (batch, n_codebooks, seq_len)
            
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch, n_cb, seq_len = tokens.shape
        
        # Embed each codebook
        embeddings = []
        for i in range(n_cb):
            emb = self.token_embeddings[i](tokens[:, i, :])
            embeddings.append(emb)
        
        # Concatenate and project
        combined = torch.cat(embeddings, dim=-1)
        return self.embedding_combine(combined)
    
    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        biometrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tokens: Audio tokens (batch, n_codebooks, seq_len)
            waveform: Raw audio (batch, channels, samples)
            labels: Target tokens for loss
            biometrics: Dict with glucose, hrv, stress_level, time_of_day
            
        Returns:
            Dict with logits, loss, hidden_states
        """
        # Get tokens from waveform if needed
        if tokens is None and waveform is not None:
            tokens = self.tokenizer.encode(waveform, sample_rate=self.sample_rate)
        
        if tokens is None:
            raise ValueError("Must provide tokens or waveform")
        
        # Embed tokens
        embeddings = self.embed_tokens(tokens)
        batch, seq_len, _ = embeddings.shape
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=embeddings.device)
        embeddings = embeddings + self.pos_embedding(positions)
        
        # Add biofeedback conditioning
        if self.use_biofeedback and biometrics is not None:
            bio_emb = self.biofeedback_embedding(
                glucose=biometrics.get("glucose"),
                hrv=biometrics.get("hrv"),
                stress_level=biometrics.get("stress_level"),
                time_of_day=biometrics.get("time_of_day"),
            )
            # Expand to sequence length and combine
            bio_emb = bio_emb.unsqueeze(1).expand(-1, seq_len, -1)
            embeddings = self.biofeedback_proj(
                torch.cat([embeddings, bio_emb], dim=-1)
            )
        
        # Pass through Mamba layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        
        # Compute logits for each codebook
        logits = torch.stack([
            head(hidden_states) for head in self.lm_heads
        ], dim=1)  # (batch, n_codebooks, seq_len, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :, :-1, :].contiguous()
            shift_labels = labels[:, :, 1:].contiguous()
            
            # Loss per codebook, averaged
            losses = []
            for i in range(self.n_codebooks):
                cb_loss = F.cross_entropy(
                    shift_logits[:, i].reshape(-1, self.vocab_size),
                    shift_labels[:, i].reshape(-1),
                    ignore_index=-100,
                )
                losses.append(cb_loss)
            loss = torch.stack(losses).mean()
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
        }
    
    @torch.no_grad()
    def generate(
        self,
        duration_seconds: float = 30.0,
        prompt_waveform: Optional[torch.Tensor] = None,
        biometrics: Optional[BiometricData] = None,
        music_params: Optional[MusicParameters] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate therapeutic audio.
        
        Args:
            duration_seconds: Target duration
            prompt_waveform: Optional audio prompt
            biometrics: User's biometric data
            music_params: Music parameters (from BiofeedbackController)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            waveform: Generated audio (1, channels, samples)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Calculate target tokens
        n_tokens = self.tokenizer.get_n_tokens(duration_seconds)
        
        # Get biofeedback conditioning
        bio_dict = None
        if biometrics is not None and self.use_biofeedback:
            bio_dict = self._biometrics_to_tensors(biometrics, device)
        
        # Get initial tokens from prompt or random
        if prompt_waveform is not None:
            current_tokens = self.tokenizer.encode(
                prompt_waveform, sample_rate=self.sample_rate
            )
        else:
            # Start with random tokens
            current_tokens = torch.randint(
                0, self.vocab_size,
                (1, self.n_codebooks, 1),
                device=device
            )
        
        # Generate tokens autoregressively
        for _ in range(n_tokens - current_tokens.shape[2]):
            # Get logits
            outputs = self(tokens=current_tokens, biometrics=bio_dict)
            next_logits = outputs["logits"][:, :, -1, :]  # (batch, n_cb, vocab)
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Sample each codebook
            next_tokens = []
            for i in range(self.n_codebooks):
                cb_logits = next_logits[:, i, :]
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = cb_logits < torch.topk(cb_logits, top_k)[0][..., -1, None]
                    cb_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(cb_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    cb_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(cb_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                next_tokens.append(token)
            
            # Stack and append
            next_tokens = torch.stack(next_tokens, dim=1)  # (batch, n_cb, 1)
            current_tokens = torch.cat([current_tokens, next_tokens], dim=2)
        
        # Decode tokens to audio
        waveform = self.tokenizer.decode(current_tokens)
        
        return waveform
    
    def _biometrics_to_tensors(
        self,
        biometrics: BiometricData,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Convert BiometricData to tensor dict."""
        bio_dict = {}
        
        if biometrics.glucose_mg_dl is not None:
            normalized = BiofeedbackEmbedding.normalize_glucose(biometrics.glucose_mg_dl)
            bio_dict["glucose"] = torch.tensor([normalized], device=device)
        
        if biometrics.hrv_ms is not None:
            normalized = BiofeedbackEmbedding.normalize_hrv(biometrics.hrv_ms)
            bio_dict["hrv"] = torch.tensor([normalized], device=device)
        
        stress = biometrics.get_stress_level()
        from src.therapeutic.biofeedback import StressLevel
        stress_idx = list(StressLevel).index(stress)
        bio_dict["stress_level"] = torch.tensor([stress_idx], device=device)
        
        time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
        if biometrics.time_of_day in time_map:
            bio_dict["time_of_day"] = torch.tensor(
                [time_map[biometrics.time_of_day]], device=device
            )
        
        return bio_dict
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SiMBATherapeutic":
        """Create model from config dict."""
        model_cfg = config.get("model", {})
        audio_cfg = config.get("audio", {})
        therapeutic_cfg = config.get("therapeutic", {})
        
        return cls(
            d_model=model_cfg.get("d_model", 512),
            n_layers=model_cfg.get("n_layers", 8),
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
            encodec_bandwidth=audio_cfg.get("encodec_bandwidth", 6.0),
            sample_rate=audio_cfg.get("sample_rate", 24000),
            max_seq_len=model_cfg.get("max_seq_len", 8192),
            use_biofeedback=therapeutic_cfg.get("use_biofeedback", True),
        )
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
