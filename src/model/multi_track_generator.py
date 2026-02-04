"""
Multi-Track Music Generator

Generates music as separate stems (bass, drums, melody, ambient)
for better therapeutic control and mixing flexibility.

Based on research: Multi-track generation allows precise control
over therapeutic elements (bass for insulin, HFC in melody, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class StemType(Enum):
    """Types of audio stems."""
    BASS = "bass"           # Low frequencies (50-200 Hz) - therapeutic bass
    DRUMS = "drums"         # Percussion/rhythm
    MELODY = "melody"       # Main melodic content, preserves HFC
    HARMONY = "harmony"     # Chords, pads
    AMBIENT = "ambient"     # Atmosphere, nature sounds, binaural
    VOCALS = "vocals"       # Optional vocal content


@dataclass
class StemConfig:
    """Configuration for a single stem."""
    stem_type: StemType
    freq_range: Tuple[float, float]  # Hz range for this stem
    default_gain_db: float = 0.0
    therapeutic_priority: int = 0   # Higher = more important for therapy
    
    @classmethod
    def therapeutic_defaults(cls) -> Dict[StemType, "StemConfig"]:
        """Get default therapeutic stem configurations."""
        return {
            StemType.BASS: cls(
                stem_type=StemType.BASS,
                freq_range=(20, 200),
                default_gain_db=0.0,
                therapeutic_priority=3,  # Highest - insulin release
            ),
            StemType.DRUMS: cls(
                stem_type=StemType.DRUMS,
                freq_range=(20, 8000),
                default_gain_db=0.0,
                therapeutic_priority=1,  # BPM control
            ),
            StemType.MELODY: cls(
                stem_type=StemType.MELODY,
                freq_range=(200, 20000),
                default_gain_db=0.0,
                therapeutic_priority=2,  # HFC preservation
            ),
            StemType.AMBIENT: cls(
                stem_type=StemType.AMBIENT,
                freq_range=(20, 20000),
                default_gain_db=-6.0,
                therapeutic_priority=2,  # Binaural carrier
            ),
        }


class StemOutput(NamedTuple):
    """Output from multi-track generation."""
    stems: Dict[StemType, torch.Tensor]  # Each stem waveform
    sample_rate: int
    duration_seconds: float
    
    def get_stem(self, stem_type: StemType) -> Optional[torch.Tensor]:
        """Get a specific stem."""
        return self.stems.get(stem_type)
    
    def mix(self, gains: Optional[Dict[StemType, float]] = None) -> torch.Tensor:
        """Mix all stems with optional gain adjustments (in dB)."""
        if gains is None:
            gains = {st: 0.0 for st in self.stems.keys()}
        
        mixed = None
        for stem_type, waveform in self.stems.items():
            gain_linear = 10 ** (gains.get(stem_type, 0.0) / 20)
            adjusted = waveform * gain_linear
            
            if mixed is None:
                mixed = adjusted
            else:
                # Pad to same length if needed
                if adjusted.shape[-1] > mixed.shape[-1]:
                    mixed = F.pad(mixed, (0, adjusted.shape[-1] - mixed.shape[-1]))
                elif mixed.shape[-1] > adjusted.shape[-1]:
                    adjusted = F.pad(adjusted, (0, mixed.shape[-1] - adjusted.shape[-1]))
                mixed = mixed + adjusted
        
        # Normalize to prevent clipping
        if mixed is not None:
            max_val = mixed.abs().max()
            if max_val > 1.0:
                mixed = mixed / max_val * 0.95
        
        return mixed


class MultiTrackHead(nn.Module):
    """
    Separate prediction heads for each stem type.
    
    Each head learns to generate tokens for its specific stem,
    allowing the model to learn stem-specific patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_codebooks: int,
        stems: List[StemType],
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.stems = stems
        
        # Separate projection for each stem
        self.stem_projections = nn.ModuleDict({
            stem.value: nn.Linear(d_model, d_model)
            for stem in stems
        })
        
        # Separate LM heads for each stem and codebook
        self.lm_heads = nn.ModuleDict({
            stem.value: nn.ModuleList([
                nn.Linear(d_model, vocab_size)
                for _ in range(n_codebooks)
            ])
            for stem in stems
        })
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        stem_type: StemType,
    ) -> torch.Tensor:
        """
        Compute logits for a specific stem.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            stem_type: Which stem to predict
            
        Returns:
            logits: (batch, n_codebooks, seq_len, vocab_size)
        """
        # Project for this stem
        projected = self.stem_projections[stem_type.value](hidden_states)
        
        # Get logits from each codebook head
        logits = torch.stack([
            head(projected) 
            for head in self.lm_heads[stem_type.value]
        ], dim=1)
        
        return logits


class StemEmbedding(nn.Module):
    """Learned embedding for each stem type to condition generation."""
    
    def __init__(self, n_stems: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_stems, d_model)
        self.n_stems = n_stems
    
    def forward(self, stem_idx: torch.Tensor) -> torch.Tensor:
        """Get stem embedding."""
        return self.embedding(stem_idx)


class MultiTrackGenerator(nn.Module):
    """
    Generates multiple audio stems in parallel or sequentially.
    
    Architecture options:
    1. Parallel: Generate all stems from same hidden states
    2. Sequential: Generate stems one by one, conditioning on previous
    3. Interleaved: Alternate between stems during generation
    
    This implementation uses parallel generation with stem-specific heads.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 1024,
        n_codebooks: int = 8,
        stems: Optional[List[StemType]] = None,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.sample_rate = sample_rate
        
        # Default therapeutic stems
        if stems is None:
            stems = [StemType.BASS, StemType.DRUMS, StemType.MELODY, StemType.AMBIENT]
        self.stems = stems
        
        # Stem embedding for conditioning
        self.stem_embedding = StemEmbedding(len(stems), d_model)
        self.stem_to_idx = {stem: i for i, stem in enumerate(stems)}
        
        # Multi-track prediction heads
        self.multi_head = MultiTrackHead(
            d_model=d_model,
            vocab_size=vocab_size,
            n_codebooks=n_codebooks,
            stems=stems,
        )
        
        # Stem configurations
        self.stem_configs = StemConfig.therapeutic_defaults()
    
    def get_stem_conditioning(
        self,
        batch_size: int,
        stem_type: StemType,
        device: torch.device,
    ) -> torch.Tensor:
        """Get conditioning embedding for a stem type."""
        stem_idx = torch.tensor(
            [self.stem_to_idx[stem_type]], 
            device=device
        ).expand(batch_size)
        return self.stem_embedding(stem_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_stems: Optional[List[StemType]] = None,
    ) -> Dict[StemType, torch.Tensor]:
        """
        Compute logits for multiple stems.
        
        Args:
            hidden_states: (batch, seq_len, d_model) from backbone
            target_stems: Which stems to compute (default: all)
            
        Returns:
            Dict mapping StemType to logits (batch, n_cb, seq_len, vocab)
        """
        if target_stems is None:
            target_stems = self.stems
        
        results = {}
        for stem_type in target_stems:
            # Add stem conditioning
            stem_cond = self.get_stem_conditioning(
                hidden_states.shape[0],
                stem_type,
                hidden_states.device,
            )
            conditioned = hidden_states + stem_cond.unsqueeze(1)
            
            # Get logits for this stem
            results[stem_type] = self.multi_head(conditioned, stem_type)
        
        return results
    
    @torch.no_grad()
    def generate_stems(
        self,
        backbone_model: nn.Module,
        tokenizer: nn.Module,
        duration_seconds: float,
        biometrics: Optional[Dict] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        target_stems: Optional[List[StemType]] = None,
    ) -> StemOutput:
        """
        Generate all stems for a piece of music.
        
        Args:
            backbone_model: The SiMBA backbone for hidden states
            tokenizer: EnCodec tokenizer for decoding
            duration_seconds: Target duration
            biometrics: Optional biometric conditioning
            temperature: Sampling temperature
            top_k: Top-k sampling
            target_stems: Which stems to generate
            
        Returns:
            StemOutput with all generated stems
        """
        if target_stems is None:
            target_stems = self.stems
        
        device = next(self.parameters()).device
        n_tokens = tokenizer.get_n_tokens(duration_seconds)
        
        stems = {}
        
        for stem_type in target_stems:
            print(f"  Generating {stem_type.value} stem...")
            
            # Start with random token
            current_tokens = torch.randint(
                0, self.vocab_size,
                (1, self.n_codebooks, 1),
                device=device
            )
            
            # Generate tokens for this stem
            for _ in range(n_tokens - 1):
                # Get embeddings from backbone
                # This would call into the main model
                # For now, simplified version
                embeddings = backbone_model.embed_tokens(current_tokens)
                
                # Add positional encoding
                positions = torch.arange(
                    current_tokens.shape[2], device=device
                )
                embeddings = embeddings + backbone_model.pos_embedding(positions)
                
                # Add stem conditioning
                stem_cond = self.get_stem_conditioning(1, stem_type, device)
                embeddings = embeddings + stem_cond.unsqueeze(1)
                
                # Pass through backbone layers
                hidden = embeddings
                for layer in backbone_model.layers:
                    hidden = layer(hidden)
                hidden = backbone_model.norm(hidden)
                
                # Get logits for this stem
                logits_dict = self.forward(hidden, [stem_type])
                next_logits = logits_dict[stem_type][:, :, -1, :]  # (1, n_cb, vocab)
                
                # Sample each codebook
                next_tokens = []
                for i in range(self.n_codebooks):
                    cb_logits = next_logits[:, i, :] / temperature
                    
                    # Top-k
                    if top_k > 0:
                        indices_to_remove = cb_logits < torch.topk(cb_logits, top_k)[0][..., -1, None]
                        cb_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(cb_logits, dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                    next_tokens.append(token)
                
                next_tokens = torch.stack(next_tokens, dim=1)
                current_tokens = torch.cat([current_tokens, next_tokens], dim=2)
            
            # Decode to audio
            waveform = tokenizer.decode(current_tokens)
            stems[stem_type] = waveform
        
        return StemOutput(
            stems=stems,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
        )


def create_therapeutic_mix(
    stem_output: StemOutput,
    bass_boost_db: float = 6.0,
    melody_preserve_hfc: bool = True,
    ambient_level_db: float = -6.0,
) -> torch.Tensor:
    """
    Create a therapeutic mix from stems.
    
    Args:
        stem_output: Generated stems
        bass_boost_db: Bass enhancement in dB
        melody_preserve_hfc: Preserve high frequencies in melody
        ambient_level_db: Ambient stem level
        
    Returns:
        Mixed waveform
    """
    gains = {
        StemType.BASS: bass_boost_db,
        StemType.DRUMS: 0.0,
        StemType.MELODY: 0.0,
        StemType.AMBIENT: ambient_level_db,
    }
    
    return stem_output.mix(gains)
