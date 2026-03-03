"""
MS-SSM Block — Multi-Scale Selective State Space Model
Nexus-Audio Beta | SnaX Company

Inspirado em: "MS-SSM: A Multi-Scale State Space Model for Efficient
Sequence Modeling" (COLM 2025)

Ideia central: em vez de um único SSM "resumindo" tudo numa resolução,
usamos 3 SSMs em paralelo, cada um ouvindo o áudio numa granularidade
diferente:

    Escala FINA  (d_state=16)  → notas individuais, transientes rápidos
    Escala MÉDIA (d_state=64)  → frases musicais, progressões de acorde
    Escala GROSSA(d_state=256) → seções inteiras, estrutura da música

Um "Scale Mixer" decide dinamicamente, pra cada token, qual escala
merece mais atenção — exatamente como um produtor musical move os
faders de acordo com o momento da música.

Complexidade: O(L) — mantém a eficiência linear dos SSMs originais.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .mamba_block import MambaBlock, RMSNorm


class MSSSMBlock(nn.Module):
    """
    Multi-Scale SSM: 3 MambaBlocks em paralelo com scale mixer dinâmico.

    Args:
        d_model:  Dimensão do modelo (igual ao SiMBA atual).
        scales:   Lista com d_state de cada escala. Default SnaX: [16, 64, 256].
        d_conv:   Kernel de convolução causal (igual ao Mamba base).
        expand:   Fator de expansão interno do Mamba.
        dropout:  Dropout aplicado antes do mixer (regularização).
        mixer_hidden: Tamanho da camada oculta do Scale Mixer.
    """

    def __init__(
        self,
        d_model: int = 512,
        scales: List[int] = [16, 64, 256],
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        mixer_hidden: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        self.n_scales = len(scales)

        # ── 3 SSMs independentes, um por escala ───────────────────────────────
        # Cada um tem seu próprio d_state, adaptado à sua resolução temporal.
        # Escala fina: estado pequeno, esquece rápido (detalhe imediato).
        # Escala grossa: estado grande, memória longa (contexto global).
        self.ssm_branches = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for d_state in scales
        ])

        # ── Scale Mixer input-dependent ────────────────────────────────────────
        # Recebe o input original + saída das 3 escalas concatenados.
        # Aprende a "mixar" as escalas dinamicamente por token.
        # Analogia: o produtor decide em tempo real qual frequência de memória
        # é mais relevante pra cada momento da música.
        self.scale_mixer = nn.Sequential(
            nn.Linear(d_model * (self.n_scales + 1), mixer_hidden),
            nn.SiLU(),
            nn.Linear(mixer_hidden, self.n_scales),
        )

        # Dropout anti-overfitting (aplicado nas saídas de cada branch)
        self.dropout = nn.Dropout(dropout)

        # Inicializa o mixer com pesos uniformes (começa igualando as escalas)
        nn.init.zeros_(self.scale_mixer[-1].weight)
        nn.init.zeros_(self.scale_mixer[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            out: (batch, seq_len, d_model)
        """
        # ── Passa pelas 3 escalas em paralelo ─────────────────────────────────
        branch_outputs = []
        for ssm in self.ssm_branches:
            h = ssm(x)                    # (B, L, d_model)
            h = self.dropout(h)
            branch_outputs.append(h)

        # ── Scale Mixer: decide quanto de cada escala usar ─────────────────────
        # Concatena input + saídas das 3 escalas → (B, L, d_model * 4)
        mixer_input = torch.cat([x] + branch_outputs, dim=-1)

        # Pesos por token: (B, L, n_scales) — cada token tem seus pesos!
        weights = F.softmax(self.scale_mixer(mixer_input), dim=-1)

        # Combina as escalas com os pesos aprendidos
        # weights[:, :, i] → peso da escala i para cada posição
        out = sum(
            weights[:, :, i].unsqueeze(-1) * branch_outputs[i]
            for i in range(self.n_scales)
        )

        return out


class ResidualMSSSMBlock(nn.Module):
    """
    MS-SSM com conexão residual e RMSNorm — drop-in replacement do ResidualMambaBlock.

    Uso:
        # Antes (v3):
        self.layers = nn.ModuleList([ResidualMambaBlock(...) for _ in range(n)])

        # Depois (Beta):
        self.layers = nn.ModuleList([ResidualMSSSMBlock(...) for _ in range(n)])
    """

    def __init__(
        self,
        d_model: int = 512,
        scales: List[int] = [16, 64, 256],
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ms_ssm = MSSSMBlock(
            d_model=d_model,
            scales=scales,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual (mesmo padrão do ResidualMambaBlock original)
        return x + self.ms_ssm(self.norm(x))
