"""
Mamba Block Implementation for SiMBA
CORRIGIDO v2: selective_scan paralelizado (sem loop Python!)

Baseado em: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
Paper: https://arxiv.org/abs/2312.00752

MUDANÇA PRINCIPAL: selective_scan agora usa cumsum vetorizado no lugar de
um loop Python por timestep. Resultado esperado: 8-12x mais rápido no treino.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space mechanism.

    Args:
        d_model: Model dimension
        d_state: SSM state expansion factor (N in paper)
        d_conv: Local convolution width
        expand: Block expansion factor (E in paper)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = rearrange(x_proj, "b l d -> b d l")
        x_proj = self.conv1d(x_proj)[:, :, :seq_len]
        x_proj = rearrange(x_proj, "b d l -> b l d")

        x_proj = F.silu(x_proj)

        y = self.ssm(x_proj)

        z = F.silu(z)
        output = y * z

        output = self.out_proj(output)

        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_inner = x.shape

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # CORRIGIDO: scan paralelo vetorizado (sem loop Python!)
        y = self.selective_scan_parallel(x, deltaA, deltaB, C)

        y = y + x * self.D

        return y

    def selective_scan_parallel(
        self,
        x: torch.Tensor,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scan seletivo PARALELO usando cumsum vetorizado.

        Fórmula: h_t = A_t * h_{t-1} + B_t * x_t
        Solução fechada: h_t = cumA_t * cumsum(B_s * x_s / cumA_{s-1})

        Onde cumA_t = prod_{u=0}^{t} A_u (produto cumulativo de A)

        Complexidade: O(L) operações, mas todas vetorizadas na GPU.
        Speedup esperado vs loop Python: 8-12x na T4/RTX.

        Args:
            x: (B, L, D)
            deltaA: (B, L, D, N) — A discretizado
            deltaB: (B, L, D, N) — B discretizado
            C: (B, L, N) — projeção de saída

        Returns:
            y: (B, L, D)
        """
        B, L, D, N = deltaA.shape

        # Produto cumulativo de A em log-space (estabilidade numérica)
        # deltaA está em (0, 1) pois A é negativo
        log_deltaA = torch.log(deltaA.clamp(min=1e-6))         # (B, L, D, N)
        cum_log_deltaA = torch.cumsum(log_deltaA, dim=1)        # (B, L, D, N)
        cum_deltaA = torch.exp(cum_log_deltaA)                  # (B, L, D, N)

        # B_t * x_t: contribuição de cada timestep
        Bx = deltaB * x.unsqueeze(-1)                          # (B, L, D, N)

        # cumA deslocado: posição t usa cumA[t-1] (antes da multiplicação de t)
        cum_deltaA_prev = torch.cat([
            torch.ones(B, 1, D, N, device=x.device, dtype=x.dtype),
            cum_deltaA[:, :-1, :, :]
        ], dim=1)                                               # (B, L, D, N)

        # Normaliza Bx pelo cumA anterior para "desfazer" o efeito acumulado
        Bx_norm = Bx / (cum_deltaA_prev + 1e-8)               # (B, L, D, N)

        # Soma cumulativa paralela (a parte que seria o loop)
        h_norm = torch.cumsum(Bx_norm, dim=1)                  # (B, L, D, N)

        # Re-aplica o produto cumulativo para obter os hidden states reais
        h = cum_deltaA * h_norm                                # (B, L, D, N)

        # Saída: y_t = sum_n C_t[n] * h_t[:, n]
        y = (h * C.unsqueeze(2)).sum(dim=-1)                   # (B, L, D)

        return y

    def selective_scan_sequential(
        self,
        x: torch.Tensor,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Versão sequencial original (mantida como fallback/referência).
        Use selective_scan_parallel para treino — esta é só pra debug.
        """
        batch, seq_len, d_inner = x.shape
        d_state = deltaA.shape[-1]

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t, :, None]
            y = (h * C[:, t, None, :]).sum(dim=-1)
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ResidualMambaBlock(nn.Module):
    """Mamba block com residual e normalização."""

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))
