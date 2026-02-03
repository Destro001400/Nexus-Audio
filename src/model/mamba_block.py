"""
Mamba Block Implementation for SiMBA

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
Paper: https://arxiv.org/abs/2312.00752

This implements the core selective state space (S6) mechanism with O(L) complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space mechanism.
    
    Key components:
    - Input projection with gating
    - 1D causal convolution for local context
    - Selective SSM (S6) for global context with O(L) complexity
    - Output projection
    
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
        
        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
            
        # Input projection: x -> (z, x_proj)
        # z is for gating, x_proj goes through SSM
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Causal 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )
        
        # SSM parameters projection
        # Projects to: delta (dt_rank), B (d_state), C (d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Delta (timestep) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias for stability
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            
        # Delta bias initialization (log-uniform between dt_min and dt_max)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        # S4D real-valued initialization for A
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Causal conv1d
        x_proj = rearrange(x_proj, "b l d -> b d l")
        x_proj = self.conv1d(x_proj)[:, :, :seq_len]  # Causal: remove padding
        x_proj = rearrange(x_proj, "b d l -> b l d")
        
        # Activation
        x_proj = F.silu(x_proj)
        
        # SSM
        y = self.ssm(x_proj)
        
        # Gating with z
        z = F.silu(z)
        output = y * z
        
        # Output projection
        output = self.out_proj(output)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model (S6) forward pass.
        
        Implements the discretized state space recurrence:
            h_t = Ā * h_{t-1} + B̄ * x_t
            y_t = C * h_t + D * x_t
            
        Where Ā and B̄ are discretized using Zero-Order Hold (ZOH).
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_inner)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        
        # Get A from log (always negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Project x to get delta, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Compute delta (timestep)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Discretization using ZOH
        # deltaA = exp(delta * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        
        # deltaB = delta * B (simplified, exact: (exp(delta*A) - I) / A * B)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Selective scan (parallel implementation)
        y = self.selective_scan(x, deltaA, deltaB, C)
        
        # Add D skip connection
        y = y + x * self.D
        
        return y
    
    def selective_scan(
        self,
        x: torch.Tensor,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel selective scan implementation.
        
        This is done sequentially for clarity, but can be parallelized
        using associative scan for GPU efficiency.
        
        Args:
            x: Input (B, L, d_inner)
            deltaA: Discretized A (B, L, d_inner, d_state)
            deltaB: Discretized B (B, L, d_inner, d_state)
            C: Output projection (B, L, d_state)
            
        Returns:
            Output (B, L, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = deltaA.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Output accumulator
        outputs = []
        
        # Sequential scan (can be parallelized with associative scan)
        for t in range(seq_len):
            # Update state: h = A_bar * h + B_bar * x
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t, :, None]
            
            # Compute output: y = C * h
            y = (h * C[:, t, None, :]).sum(dim=-1)  # (B, d_inner)
            outputs.append(y)
            
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        return y


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
    """Mamba block with residual connection and normalization."""
    
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
