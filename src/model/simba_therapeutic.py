"""
SiMBA Therapeutic Music Generation Model
BETA v4 — Motor MS-SSM (Multi-Scale Selective State Space)

Mudança principal em relação ao v3:
- ResidualMambaBlock substituído por ResidualMSSSMBlock
- 3 SSMs em paralelo com escalas [16, 64, 256] por bloco
- Scale Mixer input-dependent: modelo aprende qual escala focar por token
- Dropout=0.1 nativo no bloco MS-SSM (anti-overfitting)

Mantidos do v3 (não mexemos no que funciona):
1. Embeddings somados por codebook (d_model completo por CB)
2. Cabeças hierárquicas RVQ (CB(i+1) vê CB(i))
3. Loss ponderada por codebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import warnings

from .mamba_block import RMSNorm
from .ms_ssm_block import ResidualMSSSMBlock
from .encodec_wrapper import EnCodecWrapper, get_tokenizer, ENCODEC_AVAILABLE

try:
    from ..therapeutic.biofeedback import (
        BiofeedbackEmbedding,
        BiofeedbackController,
        BiometricData,
        MusicParameters,
        StressLevel,
    )
    BIOFEEDBACK_AVAILABLE = True
except ImportError:
    BIOFEEDBACK_AVAILABLE = False
    warnings.warn(
        "Módulo biofeedback não disponível. "
        "Treinando sem condicionamento fisiológico."
    )
    BiometricData = None
    MusicParameters = None


class SiMBATherapeutic(nn.Module):
    """
    SiMBA Therapeutic — Beta com motor MS-SSM.

    Arquitetura Beta (MS-SSM):
        EnCodec Tokens → Token Embedding (soma por CB, d_model completo)
        → MS-SSM Blocks × N (3 escalas em paralelo + scale mixer)
        → Cabeças Hierárquicas (CB(i+1) vê CB(i))
        → Logits ponderados → EnCodec Decoder → Waveform

    Args:
        d_model:            Dimensão oculta.
        n_layers:           Número de blocos MS-SSM.
        ssm_scales:         d_state de cada escala. Default SnaX: [16, 64, 256].
        d_conv:             Kernel de convolução causal.
        expand:             Fator de expansão interno do Mamba.
        dropout:            Dropout no Scale Mixer (anti-overfitting).
        encodec_bandwidth:  Bandwidth do EnCodec (6.0 kbps recomendado).
        sample_rate:        Taxa de amostragem (24000 pro EnCodec).
        max_seq_len:        Comprimento máximo de sequência.
        use_biofeedback:    Habilitar condicionamento biofeedback.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        ssm_scales: List[int] = [16, 64, 256],
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
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
        self.device_str = device
        self.ssm_scales = ssm_scales

        # Inicializa tokenizador EnCodec
        self.tokenizer = get_tokenizer(
            use_encodec=ENCODEC_AVAILABLE,
            bandwidth=encodec_bandwidth,
            d_model=d_model,
            device=device,
        )

        self.vocab_size = self.tokenizer.vocab_size
        self.n_codebooks = self.tokenizer.n_codebooks

        # ✅ FIX 1: d_model completo por codebook (era d_model//8 = 64 dims)
        # Cada codebook tem 1024 tokens — merece dimensão completa pra representar bem
        # Soma os embeddings em vez de concatenar (sem projeção extra necessária)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model)
            for _ in range(self.n_codebooks)
        ])

        # Embedding posicional
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Condicionamento biofeedback
        if self.use_biofeedback:
            self.biofeedback_embedding = BiofeedbackEmbedding(d_model)
            self.biofeedback_proj = nn.Linear(d_model * 2, d_model)

        # Camadas MS-SSM (3 escalas em paralelo por bloco)
        self.layers = nn.ModuleList([
            ResidualMSSSMBlock(
                d_model=d_model,
                scales=ssm_scales,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Normalização final
        self.norm = RMSNorm(d_model)

        # Cabeças de predição (uma por codebook)
        self.lm_heads = nn.ModuleList([
            nn.Linear(d_model, self.vocab_size, bias=False)
            for _ in range(self.n_codebooks)
        ])

        # ✅ FIX 2: Bridges hierárquicos — CB(i+1) aprende com o que CB(i) previu
        # Cada bridge projeta logits do CB(i) de volta pra d_model
        # Inicializados com std=0.001 pra não perturbar o treino no início
        self.cb_bridges = nn.ModuleList([
            nn.Linear(self.vocab_size, d_model, bias=False)
            for _ in range(self.n_codebooks - 1)
        ])

        # ✅ FIX 3: Pesos de loss decrescentes (CB1 mais importante no RVQ)
        # CB1 captura estrutura geral → peso 2.0 | CB8 detalhe mínimo → peso 0.4
        cb_weights = torch.tensor([2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4])
        self.register_buffer('cb_loss_weights', cb_weights)

        # Inicialização de pesos
        self.apply(self._init_weights)

        # Bridges iniciam quase-zero pra não atrapalhar no começo do treino
        for bridge in self.cb_bridges:
            nn.init.normal_(bridge.weight, std=0.001)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embute tokens multi-codebook somando os embeddings.

        v3: Soma (era concat + linear projection)
        Vantagem: d_model completo pra cada CB sem perda de informação

        Args:
            tokens: (batch, n_codebooks, seq_len)

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch, n_cb, seq_len = tokens.shape

        embeddings = self.token_embeddings[0](tokens[:, 0, :])
        for i in range(1, n_cb):
            embeddings = embeddings + self.token_embeddings[i](tokens[:, i, :])

        return embeddings

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        biometrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass com cabeças hierárquicas.

        Args:
            tokens: (batch, n_codebooks, seq_len)
            waveform: Áudio bruto (batch, channels, samples)
            labels: Tokens alvo (batch, n_codebooks, seq_len)
            biometrics: Dict com glucose, hrv, stress_level, time_of_day

        Returns:
            Dict com logits (batch, n_codebooks, seq_len, vocab_size), loss, hidden_states
        """
        if tokens is None and waveform is not None:
            tokens = self.tokenizer.encode(waveform, sample_rate=self.sample_rate)

        if tokens is None:
            raise ValueError("Forneça tokens ou waveform")

        # ── Embedding ──────────────────────────────────────────────────────────
        embeddings = self.embed_tokens(tokens)
        batch, seq_len, _ = embeddings.shape

        positions = torch.arange(seq_len, device=embeddings.device)
        embeddings = embeddings + self.pos_embedding(positions)

        if self.use_biofeedback and biometrics is not None:
            bio_emb = self.biofeedback_embedding(
                glucose=biometrics.get("glucose"),
                hrv=biometrics.get("hrv"),
                stress_level=biometrics.get("stress_level"),
                time_of_day=biometrics.get("time_of_day"),
            )
            bio_emb = bio_emb.unsqueeze(1).expand(-1, seq_len, -1)
            embeddings = self.biofeedback_proj(
                torch.cat([embeddings, bio_emb], dim=-1)
            )

        # ── Mamba Blocks ───────────────────────────────────────────────────────
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        # ── Cabeças Hierárquicas ───────────────────────────────────────────────
        # CB(i+1) recebe contexto do que CB(i) previu
        # .detach() evita que gradiente de CB(i+1) bagunce o aprendizado de CB(i)
        logits_list = []
        cb_hidden = hidden_states

        for i, head in enumerate(self.lm_heads):
            logits_i = head(cb_hidden)              # (B, T, vocab_size)
            logits_list.append(logits_i)

            if i < self.n_codebooks - 1:
                cb_context = self.cb_bridges[i](logits_i.detach())  # (B, T, d_model)
                cb_hidden = cb_hidden + cb_context

        logits = torch.stack(logits_list, dim=1)    # (B, n_cb, T, vocab_size)

        # ── Loss Ponderada ─────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[:, :, :-1, :].contiguous()
            shift_labels = labels[:, :, 1:].contiguous()

            losses = []
            for i in range(self.n_codebooks):
                cb_loss = F.cross_entropy(
                    shift_logits[:, i].reshape(-1, self.vocab_size),
                    shift_labels[:, i].reshape(-1),
                    ignore_index=-100,
                )
                losses.append(cb_loss)

            weights = self.cb_loss_weights[:self.n_codebooks]
            loss = (torch.stack(losses) * weights).sum() / weights.sum()

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
        biometrics=None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Gera áudio terapêutico autoregressivamente.

        Args:
            duration_seconds: Duração alvo
            prompt_waveform: Áudio inicial opcional
            biometrics: Dados biométricos do usuário
            temperature: Temperatura de amostragem
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            waveform: Áudio gerado (1, channels, samples)
        """
        self.eval()
        device = next(self.parameters()).device
        n_tokens = self.tokenizer.get_n_tokens(duration_seconds)

        bio_dict = None
        if biometrics is not None and self.use_biofeedback:
            bio_dict = self._biometrics_to_tensors(biometrics, device)

        if prompt_waveform is not None:
            current_tokens = self.tokenizer.encode(
                prompt_waveform, sample_rate=self.sample_rate
            )
        else:
            current_tokens = torch.randint(
                0, self.vocab_size,
                (1, self.n_codebooks, 1),
                device=device
            )

        for _ in range(n_tokens - current_tokens.shape[2]):
            outputs = self(tokens=current_tokens, biometrics=bio_dict)
            next_logits = outputs["logits"][:, :, -1, :]  # (1, n_cb, vocab)
            next_logits = next_logits / temperature

            next_tokens = []
            for i in range(self.n_codebooks):
                cb_logits = next_logits[:, i, :]

                if top_k > 0:
                    topk_vals = torch.topk(cb_logits, top_k)[0][..., -1, None]
                    cb_logits = cb_logits.masked_fill(cb_logits < topk_vals, float('-inf'))

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(cb_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits[remove] = float('-inf')
                    cb_logits = torch.zeros_like(cb_logits).scatter_(-1, sorted_idx, sorted_logits)

                probs = F.softmax(cb_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                next_tokens.append(token)

            next_tokens = torch.stack(next_tokens, dim=1)
            current_tokens = torch.cat([current_tokens, next_tokens], dim=2)

        waveform = self.tokenizer.decode(current_tokens)
        return waveform

    def _biometrics_to_tensors(self, biometrics, device) -> Dict[str, torch.Tensor]:
        """Converte BiometricData para dict de tensors."""
        bio_dict = {}

        if biometrics.glucose_mg_dl is not None:
            from ..therapeutic.biofeedback import BiofeedbackEmbedding
            normalized = BiofeedbackEmbedding.normalize_glucose(biometrics.glucose_mg_dl)
            bio_dict["glucose"] = torch.tensor([normalized], device=device)

        if biometrics.hrv_ms is not None:
            from ..therapeutic.biofeedback import BiofeedbackEmbedding
            normalized = BiofeedbackEmbedding.normalize_hrv(biometrics.hrv_ms)
            bio_dict["hrv"] = torch.tensor([normalized], device=device)

        stress = biometrics.get_stress_level()
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
        """Cria modelo a partir de um dict de configuração."""
        model_cfg = config.get("model", {})
        audio_cfg = config.get("audio", {})
        therapeutic_cfg = config.get("therapeutic", {})

        return cls(
            d_model=model_cfg.get("d_model", 512),
            n_layers=model_cfg.get("n_layers", 8),
            ssm_scales=model_cfg.get("ssm_scales", [16, 64, 256]),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
            dropout=model_cfg.get("dropout", 0.1),
            encodec_bandwidth=audio_cfg.get("encodec_bandwidth", 6.0),
            sample_rate=audio_cfg.get("sample_rate", 24000),
            max_seq_len=model_cfg.get("max_seq_len", 8192),
            use_biofeedback=therapeutic_cfg.get("use_biofeedback", True),
        )

    def count_parameters(self) -> int:
        """Conta parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
