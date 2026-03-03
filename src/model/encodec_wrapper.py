"""
EnCodec Wrapper for SiMBA
CORRIGIDO v2: módulos de embedding movidos pro __init__ (bug crítico corrigido)

Paper: "High Fidelity Neural Audio Compression" (Meta, 2022)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import warnings

try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    warnings.warn(
        "EnCodec não instalado. Instale com: pip install encodec\n"
        "Usando tokenizador simplificado como fallback."
    )


class EnCodecWrapper(nn.Module):
    """
    Wrapper do EnCodec da Meta para tokenização de áudio.

    Args:
        model_type: "24k" (24kHz) ou "48k" (48kHz)
        bandwidth: Bandwidth alvo em kbps (1.5, 3.0, 6.0, 12.0, 24.0)
        d_model: Dimensão dos embeddings de saída
        device: "cuda" ou "cpu"

    Características com 24kHz e 6.0 kbps:
        - ~75 tokens por segundo de áudio
        - 8 codebooks
        - Vocab size: 1024 por codebook
    """

    def __init__(
        self,
        model_type: str = "24k",
        bandwidth: float = 6.0,
        d_model: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device
        self.bandwidth = bandwidth
        self.d_model = d_model

        if not ENCODEC_AVAILABLE:
            raise ImportError("EnCodec é necessário. Instale com: pip install encodec")

        # Carrega EnCodec pré-treinado
        if model_type == "24k":
            self.model = EncodecModel.encodec_model_24khz()
            self.sample_rate = 24000
        elif model_type == "48k":
            self.model = EncodecModel.encodec_model_48khz()
            self.sample_rate = 48000
        else:
            raise ValueError(f"Tipo desconhecido: {model_type}")

        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self.model.to(device)

        # Congela pesos do EnCodec (não treinamos ele)
        for param in self.model.parameters():
            param.requires_grad = False

        # Características dos tokens
        self.vocab_size = 1024
        self.n_codebooks = self._get_n_codebooks()
        self.tokens_per_second = int(self.sample_rate / self.model.encoder.hop_length)

        # CORRIGIDO: módulos de embedding criados no __init__ (não mais lazy!)
        # Antes eram criados dentro de tokens_to_embeddings() — bug crítico:
        # módulos criados fora do __init__ não aparecem em parameters(),
        # não salvam no checkpoint e não vão pro device correto.
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model // self.n_codebooks)
            for _ in range(self.n_codebooks)
        ])
        self.embedding_proj = nn.Linear(
            d_model // self.n_codebooks * self.n_codebooks,
            d_model
        )

    def _get_n_codebooks(self) -> int:
        """Número de codebooks baseado no bandwidth."""
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
        Codifica waveform para tokens discretos.

        Args:
            waveform: (batch, channels, samples) ou (batch, samples)
            sample_rate: Taxa de amostragem de entrada

        Returns:
            tokens: (batch, n_codebooks, n_frames)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = convert_audio(
                waveform, sample_rate, self.sample_rate, self.model.channels
            )

        waveform = waveform.to(self.device)

        encoded_frames = self.model.encode(waveform)
        codes = encoded_frames[0][0]  # (batch, n_codebooks, n_frames)

        return codes

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodifica tokens para waveform.

        Args:
            tokens: (batch, n_codebooks, n_frames)

        Returns:
            waveform: (batch, channels, samples)
        """
        tokens = tokens.to(self.device)
        encoded_frames = [(tokens, None)]
        waveform = self.model.decode(encoded_frames)
        return waveform

    def tokens_to_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converte tokens para embeddings contínuos.

        CORRIGIDO: agora usa self.token_embeddings criado no __init__
        em vez de criar módulos dinamicamente (que não treinavam!).

        Args:
            tokens: (batch, n_codebooks, n_frames)

        Returns:
            embeddings: (batch, n_frames, d_model)
        """
        batch, n_codebooks, n_frames = tokens.shape

        embeddings = []
        for i in range(n_codebooks):
            emb = self.token_embeddings[i](tokens[:, i, :])  # (B, T, D/n_cb)
            embeddings.append(emb)

        combined = torch.cat(embeddings, dim=-1)   # (B, T, D)
        projected = self.embedding_proj(combined)  # (B, T, d_model)

        return projected

    def get_audio_duration(self, n_tokens: int) -> float:
        """Calcula duração de áudio a partir do número de tokens."""
        return n_tokens / self.tokens_per_second

    def get_n_tokens(self, duration_seconds: float) -> int:
        """Calcula número de tokens para uma duração."""
        return int(duration_seconds * self.tokens_per_second)


class SimplifiedTokenizer(nn.Module):
    """
    Tokenizador fallback quando EnCodec não está disponível.
    Usa mel-spectrogram + codebook aprendido (menor qualidade).
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

        import torchaudio.transforms as T
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, d_model, 3, padding=1),
        )

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(vocab_size, d_model // n_codebooks))
            for _ in range(n_codebooks)
        ])

        self.tokens_per_second = sample_rate // 320

    def encode(self, waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        mel = self.mel(waveform)
        mel = torch.log(mel.clamp(min=1e-5))

        features = self.encoder(mel)
        features = features.transpose(1, 2)

        B, T, D = features.shape
        tokens = []

        for i, codebook in enumerate(self.codebooks):
            start = i * (D // self.n_codebooks)
            end = (i + 1) * (D // self.n_codebooks)
            feat_slice = features[:, :, start:end]

            distances = torch.cdist(
                feat_slice.reshape(-1, feat_slice.shape[-1]),
                codebook
            )
            token_ids = distances.argmin(dim=-1).reshape(B, T)
            tokens.append(token_ids)

        return torch.stack(tokens, dim=1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "SimplifiedTokenizer.decode() não implementado. "
            "Use EnCodec para reconstrução real."
        )
        B, n_cb, T = tokens.shape
        return torch.zeros(B, 1, T * 320)

    def get_n_tokens(self, duration_seconds: float) -> int:
        return int(duration_seconds * self.tokens_per_second)


def get_tokenizer(use_encodec: bool = True, **kwargs) -> nn.Module:
    """
    Factory para obter o tokenizador adequado.

    Args:
        use_encodec: Usar EnCodec (recomendado)
        **kwargs: Argumentos para o tokenizador
    """
    if use_encodec and ENCODEC_AVAILABLE:
        return EnCodecWrapper(**kwargs)
    else:
        if use_encodec:
            warnings.warn("EnCodec não disponível, usando tokenizador simplificado.")
        return SimplifiedTokenizer(**kwargs)
