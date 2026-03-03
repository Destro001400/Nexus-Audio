# ==============================================================================
# 🎵 NEXUS-AUDIO: GERAÇÃO DE ÁUDIO TERAPÊUTICO
# CORRIGIDO v2 — 4 bugs críticos resolvidos:
#
# BUG 1 (CORRIGIDO): range(300) = só 4 segundos. Agora usa get_n_tokens().
# BUG 2 (CORRIGIDO): EnCodec decode com shape errado (1 codebook em vez de 8).
# BUG 3 (CORRIGIDO): torch.argmax (greedy = loops) → torch.multinomial (sampling).
# BUG 4 (CORRIGIDO): Arquitetura Transformer ≠ SiMBATherapeutic dos pesos salvos.
# ==============================================================================

# 1. INSTALAÇÃO (só roda uma vez)
# !pip install -q encodec torchaudio einops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import sys
from pathlib import Path
from IPython.display import Audio, display

# ============================================
# 2. SETUP
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎮 Dispositivo: {device}")

# Adiciona o path do código (ajuste conforme seu setup no Kaggle)
sys.path.insert(0, "/kaggle/input/nexus-audio-code")

# ============================================
# 3. IMPORTA O MODELO CORRETO
# ============================================
# CORRIGIDO: usa SiMBATherapeutic — a mesma arquitetura usada no treino!
# O generate.py antigo usava um Transformer diferente, então os pesos
# salvos pelo train-gpu-turbo.ipynb nunca carregavam corretamente.

from src.model import SiMBATherapeutic

# ============================================
# 4. CONFIG (deve ser IDÊNTICA ao treino!)
# ============================================
CONFIG = {
    "model": {
        "d_model": 256,
        "n_layers": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "max_seq_len": 256,
    },
    "audio": {
        "sample_rate": 24000,
        "encodec_bandwidth": 6.0,
    },
    "therapeutic": {
        "use_biofeedback": False,  # Desativado — igual ao treino
    },
}

# ============================================
# 5. CRIA E CARREGA O MODELO
# ============================================
print("🏗️  Criando modelo SiMBATherapeutic...")
model = SiMBATherapeutic.from_config(CONFIG).to(device)
print(f"   Parâmetros: {model.count_parameters():,}")

# Procura checkpoint automaticamente
print("\n🔍 Procurando checkpoint...")
checkpoint_path = ""

search_dirs = [
    "/kaggle/input",
    "/kaggle/working/checkpoints",
    "/kaggle/working",
]

for search_dir in search_dirs:
    if not os.path.exists(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        for fname in ["final.pt", "nexus_trained.pt", "step2000.pt", "step1000.pt"]:
            candidate = os.path.join(root, fname)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                print(f"✅ Encontrado: {checkpoint_path}")
                break
        if checkpoint_path:
            break
    if checkpoint_path:
        break

if not checkpoint_path:
    print("❌ Checkpoint não encontrado.")
    print("   Adicione o dataset do treino em 'Add Data' no Kaggle.")
    raise FileNotFoundError("Checkpoint necessário para geração.")

# Carrega os pesos
checkpoint = torch.load(checkpoint_path, map_location=device)

# Suporta tanto checkpoint completo quanto só state_dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print(f"   Loss do treino: {checkpoint.get('loss', checkpoint.get('best_val_loss', '???'))}")
else:
    state_dict = checkpoint  # Arquivo salvo direto como state_dict

model.load_state_dict(state_dict, strict=True)
model.eval()
print("🧠 Pesos carregados com sucesso!")

# ============================================
# 6. PARÂMETROS DE GERAÇÃO
# ============================================

DURACAO_SEGUNDOS = 60   # Quanto áudio gerar (em segundos)
TEMPERATURA = 0.9        # 0.7 = mais conservador | 1.2 = mais criativo
TOP_K = 50               # Só amostra dos 50 tokens mais prováveis
TOP_P = 0.95             # Nucleus sampling (95% da probabilidade acumulada)

# ============================================
# 7. GERAÇÃO
# ============================================
# CORRIGIDO BUG 1: get_n_tokens() calcula tokens corretos para a duração
# EnCodec 24kHz + 6.0kbps = 75 tokens/segundo
# Para 60s → 4500 tokens. O código antigo usava range(300) = 4 segundos fixos.

n_tokens = model.tokenizer.get_n_tokens(DURACAO_SEGUNDOS)
print(f"\n🎵 Gerando {DURACAO_SEGUNDOS}s de áudio ({n_tokens} tokens)...")
print(f"   Temperatura: {TEMPERATURA} | Top-K: {TOP_K} | Top-P: {TOP_P}")

with torch.no_grad():
    # Tokens iniciais: um token aleatório por codebook
    # CORRIGIDO BUG 2: shape correto (1, n_codebooks, 1) em vez de (1, 1, 1)
    # EnCodec 6.0kbps usa 8 codebooks — o código antigo passava só 1.
    n_codebooks = model.n_codebooks
    current_tokens = torch.randint(
        0, model.vocab_size,
        (1, n_codebooks, 1),
        device=device
    )

    for step in range(n_tokens - 1):
        # Janela de contexto (não ultrapassa max_seq_len)
        context = current_tokens[:, :, -CONFIG["model"]["max_seq_len"]:]

        outputs = model(tokens=context)
        next_logits = outputs["logits"][:, :, -1, :]  # (1, n_cb, vocab_size)

        # Aplica temperatura
        next_logits = next_logits / TEMPERATURA

        # Amostra cada codebook independentemente
        next_tokens = []
        for i in range(n_codebooks):
            cb_logits = next_logits[:, i, :]  # (1, vocab_size)

            # Top-K filtering
            if TOP_K > 0:
                topk_vals = torch.topk(cb_logits, TOP_K)[0][..., -1, None]
                cb_logits = cb_logits.masked_fill(cb_logits < topk_vals, float('-inf'))

            # Top-P (nucleus) filtering
            if TOP_P < 1.0:
                sorted_logits, sorted_idx = torch.sort(cb_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > TOP_P
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                sorted_remove = remove.scatter(-1, sorted_idx, remove)
                cb_logits = cb_logits.masked_fill(sorted_remove, float('-inf'))

            # CORRIGIDO BUG 3: multinomial em vez de argmax
            # argmax (greedy) trava em loops repetitivos.
            # multinomial amostra com probabilidade — áudio mais variado.
            probs = F.softmax(cb_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            next_tokens.append(token)

        # Junta tokens de todos os codebooks: (1, n_cb, 1)
        next_tokens = torch.stack(next_tokens, dim=1)
        current_tokens = torch.cat([current_tokens, next_tokens], dim=2)

        # Log de progresso a cada 10%
        if (step + 1) % (n_tokens // 10) == 0:
            pct = int((step + 1) / n_tokens * 100)
            print(f"   {pct}% concluído ({step + 1}/{n_tokens} tokens)")

print("✅ Tokens gerados!")

# ============================================
# 8. DECODE PARA ÁUDIO
# ============================================
print("\n🔊 Decodificando tokens para áudio...")

# CORRIGIDO BUG 2: tokenizer.decode() recebe (1, n_codebooks, n_frames)
# O código antigo fazia unsqueeze(1) resultando em shape (1, 1, 300)
# — 1 codebook em vez de 8. Resultado: áudio todo distorcido.
# Agora usamos o decode do próprio tokenizer que já sabe o shape correto.

waveform = model.tokenizer.decode(current_tokens)  # (1, channels, samples)

if waveform.dim() == 3:
    waveform = waveform.squeeze(0)  # (channels, samples)

# ============================================
# 9. SALVA E REPRODUZ
# ============================================
save_path = "/kaggle/working/nexus_audio_output.wav"
torchaudio.save(save_path, waveform.cpu(), 24000)

duracao_real = waveform.shape[-1] / 24000
print(f"\n💾 Salvo: {save_path}")
print(f"   Duração real: {duracao_real:.1f}s")
print(f"   Shape: {waveform.shape}")
print("\n👇 Dá o play!")
display(Audio(save_path))
