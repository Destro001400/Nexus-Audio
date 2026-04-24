<p align="center">
  <h1 align="center">🎵 NexMOSHA</h1>
  <p align="center">
    <strong>Neural EXpert Model for Optimized Sound & Harmonic Architecture</strong><br>
    Neural Therapeutic Audio Generation · Zero-Cost Research
  </p>
  <p align="center">
    <a href="#fase-1-beta-1">Beta-1</a> ·
    <a href="#fase-2-beta-2">Beta-2</a> ·
    <a href="#fase-3-beta-3">Beta-3</a> ·
    <a href="#fase-4-beta-4-atual-">Beta-4</a> ·
    <a href="#-paper-técnico">Paper</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-Beta_4_Training-orange?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/params-116M_→_1.3B-red?style=flat-square" alt="Params">
    <img src="https://img.shields.io/badge/GPU-Lightning_AI_(RTXP_6000)-blue?style=flat-square" alt="GPU">
    <img src="https://img.shields.io/badge/codec-EnCodec_24kHz-purple?style=flat-square" alt="Codec">
    <img src="https://img.shields.io/badge/dataset-4.5B_Tokens-green?style=flat-square" alt="Dataset">
    <img src="https://img.shields.io/badge/license-Apache_2.0-lightgrey?style=flat-square" alt="License">
  </p>
</p>

---

## 🧠 O que é o NexMOSHA?

**NexMOSHA** é um modelo de IA generativa para **áudio terapêutico personalizado**, desenvolvido pela **SnaX Company**. Ele gera música com propriedades terapêuticas usando uma arquitetura híbrida que combina **State Space Models** (eficiência linear) com **Kimi Delta Attention** (memória global), treinado em milhares de horas de música terapêutica.

O projeto nasceu da convergência entre:
- 🧬 **Validação científica** — ETH Zurich (Nature Metabolism, 2023) demonstrou mecanismo biológico direto entre frequências sonoras e respostas fisiológicas
- ⚡ **SSMs eficientes** — complexidade O(L) viabiliza treino de áudio longo em hardware acessível
- 🎵 **Codecs neurais** — EnCodec comprime áudio em tokens discretos treináveis por modelos de linguagem

```text
Beta-1 (2025)      →  Beta-2              →  Beta-3           →  Beta-4 (2026)
SSM Puro              NexMOSHA Híbrido       LLM Fine-Tuning     KDA Híbrido O(L)
SiMBA · 42M           MS-SSM+MHA · 77M      Qwen3.5-2B · 2B     MS-SSM+KDA · 116M→1.3B
EnCodec               DualCodec             DualCodec            EnCodec
```

---

## 🏗️ Arquitetura e Evolução

### Fase 1: Beta-1
> *Exploração com SSM Puro — "Onde tudo começou"*
- **Modelo:** SiMBA (4 camadas, d_model=512) + EnCodec 75Hz
- **Descoberta:** SSM puro atinge teto em loss ~3.62. A adição de atenção causal causou breakthrough instantâneo (loss 1.28).
- **Bug histórico:** `nn.MultiheadAttention` do PyTorch **não** aplica máscara causal por padrão — lição documentada para a comunidade.

### Fase 2: Beta-2
> *A Primeira Arquitetura Customizada — "MS-SSM + Atenção Midpoint"*
- **Modelo:** 8 camadas, MS-SSM com 3 escalas (16, 64, 256) + Atenção causal no meio
- **Inovações:** Predição hierárquica via `cb_bridges` + KV Cache com **240× speedup** (1.5h → 37s)
- **Baseline:** Val Loss **4.66** — teto da arquitetura com 77M parâmetros

### Fase 3: Beta-3
> *O Transplante Codec-LLM — "Vocabulário transplantado"*
- **Modelo:** Qwen3.5-2B com embeddings substituídos para tokens DualCodec
- **Jornada:** 9 sessões. Bugs de embeddings (S1-S3), breakthrough multi-GPU (S6: 6.69), Curriculum Learning no Lightning AI (S9: **Val 4.97 🏆**)
- **Encerramento:** DualCodec usava backbone SSL treinado para **fala**, não música. Fundação incompatível com geração musical.

### Fase 4: Beta-4 (ATUAL 🚀)
> *O Estado da Arte — Eficiência O(L) com Memória Infinita*

Retorno à arquitetura custom com duas inovações fundamentais: **Kimi Delta Attention** (atenção linear O(L) com Delta Rule) e **proporção 3:1** (3 camadas locais MS-SSM para cada 1 camada global KDA).

| Componente | Detalhe |
|---|---|
| **Arquitetura** | Híbrida 3:1 (MS-SSM local + KDA global) |
| **Atenção** | KDA — Delta Rule, O(L), 8 cabeças |
| **SSM** | Mamba-2 multi-escala (32, 64, 128) |
| **Escalonamento** | 4 Presets: *Tiny* (116M) → *Large* (1.3B) |
| **Contexto** | Janela deslizante de 30s (2250 tokens) com 50% overlap |
| **Codec** | EnCodec 24kHz @ 6kbps (vocab=1024, 8 codebooks RVQ) |
| **Dataset** | ~748K chunks · **4.5 Bilhões de tokens** · ~2.078 horas |
| **Hardware** | Lightning AI (RTXP 6000, 96GB VRAM) |

```
tokens [B, 8, 2250]
    ↓ 8× Embedding(1024, 512) → Linear(4096, 512)
    ↓ Layers 0-2: MS-SSM (local)       ┐
    ↓ Layer 3:    KDA (global)          │ 3:1
    ↓ Layers 4-6: MS-SSM (local)       │
    ↓ Layer 7:    KDA (global)          ┘
    ↓ RMSNorm → 8× Linear(512, 1024)
logits: {cb0..cb7: [B, T, 1024]}
```

---

## 📊 Evolução da Loss

```
Beta-1 (SiMBA puro):     ~4.50 → 3.62 → 1.28* → 0.87*
Beta-2 (NexMOSHA):       4.97 → 4.66 🏆 (teto)
Beta-3 (Qwen3.5 LoRA):   14.x → 6.69 → 6.27 → 4.97 🏆
Beta-4 (NexMOSHA v2.5):  ⏳ treino iminente

* Beta-1 S8-S9: values after causal mask fix + attention injection
```

---

## 💾 Dataset (A Fazenda de Tokens)

**~2.078 horas** de música terapêutica minerada do **Jamendo** (32 tags: `ambient`, `relaxing`, `meditation`, `bossa-nova`, etc.)

- **Pipeline:** Download → Resample 24kHz → Chunks de 10s → EnCodec 6kbps → Tensor `[8, 750]`
- **Tokenização:** Dual-GPU paralela em Kaggle (2× T4, ~2× throughput)
- **Total:** ~748K chunks · **4.5 Bilhões de tokens**

---

## 🚀 Quick Start

### Setup (Lightning AI)
```bash
cd Beta-4
bash setup_lightning.sh
```

### Treinamento
```bash
python train_lightning.py
```
> Edite `MODEL_SIZE` para alternar: `tiny` (116M) · `small` (371M) · `medium` (500M) · `large` (1.3B)

### Inferência (DJ Mode)
```bash
python inference.py
```

---

## 📑 Paper Técnico

Documento técnico completo documentando toda a jornada arquitetural (Beta-1 → Beta-4), incluindo bugs descobertos, breakthroughs, decisões de design e métricas:

- 📄 [`free/nexus_paper_tecnico.md`](free/nexus_paper_tecnico.md) — Documento técnico completo

### Citar

```bibtex
@article{destro2026nexmosha,
  title={NexMOSHA: Multi-scale Contextual State Space Hybrid Attention 
         for Neural Therapeutic Audio Generation},
  author={Destro, Guilherme},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 🔬 Insights Técnicos & Lições Aprendidas

Descobertas que podem ajudar outros pesquisadores:

1. **🎯 Proporção 3:1 é o sweet spot** — 3 camadas locais (MS-SSM) para cada 1 global (KDA). Validado pelo Kimi Linear paper e confirmado empiricamente.
2. **⚡ Atenção O(L) é obrigatória para áudio** — 2250 tokens = 30s. Atenção quadrática explode VRAM. A Delta Rule do KDA mantém contexto infinito em O(L).
3. **🎵 Pule codecs de fala para música** — DualCodec (W2V-BERT 2.0) captura fonemas, não harmonia. EnCodec base é a melhor escolha para áudio musical.
4. **🐛 Causal mask do PyTorch é uma armadilha** — `nn.MultiheadAttention` não aplica mask causal por padrão, mesmo com `is_causal=True`. Sempre passe a máscara explicitamente.
5. **💾 Embeddings ≠ LoRA** — PEFT não salva embeddings customizados. Salve separadamente ou perca 3 sessões de treino.

---

## 🎯 Roadmap

- [x] **Beta-1:** Validar SSM + Atenção híbrida
- [x] **Beta-2:** Arquitetura NexMOSHA customizada (77M params)
- [x] **Beta-3:** Qwen3.5-2B + LoRA (9 sessões, Val 4.97)
- [x] **Beta-4:** Migração para EnCodec, KDA 3:1, janela 30s (Até 1.3B)
- [x] **Dataset:** 4.5B tokens via tokenizador Dual-GPU
- [ ] 🔄 **Treino Beta-4:** Escalonamento Tiny → Large na Lightning AI
- [ ] **Avaliação:** Métricas perceptuais (FAD, CLAP)
- [ ] **Avaliação:** Testes de escuta humana (MOS)
- [ ] **Publicação:** Submissão ao arXiv

---

## 📜 Licença

**Apache License 2.0** — Comercialização permitida, cópia permitida, modificação permitida.

[SnaX Company](https://snax-page.vercel.app) © 2026

<p align="center">
  <sub>Feito com 🧠 + ☕ + 0 GPUs próprias por <strong>SnaX Company</strong></sub>
</p>
