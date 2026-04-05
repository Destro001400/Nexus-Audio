<p align="center">
  <h1 align="center">🎵 NexMOSHA</h1>
  <p align="center">
    <strong>Multi-scale cOntextual State space Hybrid Attention</strong><br>
    Neural Therapeutic Audio Generation · Zero-Cost Research
  </p>
  <p align="center">
    <a href="#fase-1-beta-1">Beta-1</a> ·
    <a href="#fase-2-beta-2">Beta-2</a> ·
    <a href="#fase-3-beta-3">Beta-3</a> ·
    <a href="#paper">Paper</a> ·
    <a href="#resultados">Resultados</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/custo-$0_(Kaggle_Free)-brightgreen?style=flat-square" alt="Custo $0">
    <img src="https://img.shields.io/badge/GPU-2x_Tesla_T4-blue?style=flat-square" alt="GPU">
    <img src="https://img.shields.io/badge/codec-DualCodec_12.5Hz-purple?style=flat-square" alt="Codec">
    <img src="https://img.shields.io/badge/status-S4_Training-orange?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/license-Apache_2.0-lightgrey?style=flat-square" alt="License">
  </p>
</p>

---

## 🧠 O que é o NexMOSHA?

**NexMOSHA** é um sistema de geração de música terapêutica neural desenvolvido pela **SnaX Company**, projetado para rodar inteiramente em hardware gratuito (Kaggle Free Tier). O projeto explora a interseção entre **inteligência artificial** e **musicoterapia**, inspirado por pesquisas da ETH Zurich que demonstraram que frequências sonoras específicas (50-60 Hz) podem induzir secreção de insulina em células beta pancreáticas.

O projeto evoluiu em **3 fases**, cada uma representando uma abordagem arquitetural distinta:

```
Beta-1 (SSM Puro)  →  Beta-2 (NexMOSHA Híbrido)  →  Beta-3 (LLM Fine-Tuning)
    SiMBA + EnCodec       MS-SSM + Atenção + DualCodec     Qwen3.5-2B + LoRA
```

---

## 🏗️ Arquitetura

### Fase 1: Beta-1
> *Exploração com SSM Puro — "Onde tudo começou"*

- **Modelo:** SiMBA (4 camadas de MambaBlock)
- **Codec:** EnCodec 75 Hz
- **Descoberta chave:** Bug silencioso da máscara causal no PyTorch — `nn.MultiheadAttention` **não** aplica máscara causal por padrão
- **Breakthrough:** Adição de atenção causal reduziu loss de 3.69 → **1.29** (−65% em 1 sessão)
- **Best loss:** 0.87 (PPL 2.7)

### Fase 2: Beta-2
> *Arquitetura Customizada NexMOSHA — "A inovação"*

A arquitetura original que dá nome ao projeto:

```
Input [B, 8, 749]
    ↓
Embedding + Positional
    ↓
Camadas 1-3: MS-SSM (Multi-Scale State Space)
    ↓                    ┌──────────────────┐
Camada 4: MS-SSM +       │ Scale Mixer:     │
  Atenção Causal  ◄──────│ d_state: 16/64/256│
    ↓                    └──────────────────┘
Camadas 5-8: MS-SSM
    ↓
RMSNorm → LM Heads + cb_bridges (hierárquico)
```

| Componente | Detalhe |
|---|---|
| **MS-SSM** | 3 escalas paralelas (d_state: 16, 64, 256) com Scale Mixer dinâmico |
| **Atenção** | Causal na camada 4/8 (midpoint strategy) |
| **cb_bridges** | Predição hierárquica de codebooks com stop-gradient |
| **KV Cache** | Refatoração para inferência O(1) → **240× speedup** |
| **Codec** | DualCodec 12.5 Hz (semântico + acústico, 8 codebooks) |
| **Parâmetros** | ~77M |
| **Best val loss** | 4.66 |

### Fase 3: Beta-3
> *Transplante Codec-LLM — "O poder dos LLMs"*

Substituição do vocabulário de texto do **Qwen3.5-2B** pelo vocabulário de áudio do DualCodec:

| Componente | Detalhe |
|---|---|
| **Base model** | Qwen3.5-2B (2B params) |
| **Adaptação** | LoRA r=16 em W_q, W_k, W_v, W_o |
| **Vocab** | 20,480 tokens de áudio (16K semântico + 4K×7 acústico) |
| **Precisão** | FP16 |
| **Hardware** | 2× Tesla T4 (16 GB cada) |
| **Status** | 🔄 Em treinamento (S4 - fix) |

**Bug crítico resolvido:** A biblioteca PEFT/LoRA **não** salva embeddings customizados. Embeddings devem ser persistidos separadamente (`custom_embeddings.pt`) antes do checkpoint LoRA.

---

## 📊 Resultados

### Progressão de Loss por Fase

| Fase | Sessões | Melhor Loss | PPL | Nota |
|---|---|---|---|---|
| Beta-1 (SSM puro) | S1-S7 | 3.62 | — | Platô / instabilidade |
| Beta-1 (híbrido) | S8-S9 | **0.87** | 2.7 | +Atenção causal + mask fix |
| Beta-2 (NexMOSHA) | S1-S5 | **4.66** | — | DualCodec 12.5Hz, 77M params |
| Beta-3 (Qwen+LoRA) | S1-S3 | ~14.x | — | ❌ Bug embeddings |
| Beta-3 (fix) | S4+ | *pendente* | — | 🔄 Em treinamento |

### Comparação com Outros Modelos

| Modelo | Params | Custo | Codec |
|---|---|---|---|
| MusicGen Small (Meta) | 300M | ~$8,400 | EnCodec 50Hz |
| SiMBA (paper original) | ~42M | ~$700 | CLAP tokens |
| **NexMOSHA Beta-2** | **77M** | **$0** | DualCodec 12.5Hz |
| **NexMOSHA Beta-3** | **~2B** | **$0** | DualCodec 12.5Hz |

---

---

## 🚀 Quick Start

### Pré-requisitos

- Conta no [Kaggle](https://www.kaggle.com) (gratuita)
- Python 3.10+ (para scripts locais)
- `pip install kaggle` (para CLI)

### Treinar o modelo

1. Acesse o notebook de treino no Kaggle
2. Configure o acelerador como **GPU T4 x2**
3. Execute todas as células
4. O modelo salva automaticamente no Kaggle Models

### Gerar áudio

1. Acesse o notebook de geração no Kaggle
2. Configure **GPU T4 x2**
3. O notebook:
   - Carrega o modelo em 2 GPUs em paralelo
   - Gera 4 variações de áudio simultaneamente
   - Decodifica com DualCodec → WAV
   - Envia automaticamente via Telegram

### Monitorar treinos (local)

```bash
# CLI interativo
python scripts/kaggle_cli.py

# Ou direto pela CLI do Kaggle
kaggle kernels status destro01400/train-beta-3
```

---

## 📑 Paper

O paper científico está disponível em duas versões:

- 🇺🇸 [`research/paper_en.tex`](research/paper_en.tex) — Inglês (para arXiv)
- 🇧🇷 [`research/paper_pt.tex`](research/paper_pt.tex) — Português

**Status:** Preprint em preparação. Aguardando resultados da S4 (Beta-3 com fix de embeddings) para completar a seção de resultados.

### Citar

```bibtex
@article{destro2026nexmosha,
  title={NexMOSHA: From Custom Hybrid SSM-Attention to LLM Fine-Tuning 
         for Neural Therapeutic Audio Generation},
  author={Destro, Guilherme},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 🔬 Insights Técnicos

Descobertas que podem ajudar outros pesquisadores:

### 1. Bug da Máscara Causal do PyTorch
O `nn.MultiheadAttention` **não** aplica máscara causal automaticamente. Sempre forneça uma máscara triangular explícita com `generate_square_subsequent_mask()`.

### 2. Embeddings + LoRA = Problema
A biblioteca PEFT/LoRA **não salva** embeddings customizados. Se você substituiu o vocabulário de um LLM, salve `embed_tokens` e `lm_head` separadamente e carregue **antes** do adapter LoRA.

### 3. KV Cache em SSM Híbrido
Refatorar `nn.MultiheadAttention` em projeções Q/K/V explícitas permite cache incremental. Resultado: **240× mais rápido** (1.5h → 37s para 60s de áudio).

### 4. Multi-Escala > Mais Camadas
Diversidade representacional (d_state: 16/64/256) com Scale Mixer dinâmico contribui mais para convergência do que simplesmente empilhar camadas.

---

## 🎯 Roadmap

- [x] Beta-1: Validar SSM + Atenção híbrida
- [x] Beta-2: Arquitetura NexMOSHA customizada (77M params)
- [x] Beta-3: Setup Qwen3.5-2B + LoRA + DualCodec
- [x] Fix: Persistência de embeddings customizados
- [x] Paper: Estrutura LaTeX (EN + PT)
- [ ] 🔄 Beta-3 S4: Treino com fix de embeddings (S4)
- [ ] Avaliação: Métricas perceptuais (CLAP, FAD)
- [ ] Avaliação: Testes de escuta humana (MOS)
- [ ] Publicação: Submissão ao arXiv
- [ ] Beta-4: Dados expandidos + avaliação terapêutica

---

## 📜 Licença

**Apache License 2.0** — Comercialização permitida, cópia permitida, modificação permitida e **proteção de patentes** garantida.

[SnaX Company](https://snax-page.vercel.app) © 2026

---

<p align="center">
  <sub>Feito com 🧠 + ☕ + 0 GPUs próprias por <strong>SnaX Company</strong></sub>
</p>
