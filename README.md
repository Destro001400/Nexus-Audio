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
    <a href="#fase-4-beta-4-atual">Beta-4</a> ·
    <a href="#paper">Paper</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-Beta_4_Training-orange?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/GPU-Lightning_AI_(RTXP_6000)-blue?style=flat-square" alt="GPU">
    <img src="https://img.shields.io/badge/codec-EnCodec_24kHz-purple?style=flat-square" alt="Codec">
    <img src="https://img.shields.io/badge/dataset-4.5B_Tokens-green?style=flat-square" alt="Dataset">
    <img src="https://img.shields.io/badge/license-Apache_2.0-lightgrey?style=flat-square" alt="License">
  </p>
</p>

---

## 🧠 O que é o NexMOSHA?

**NexMOSHA** é um sistema de geração de música terapêutica neural desenvolvido pela **SnaX Company**. O projeto explora a interseção entre **inteligência artificial** e **musicoterapia**, inspirado por pesquisas que demonstram como frequências e estruturas sonoras podem impactar a fisiologia humana.

O projeto evoluiu em **4 fases**, cada uma representando uma abordagem arquitetural distinta:

```text
Beta-1         →  Beta-2              →  Beta-3           →  Beta-4 (Atual)
SSM Puro          NexMOSHA (Híbrido)     LLM Fine-Tuning     KDA Híbrido O(L)
SiMBA             MS-SSM + Atenção       Qwen3.5-2B          MS-SSM + Kimi Attention
```

---

## 🏗️ Arquitetura e Evolução

### Fase 1: Beta-1
> *Exploração com SSM Puro — "Onde tudo começou"*
- **Modelo:** SiMBA (4 camadas) + EnCodec 75 Hz
- **Breakthrough:** A adição de atenção causal reduziu a loss massivamente. Descobriu-se o clássico bug de vazamento futuro (falta de máscara causal nativa no PyTorch).

### Fase 2: Beta-2
> *A Primeira Arquitetura Customizada — "A inovação"*
- **Modelo:** MS-SSM (3 escalas: 16, 64, 256) + Atenção Midpoint
- **Feitos:** Introduziu o conceito de predição hierárquica (cb_bridges) e cache KV O(1) com **240× speedup**.

### Fase 3: Beta-3
> *O Transplante Codec-LLM — "O poder dos gigantes"*
- **Modelo:** Qwen3.5-2B com vocabulário substituído (20,480 tokens acústicos).
- **Feitos:** Treinado via LoRA (r=16), mas expôs limitações da biblioteca PEFT para salvar embeddings customizados, gerando a necessidade de scripts isolados de persistência.

### Fase 4: Beta-4 (ATUAL 🚀)
> *O Estado da Arte — Janelas Longas e Memória Linear*

A Beta-4 abandona o peso do Qwen e volta para uma arquitetura "from scratch", mas com proporções gigantescas e eficiência O(L). Focada inteiramente em **música contínua (30s sliding window)**.

| Componente | Detalhe |
|---|---|
| **Arquitetura** | Híbrida 3:1 (3 locais para 1 global) |
| **Local (Short-term)** | MS-SSM (Multi-Scale State Space Models) |
| **Global (Long-term)** | KDA (Kimi Delta Attention) com Delta Rule (O(L)) |
| **Escalonamento** | 4 Presets: *Tiny* (116M) → *Large* (1.3B) |
| **Contexto** | Janela contínua de 30s (2250 tokens) |
| **Dataset** | ~748K chunks de 10s (**4.5 Bilhões de tokens**) extraídos do Jamendo via Dual-GPU. |
| **Hardware** | Lightning AI (Instância RTXP 6000 96GB VRAM) |

---

## 💾 Dataset e Tokenização (A Fazenda de Tokens)

A Beta-4 é alimentada por um processo massivo de mineração no **Jamendo** (focado em tags terapêuticas como `ambient`, `relaxing`, `bossa-nova`, `meditation`).

- **Ferramenta:** Tokenizador customizado rodando paralelamente em contas Kaggle (2x T4).
- **Pipeline:** Download em RAM → Resample 24kHz → Chunks de 10s com overlap de 2s → EnCodec 6kbps → Tensor `[8, 750]`.
- **Tamanho Total:** **~2,078 horas de música** (86 dias ininterruptos), resultando em 4.5 Bilhões de tokens de áudio de altíssima qualidade.

---

## 🚀 Quick Start (Beta-4)

### Pré-requisitos
- Ambiente com PyTorch Lightning, Mamba-SSM (Triton) e EnCodec.
- (Use o `setup_lightning.sh` incluso na pasta).

### Treinamento (Nuvem / Lightning AI)
```bash
cd Beta-4
python train_lightning.py
```
> Edite a variável `MODEL_SIZE` no arquivo para alternar entre as 4 escalas de modelo (`tiny`, `small`, `medium`, `large`).

### Inferência e Geração (DJ Mode)
A inferência busca um "prompt acústico" na API do Jamendo (opcional) e gera a continuação:
```bash
python Beta-4/inference.py
```

---

## 📑 Paper Científico

O paper científico documentando toda essa jornada arquitetural (da Beta-1 à Beta-4) está disponível em duas versões:

- 🇺🇸 [`research/paper_en.tex`](research/paper_en.tex) — Inglês (preparação para arXiv)
- 🇧🇷 [`research/paper_pt.tex`](research/paper_pt.tex) — Português

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

1. **Atenção O(L) é obrigatória para áudio:** Áudio cru/codec gera sequências absurdas (2250 tokens = 30s). Attention padrão ($O(N^2)$) explode VRAM quase instantaneamente. A Delta Rule do KDA salva o modelo mantendo contexto infinito.
2. **Misture Local e Global:** A proporção mágica encontrada foi **3:1**. 3 camadas capturando textura e melodia imediata (MS-SSM) para 1 camada olhando a estrutura da música inteira (KDA).
3. **Pule o Dataset de Fala:** Codecs treinados puramente para fala destróem a estrutura harmônica. O retorno para o EnCodec base da Meta provou ser a melhor escolha musical.

---

## 🎯 Roadmap

- [x] **Beta-1:** Validar SSM + Atenção híbrida
- [x] **Beta-2:** Arquitetura NexMOSHA customizada (77M params)
- [x] **Beta-3:** Setup Qwen3.5-2B + LoRA
- [x] **Beta-4:** Migração para EnCodec, Janela de 30s, KDA 3:1 (Até 1.3B)
- [x] **Dataset:** Coletar 4.5B tokens via tokenizador Dual-GPU no Kaggle.
- [ ] 🔄 **Beta-4 Training:** Treinamento escalonado (Tiny → Large) na Lightning AI.
- [ ] **Avaliação:** Métricas perceptuais (FAD, CLAP)
- [ ] **Avaliação:** Testes de escuta humana e eficácia terapêutica (MOS)
- [ ] **Publicação:** Submissão ao arXiv

---

## 📜 Licença

**Apache License 2.0** — Comercialização permitida, cópia permitida, modificação permitida.

[SnaX Company](https://snax-page.vercel.app) © 2026

<p align="center">
  <sub>Feito com 🧠 + ☕ + 0 GPUs próprias por <strong>SnaX Company</strong></sub>
</p>
