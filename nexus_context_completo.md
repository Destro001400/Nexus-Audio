# NEXUS-AUDIO — Documento Completo de Contexto
## SnaX Company | Guilherme Destro | Abril 2026

---

## PARTE 1 — A EMPRESA E O PRODUTO

### 1.1 SnaX Company

**Fundador/CEO:** Guilherme Destro, 18 anos, Cacoal, Rondônia  
**Contato:** info.snaxcompany@gmail.com  
**Missão:** Democratizar o acesso à musicoterapia personalizada via IA

### 1.2 Nexus-Audio

Plataforma B2B2C de musicoterapia neural. O fluxo é: profissionais de saúde prescrevem música terapêutica personalizada gerada por IA para seus pacientes. A IA não substitui o profissional — ela executa a prescrição.

**Posicionamento estratégico:** Ferramenta auxiliar sob supervisão profissional, complementando tratamento convencional. A linguagem científica usada em todos os materiais é sempre conservadora: "estudos preliminares sugerem", nunca afirmações definitivas de eficácia. Isso é calculado — overclaiming causa problemas regulatórios e de credibilidade.

---

## PARTE 2 — POR QUE ISSO VALE A PENA (A CIÊNCIA)

### 2.1 The Triple Convergence Hypothesis

Três linhas de pesquisa independentes convergiram para tornar o Nexus-Audio viável:

**Linha 1 — Eficiência dos SSMs:**
State Space Models (SSMs) como Mamba e SiMBA reduziram o custo de treino de modelos de áudio em ~12×. O que antes custava $50–150k para treinar agora custa $700–1.500. Isso desbloqueou o mercado para startups.

**Linha 2 — Validação Científica:**
Estudo da ETH Zurich (Nature Metabolism, 2023) demonstrou que música com bass proeminente (50–60 Hz) pode acionar mecanotransdução celular → influxo de Ca²⁺ → exocitose de insulina. Em camundongos diabéticos, reduziu glicemia de >300 mg/dL para ~100 mg/dL. O mecanismo biológico existe.

**Linha 3 — Economia de Treinamento:**
A combinação de DualCodec 12.5Hz (6× menos tokens que EnCodec) + SSMs O(L) torna viável treinar modelos de qualidade em hardware de consumidor (Kaggle T4 gratuito).

### 2.2 The Intention Gap

Pesquisa do CHI 2025 identificou o principal problema não resolvido pela IA musical: usuários sabem como querem que a música soe, mas não conseguem descrever isso para a IA. E a IA não entende o que eles sentem.

Todos os modelos atuais (Suno, Udio, MusicGen, Stable Audio) são sistemas de geração única — sem iteração, sem diálogo. Nenhum foi projetado para a conversa que o usuário quer ter com a IA.

---

## PARTE 3 — TECNOLOGIA: O MODELO

### 3.1 O Codec — DualCodec 12.5Hz

O DualCodec é a fundação de tudo. Ele transforma áudio contínuo em tokens discretos que um modelo de linguagem pode processar.

**Por que DualCodec e não EnCodec ou DAC?**
- EnCodec: 75Hz → 750 tokens por 10s de áudio
- DualCodec: 12.5Hz → 125 tokens por 10s de áudio
- Resultado: 6× menos tokens → sequências 6× menores → viável em GPU de 16GB

**Estrutura do vocabulário:**
- CB0 (semântico, SSL): 16.384 tokens — captura estrutura, melodia, progressões
- CB1–7 (acústico): 4.096 tokens cada — captura timbre, textura, detalhes finos
- Total: 20.480 tokens únicos

**Representação de 60s de áudio:**
- 749 timesteps × 8 codebooks = 5.992 tokens no formato intercalado

### 3.2 Arquitetura SSM — Por Que Não Transformer

Transformers tradicionais têm complexidade O(L²) — dobrar a duração do áudio quadruplica o custo computacional. Para áudio terapêutico longo (5–30 minutos), isso é inviável.

State Space Models têm complexidade O(L) — crescem linearmente. Com memória constante (estado fixo de tamanho fixo), processam sequências teoricamente infinitas.

**O problema do SSM puro:** "Late homogenization" — os estados internos tendem a homogeneizar com o tempo, perdendo distinção entre tokens. O modelo "embruma" após alguns segundos de geração (o fenômeno chamado de "embrumation").

**A solução:** Arquitetura híbrida SSM + Attention. Os SSMs processam a sequência com eficiência linear, e um bloco de atenção no meio integra contexto global periodicamente, prevenindo a homogenização.

---

## PARTE 4 — JORNADA DE DESENVOLVIMENTO

### 4.1 Beta-1 — SiMBATherapeutic v1

**Período:** 2025  
**Arquitetura:** SiMBA (Simplified Mamba-Based Architecture) puro, 4 layers, d_model=512  
**Codec:** EnCodec 75Hz, vocab=1024  
**Dataset:** ~377k chunks (~283M tokens)

**Progresso de loss:**
```
S1: ~4.50 → S2: 3.9690 → S3: 3.7853 → S4: 3.6207 (record SiMBA puro)
S5: 3.6942 → S6: 4.2–4.9 → S7: 4.2239
```

**Bug catastrófico descoberto:**
Das sessões S8 a S12, todo o treinamento foi comprometido por um bug sutil: `nn.MultiheadAttention` do PyTorch não aplica máscara causal por padrão. Sem ela, o modelo "vê tokens do futuro" durante o treino e aprende a copiar em vez de gerar → overfitting instantâneo (loss → 0.018, PPL → 1.0).

Fix: passar máscara triangular explícita via `generate_square_subsequent_mask`. O `is_causal=True` sozinho é insuficiente no módulo nativo do PyTorch.

**Breakthrough da Beta-1:**
Com o bug corrigido e um `AttentionBlock` injetado após a layer 2 (SSMLayerWithAttn):
- S8: loss **1.2885** 🏆
- S9: **0.8709** (PPL 2.7)

**Motivo do encerramento:** Modelo atingiu teto. Decisão de migrar para DualCodec + arquitetura maior.

---

### 4.2 Beta-2 — NexMOSHA

**Nome completo:** Multi-scale cOntextual State space Hybrid Attention  
**Período:** Final 2025 – Início 2026

#### Arquitetura Final
| Componente | Valor |
|---|---|
| d_model | 384 |
| n_layers | 8 blocos SSM |
| n_heads | 6 (no MHA da layer 4) |
| ssm_scales | [16, 64, 256] (d_state de cada escala) |
| attn_layer | 4 (injetado no meio da rede) |
| Parâmetros | ~77M |
| Codec | DualCodec 12.5Hz |
| Sequência | 749 tokens = ~60s de áudio |

**Inovações sobre a Beta-1:**
- DualCodec em vez de EnCodec (6× mais eficiente)
- 8 layers em vez de 4 (mais profundo)
- d_model reduzido de 512 → 384 (para caber na VRAM com 8 layers)
- MHA injetado na layer 4 (meio) em vez de layer 2 (muito cedo)
- Vocab diferente por codebook: CB0=16.384, CB1-7=4.096 (economiza params)
- `cb_loss_weights` rebalanceados: CB0 semântico recebe peso 3.0 vs 0.5 do CB7

**Sessões Beta-2:**
| Sessão | Configuração | Best Val Loss |
|---|---|---|
| S1 | 5 layers, do zero | 4.97 |
| S2 | 8 layers, do zero | 5.20 |
| S3 | 8 layers, resume S2 | 4.75 |
| S4 | 8 layers, resume S3, optimizer reset | 4.79 |
| **S5** | **8 layers, resume S3, warmup=100** | **4.66 🏆** |

**Beta-2 atingiu teto em ~4.66. É o baseline de referência para todo desenvolvimento futuro.**

#### KV Cache — Geração 37s em vez de 1.5h

O `AttentionBlock` foi refatorado: em vez de `nn.MultiheadAttention` monolítico, agora tem projeções q/k/v separadas. Durante treino: comportamento idêntico. Durante inferência: KV cache O(1) por token — a geração de 60s caiu de ~1.5h para **37 segundos**.

---

### 4.3 Beta-3 — Qwen3.5 Fine-tuning

**Período:** Início 2026 (atual)

#### Decisão de Mudar de Estratégia

A Beta-2 com 77M parâmetros treinados do zero levou 5 sessões para atingir teto em 4.66. Para superar isso seria necessário mudar a arquitetura fundamentalmente. A alternativa: fazer fine-tuning de um LLM pré-treinado, que já sabe "prever o próximo token", e ensiná-lo a falar a linguagem dos tokens DualCodec.

**Por que Qwen3.5:**
- Apache 2.0 (pode usar comercialmente)
- 2B ou 4B parâmetros (cabe em T4)
- Janela de contexto 262K tokens
- GDN nativo (atenção linear O(L)) — potencialmente melhor que MHA

**Abordagem:** Trocar as camadas de embedding de texto por embeddings de áudio DualCodec, e fazer LoRA fine-tuning.

#### Hacks do Ambiente Kaggle (Críticos)

O ambiente Kaggle tem várias peculiaridades que causam crashes. Estes hacks são obrigatórios:

1. `config.model_type = "qwen2"` — o registro "qwen3_5" não é reconhecido
2. Mocks de `google.cloud`, `protobuf`, `tensorboard`, `tensorflow`
3. Desempacotamento correto dos tokens: `reshape(1,-1,8).transpose(1,2)`, com clamp e offset por codebook
4. `model.to(torch.float16)` + `model.config.torch_dtype = torch.float16`
5. `CUDA_VISIBLE_DEVICES = "0"` para evitar DataParallel com bitsandbytes
6. `dataloader_pin_memory = False`

#### Bug Crítico S1–S3 (Agora Corrigido)

O bug que manteve a loss em ~14.x por 3 sessões seguidas: os embeddings (`embed_tokens` e `lm_head`) eram re-inicializados com `nn.init.normal_` a cada sessão, antes de carregar o LoRA. Então cada sessão começava do zero em vez de continuar do progresso anterior.

**Fix:** Salvar os embeddings customizados no final de cada sessão como arquivo separado, e carregá-los no início da sessão seguinte ANTES de carregar o LoRA.

| Sessão | Val Loss | Status |
|---|---|---|
| S1 | ~14.x | Bug embeddings |
| S2 | ~14.x | Bug embeddings |
| S3 | ~14.18 | Bug embeddings |
| S4+ | ⏳ | Fix aplicado, aguardando resultado |

---

## PARTE 5 — INFRAESTRUTURA E FERRAMENTAS

### 5.1 Hardware
- Kaggle free tier: 2× Tesla T4 (16GB VRAM cada), sessões de 12h
- Múltiplas contas para tokenização paralela durante downtime de GPU

### 5.2 Datasets e Paths
```
Tokens: /kaggle/input/datasets/destro01400/nexus-audio-dataset-beta-2/tokens
Tokens2: /kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens
DualCodec: /kaggle/input/datasets/destro01400/dualcodec-tokenizer/ckpt/
Qwen3.5-2B: /kaggle/input/models/barnobarno/qwen3.5-2b/transformers/unsloth/1
Qwen3.5-4B: abebe9849/qwen35-4b (dataset Kaggle)
```

### 5.3 Jamendo API
- `client_id: e92f52d5`
- Tags terapêuticas para Auto-Seed no Gerador VIP
- Retry com exponential backoff

### 5.4 Notificações
- Telegram bot: `7781713605:AAFzHiduYdHUMAWuw56MY5VHq4LRbty4hZQ`
- Chat ID: `6908956487`
- Alerta "VIP da SnaX Company" ao final de cada sessão

---

## PARTE 6 — PESQUISA E HORIZONTE TÉCNICO

### 6.1 Gemma 4 (Lançado 2 Abril 2026)

O Google lançou a família Gemma 4 com modelos E2B e E4B que suportam áudio como entrada (ASR). Porém, os modelos ainda **não geram** tokens discretos de áudio — apenas reconhecem fala. Para o Nexus-Audio, ainda seria necessário o mesmo hack de substituição de embeddings.

**Além disso:** Bugs dia-zero foram identificados pela comunidade (`Gemma4ClippableLinear` não suportado pelo PEFT, `mm_token_type_ids` exigido mesmo em fine-tuning texto-only). Decisão: aguardar 2-3 semanas antes de testar.

### 6.2 Unsloth (Ainda Não Testado)

Biblioteca que promete 2× mais velocidade e 70% menos VRAM no fine-tuning de LLMs. Suporte dia-zero para Gemma 4 e Qwen3.5. Deve ser testado na próxima janela de GPU disponível.

### 6.3 GDN — Gated Delta Networks

A arquitetura Qwen3.5 usa GDN nativamente (proporção 3:1 GDN:full-attention). GDN é atenção linear O(L) com memória de estado fixo. Mais eficiente que MHA para sequências longas. Candidato a substituir o MHA da layer 4 do NexMOSHA em versão futura.

### 6.4 Energy-Aware Guidance (EAG)

Técnica proposta para resolver o problema de embrumation em geração longa. Monitora a energia do sinal gerado em tempo real e amortece componentes instáveis adaptativamente. A implementar no gerador quando a prioridade de treinamento estiver resolvida.

---

## PARTE 7 — NEGÓCIO E ESTRATÉGIA

### 7.1 Mercado
- Mercado global de síntese de áudio neural: USD 3-4B (2024) → USD 20-54B projetado (2030-2033)
- CAGR: 26-37%
- Tendência: transição de Transformers O(L²) → SSMs O(L) no setor

### 7.2 Modelo de Negócio
**B2B2C com tripla validação:**
1. Profissional de saúde prescreve (B2B)
2. Plataforma executa a prescrição (software)
3. Paciente recebe e usa (C)

Isso posiciona o produto como ferramenta profissional, não como app de saúde direto ao consumidor — reduz exposição regulatória.

### 7.3 Rodada de Investimento (Planejada)
- **Alvo:** Angel round R$80-100k por 15-20% de equity
- **Pré-requisito:** 30+ respostas de validação de profissionais de saúde
- **Estratégia de validação:** Questionário dual — um focado na condição específica (para investidores), outro amplo multi-condição (para descoberta de mercado)

---

## PARTE 8 — HISTÓRICO COMPLETO DE LOSS

```
Beta-1 (EnCodec, SiMBA puro):
  S1: ~4.50 | S2: 3.9690 | S3: 3.7853 | S4: 3.6207
  S5: 3.6942 | S6: 4.2–4.9 | S7: 4.2239
  S8: 1.2885 🏆 (breakthrough SSM+MHA)
  S9: 0.8709 (PPL 2.7)

Beta-2 (DualCodec, NexMOSHA):
  S1: 4.97 | S2: 5.20 | S3: 4.75 | S4: 4.79 | S5: 4.66 🏆

Beta-3 (Qwen3.5, fine-tuning):
  S1–S3: ~14.x (bug embeddings)
  S4+: ⏳ (fix aplicado)
```

---

## PARTE 9 — DECISÕES E PORQUÊS (PARA REFERÊNCIA FUTURA)

| Decisão | Alternativa Rejeitada | Motivo da Decisão |
|---|---|---|
| DualCodec 12.5Hz | EnCodec 75Hz | 6× menos tokens, viável em T4 |
| SSM + MHA híbrido | SSM puro | SSM puro causa embrumation |
| MHA na layer 4/8 | MHA na layer 2 | SSM processa primeiro, atenção integra no meio |
| Causal mask explícita | `is_causal=True` | PyTorch MHA não aplica mask por padrão |
| Fine-tuning LLM | Treino do zero | 5–10× menos sessões para convergir |
| adamw_torch | adamw_hf / bitsandbytes | Incompatibilidade CUDA 12.8 no Kaggle |
| GPU Kaggle | TPU Kaggle | XLA incompatível com selective scan |
| Gemma 4 (aguardar) | Migrar agora | Bugs dia-zero (abril 2026) |
| Linguagem científica conservadora | Claims de eficácia | Risco regulatório e de credibilidade |
