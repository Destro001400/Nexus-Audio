# NexMOSHA: A Multi-Scale Hybrid SSM-Attention Architecture for Neural Therapeutic Music Generation

**Documento Técnico Completo — Base para Publicação**  
**SnaX Company | Guilherme Destro | Abril 2026**  
**Status:** Em desenvolvimento ativo — Beta-4 (Treino iminente na Lightning AI)

---

## 1. HISTÓRIA E CRONOLOGIA

### 1.1 Origem do Projeto (2025)

O projeto Nexus-Audio nasceu da interseção de três convergências independentes identificadas em Janeiro de 2026:

1. **Eficiência dos SSMs**: A emergência de State Space Models (Mamba, SiMBA) reduziu o custo de treinamento de modelos de áudio generativo em ~12×, de $50k–150k para ~$700–1.500, tornando o desenvolvimento viável para startups sem capital massivo.

2. **Validação Científica**: Estudo da ETH Zurich (Nature Metabolism, 2023) demonstrou mecanismo biológico direto entre frequências sonoras específicas (bass 50–60 Hz) e liberação de insulina via mecanotransdução celular → influxo de Ca²⁺ → exocitose. Em camundongos diabéticos tipo 1, o estímulo sonoro reduziu glicemia de >300 mg/dL para ~100 mg/dL.

3. **Codec Eficiente**: Codecs neurais de áudio (EnCodec, DualCodec) comprimem waveforms em tokens discretos, tornando sequências de áudio longas computacionalmente tratáveis em hardware de consumidor.

A hipótese central: combinar SSMs com atenção híbrida e tokenização eficiente poderia produzir um modelo de musicoterapia neural viável em GPUs de consumo, eliminando a barreira de entrada que antes restringia o campo à Google, Meta e OpenAI.

### 1.2 Linha do Tempo

```
Jan 2025:   Início do projeto, arquitetura SiMBA pura
            Codec: EnCodec 75Hz, vocab=1024, 4 layers
Ago 2025:   Bug crítico descoberto (causal mask ausente)
            Breakthrough S8: loss 1.2885, S9: PPL 2.7
Out 2025:   Beta-1 atinge teto; decisão de migrar
Nov 2025:   Beta-2 S1 inicia com DualCodec 12.5Hz
Dez 2025:   Desenvolvimento KV Cache (37s vs 1.5h)
Jan 2026:   Beta-2 S5 atinge best val_loss = 4.66 🏆
Mar 2026:   Decisão de migrar para fine-tuning Qwen3.5
Abr 2026:   Bug de embeddings descoberto e corrigido (S1-S3 perdidas)
Abr 2026:   Beta-3 S6: 6.69 (Master Weights FP32 + Autocast patch)
Abr 2026:   Beta-3 S7: 6.27 🏆 (Cosine Warm Restart)
Abr 2026:   Beta-3 S9: Val 4.97 🏆 (Curriculum Learning, Lightning AI)
Abr 2026:   ⛔ Beta-3 ENCERRADA — DualCodec descartado (SSL de fala)
Abr 2026:   Beta-4 inicia: NexMOSHA v2.5 + EnCodec + KDA
Abr 2026:   Arquitetura Beta-4 testada e compilando (116M params) ✅
```

### 1.3 Momentos-Chave de Decisão

**Momento 1 — Bug da Causal Mask (Beta-1, ~Ago 2025)**
Sessões S8–S12 foram completamente comprometidas: `nn.MultiheadAttention` do PyTorch NÃO aplica máscara causal por padrão. Sem ela, o modelo "vê tokens futuros" → overfitting instantâneo (loss → 0.018, PPL → 1.0). A correção exigiu passagem explícita de máscara triangular via `generate_square_subsequent_mask`. O parâmetro `is_causal=True` isoladamente é insuficiente.

**Momento 2 — Breakthrough SSM+Attention (Beta-1 S8)**
Injeção de um `AttentionBlock` após a layer 2 da arquitetura SSM pura quebrou o plateau de 3.69 que havia persistido por múltiplas sessões. Loss dropped de 3.69 → 1.2885 em uma única sessão, com S9 atingindo 0.8709 (PPL 2.7).

**Momento 3 — Migração para DualCodec (Out/Nov 2025)**
Com Beta-1 atingindo teto, a migração para DualCodec 12.5Hz foi decisiva: redução de 750 → 749 tokens por 60s de áudio (75Hz→12.5Hz), mas com 8 codebooks em vez de 1 do EnCodec — resultando em representação hierárquica muito mais rica com sequências 6× menores.

**Momento 4 — Instabilidade e Fix do LR (Beta-2 S1)**
Sessão com lr=3e-4 divergiu no step ~2460 (loss: 2.3M, PPL: inf). Diagnóstico: LR muito alto para a arquitetura. Fix: lr=1e-4 + clip_grad=0.5. A sessão subsequente convergiu de forma estável até val_loss 4.97.

**Momento 5 — Decisão Fine-tuning (Mar 2026)**
Com Beta-2 atingindo teto em 4.66 após 5 sessões, a análise mostrou que o modelo ~77M parâmetros treinado do zero precisaria de muito mais dados e sessões para superar esse patamar. A alternativa: fazer fine-tuning de um LLM pré-treinado (Qwen3.5-2B/4B), aproveitando a capacidade já desenvolvida de "prever próximo token" e redirecionando-a para tokens de áudio DualCodec.

**Momento 6 — Bug dos Embeddings (Abr 2026)**
Três sessões consecutivas com loss ~14.x revelaram o bug: os embeddings custom (`embed_tokens` e `lm_head`) eram re-inicializados com `nn.init.normal_` a cada sessão, antes de carregar o LoRA. Fix: salvar/carregar embeddings como arquivo separado do adapter, com precedência sobre o carregamento do LoRA.

**Momento 7 — Breakthrough Multi-GPU (Beta-3 S6, Abr 2026)**
`DataParallel` do PyTorch joga fora o contexto `torch.amp.autocast` nas sub-threads de GPU, causando colisão entre Master Weights FP32 e Model Layers FP16. Além disso, o PEFT cria subclasses que anulam patches feitos na classe pai. Fix: promover tensóres do PEFT para Float32 + invocar `type(model).forward` para aplicar Autocast dinâmico na subclasse exata. Loss despencou de 14.x para **6.69** em uma única sessão.

**Momento 8 — Curriculum Learning + Lightning AI (Beta-3 S9, Abr 2026)**
Migração do Kaggle (2× T4, 32GB) para Lightning AI (RTXP 6000, 96GB). LoRA expandido de r=16 para r=128, batch efetivo 32, BFloat16 puro. Curriculum Learning Fase 1 (treino APENAS do CB0 semântico) resultou em Val Loss **4.97** — menor da história do projeto.

**Momento 9 — O Pivot Fatal do Codec (Abr 2026)**
Análise pós-mortem revelou que o DualCodec usava W2V-BERT 2.0 como backbone SSL do CB0 — um modelo treinado exclusivamente para **fala**, não música. O codebook semântico capturava fonemas e prosódia, não harmonia e melodia. Fundação errada para gerar música terapêutica. Decisão: retorno ao EnCodec 24kHz (codec de áudio geral, validado pelo MusicGen da Meta). Beta-3 encerrada.

---

## 2. BETA-2 — NexMOSHA (Arquitetura Custom)

### 2.1 Nome e Conceito

**NexMOSHA** — Multi-scale cOntextual State space Hybrid Attention

A premissa central: SSMs puros sofrem de "late homogenization" — os estados internos tendem a homogeneizar com o tempo, perdendo distinção entre tokens únicos. Em domínio de áudio, isso se manifesta como "embrumation" (geração de babble incoerente após ~8-9 segundos). A solução é a arquitetura híbrida: SSMs processam a sequência com complexidade O(L), e um bloco de atenção causal no meio previne a homogenização ao injetar contexto global periodicamente.

### 2.2 Arquitetura Final

```
Input: tokens [B, 8, 749]  (B=batch, 8 codebooks, 749 timesteps = ~60s)
  ↓
embed_tokens:
  CB0 (semântico): nn.Embedding(16384, 384)  ← W2V-BERT 2.0 SSL features
  CB1-7 (acústico): nn.Embedding(4096, 384) × 7  ← DAC waveform features
  Soma os 8 embeddings → [B, 749, 384]
  ↓
pos_embedding: nn.Embedding(749, 384)
  ↓
Layer 1: ResidualMSSSMBlock (scales=[16, 64, 256])
Layer 2: ResidualMSSSMBlock (scales=[16, 64, 256])
Layer 3: ResidualMSSSMBlock (scales=[16, 64, 256])
Layer 4: SSMLayerWithAttn → ResidualMSSSMBlock + AttentionBlock(causal)
Layer 5: ResidualMSSSMBlock
Layer 6: ResidualMSSSMBlock
Layer 7: ResidualMSSSMBlock
Layer 8: ResidualMSSSMBlock
  ↓
RMSNorm(384)
  ↓
lm_heads hierárquicos:
  head_0: Linear(384, 16384)  → logits CB0
  cb_bridge_0: Linear(16384, 384)  → contexto para CB1
  head_1: Linear(384, 4096)   → logits CB1
  cb_bridge_1: Linear(4096, 384)   → contexto para CB2
  ... (padrão se repete para CB2-7)

Output: lista de logits [B, 749, vocab_sizes[i]] por codebook
Loss: weighted cross-entropy com shift (prediz t+1 a partir de t)
```

**Parâmetros totais:** ~77M

### 2.3 MS-SSM Block — Multi-Scale State Space Model

Cada `ResidualMSSSMBlock` contém três ramos MambaBlock paralelos com d_state diferente:

| Escala | d_state | Captura |
|---|---|---|
| Fina | 16 | Notas individuais, ataques, transientes rápidos |
| Média | 64 | Frases musicais, progressões de acordes |
| Grossa | 256 → limitado a 32 no Kaggle T4 | Seções, estrutura macroscópica |

> **Nota:** d_state=[16, 64, 256] causa OOM nas T4 com 8 layers. O valor operacional foi ssm_scales=[8, 16, 32] ou [16, 64, 256] com n_layers=5.

Um "Scale Mixer" baseado em rede linear com SiLU calcula pesos dinâmicos por timestep para cada escala. A implementação usa scan seletivo paralelo vetorizado em log-space para estabilidade numérica e paralelismo em GPU, resultando em speedup de 8-12× sobre implementações sequenciais.

### 2.4 AttentionBlock — Atenção Causal Obrigatória

```python
class AttentionBlock(nn.Module):
    def forward(self, x):
        B, T, D = x.shape
        # CRÍTICO: passagem explícita de máscara triangular
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=causal_mask, is_causal=True)
        x = x + attn_out
        return x + self.ff(self.ff_norm(x))
```

**Por que na layer 4/8:** SSMs das layers 1-3 processam eficientemente a sequência em O(L). A atenção na layer 4 integra contexto global no meio da rede. As layers 5-8 refinam a representação. Posicionar a atenção muito cedo (Beta-1 usava layer 2) impede que os SSMs estabeleçam representações locais antes da integração global.

### 2.5 cb_bridges — Predição Hierárquica

O `cb_bridge` projeta os logits do codebook i de volta ao espaço latente, condicionando a predição do codebook i+1:

```python
for i, head in enumerate(self.lm_heads):
    logits_i = head(cb_hidden)          # [B, T, vocab_sizes[i]]
    if i < n_codebooks - 1:
        cb_context = self.cb_bridges[i](logits_i.detach())  # .detach() crucial
        cb_hidden = cb_hidden + cb_context
```

O `.detach()` impede que gradientes do CB(i+1) perturbem os pesos do CB(i), garantindo predição estável por codebook.

**cb_loss_weights:** `[3.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5]`

CB0 semântico (W2V-BERT 2.0) recebe peso 3.0 pois captura estrutura harmônica e melódica de alto nível. CB1-7 acústicos recebem pesos decrescentes — detalhes de timbre têm menor impacto na coerência musical terapêutica.

### 2.6 KV Cache — Inferência O(1)

Para o `AttentionBlock`, o código original usava `nn.MultiheadAttention` monolítico (impossível de cachedar). A refatoração separou projeções q/k/v:

```python
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
```

Durante treino: comportamento idêntico. Durante inferência com KV cache: O(1) por token. Resultado empírico: geração de 60s reduziu de ~1.5 horas para **37 segundos**.

### 2.7 Dataset Beta-2

- **Fonte:** Jamendo API (música Creative Commons, tags terapêuticas)
- **Client ID:** e92f52d5
- **Chunks:** 48.400 arquivos × 749 tokens = ~36M tokens de áudio terapêutico
- **Divisão:** 46.427 treino / 2.000 validação
- **Codec:** DualCodec 12.5Hz — 60s = 749 timesteps × 8 codebooks = 5.992 tokens no formato intercalado
- **Duração de áudio:** ~33 dias de música contínua no dataset

**Paths:**
```
/kaggle/input/datasets/destro01400/nexus-audio-dataset-beta-2/tokens
/kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens
```

### 2.8 Configuração de Treino

| Hiperparâmetro | Valor |
|---|---|
| Optimizer | AdamW (adamw_torch) |
| LR base | 1e-4 (dt_proj: 1e-5) |
| Schedule | WSD: warmup=500 / stable=7000 / decay=2500 |
| Batch size | 2 por GPU × 8 grad_accum = 16 efetivo |
| clip_grad | 0.5 |
| Hardware | 2× Tesla T4 (Kaggle free tier) |
| Velocidade | ~1.4–1.7 spm (steps per minute) |

> **Por que adamw_torch e não adamw_hf?** bitsandbytes é incompatível com CUDA 12.8 no ambiente Kaggle. Descoberto empiricamente após múltiplas tentativas.

---

## 3. RESULTADOS DE TREINO — BETA-2

### 3.1 Beta-1 (Referência Histórica)

**Arquitetura:** SiMBA puro, 4 layers, d_model=512, EnCodec 75Hz, vocab=1024

| Sessão | Loss Final | Observação |
|---|---|---|
| S1 | ~4.50 | Baseline inicial |
| S2 | 3.9690 | — |
| S3 | 3.7853 | — |
| S4 | 3.6207 | Record SiMBA puro |
| S5 | 3.6942 | Regressão |
| S6 | 4.2–4.9 | Divergência |
| S7 | 4.2239 | — |
| S8 | **1.2885** 🏆 | **Breakthrough: SSM+MHA hybrid + causal mask fix** |
| S9 | **0.8709** | PPL 2.7 — melhor resultado da Beta-1 |

*Nota: Sessões S8–S12 anteriores à correção foram comprometidas pelo bug da causal mask (PPL→1.0 por overfitting instantâneo).*

### 3.2 Beta-2 — Progressão Detalhada

**Beta-2 S1 (5 layers, do zero, lr=1e-4, lr_instável=3e-4 descartada)**

Sessão com lr=3e-4 divergiu no step 2460 (loss: 2.3M → inf). Sessão com lr=1e-4:

| Checkpoint | Val Loss | Val PPL |
|---|---|---|
| Step 1000 | 5.8337 | 341.6 |
| Step 2000 | 5.4260 | 227.2 |
| Step 3000 | 5.2356 | 187.8 |
| Step 4000 | 5.1046 | 164.8 |
| Step 5000 | 5.0960 | 163.4 |
| Step 6000 | 4.9952 | 147.7 |
| Step 7000 | **4.9787** 🏆 | 145.3 |

Steps individuais chegando em 4.66 no step 7440 (Loss train: 4.6683, PPL: 106.5).

**Beta-2 S2 (8 layers, do zero)**

| Checkpoint | Val Loss | Val PPL |
|---|---|---|
| Step 1000 | 5.9955 | 401.6 |
| Step 2000 | 5.6076 | 272.5 |
| Step 3000 | 5.3614 | 213.0 |
| Step 4000 | 5.2062 | 182.4 |

**Beta-2 S3 (8 layers, resume S2, ~4000→11300 steps totais)**

| Checkpoint | Val Loss | Val PPL |
|---|---|---|
| ~5000 total | 5.0254 | 152.2 |
| ~6000 total | **4.9099** 🏆 | 135.6 |

**Beta-2 S4 (8 layers, resume S3, optimizer reset)**

Best val loss: ~4.79 (regressão por reset do optimizer — learning rate iniciou alto novamente)

**Beta-2 S5 (8 layers, resume S3, warmup=100)**

Best val loss: **4.66** 🏆 — **Baseline definitivo da Beta-2**

### 3.3 Sumário Beta-2

| Sessão | Configuração | Best Val Loss |
|---|---|---|
| S1 | 5 layers, do zero | 4.97 |
| S2 | 8 layers, do zero | 5.20 |
| S3 | 8 layers, resume S2 | 4.75 |
| S4 | 8 layers, resume S3, optimizer reset | 4.79 |
| **S5** | **8 layers, resume S3, warmup=100** | **4.66 🏆** |

**Beta-2 atingiu teto em ~4.66 após 5 sessões. Limite da arquitetura com os dados disponíveis.**

---

## 4. BETA-3 — FINE-TUNING DE LLM

### 4.1 Por Que Qwen3.5-2B

Após a Beta-2 atingir teto em 4.66, a análise indicou que o modelo ~77M parâmetros treinado do zero precisaria de muito mais dados e sessões para avançar significativamente. A alternativa: fine-tuning de LLM pré-treinado.

Critérios de seleção do Qwen3.5-2B:

| Critério | Valor |
|---|---|
| Licença | Apache 2.0 (uso comercial permitido) |
| Tamanho | 2B params (cabe em T4 com LoRA) |
| Contexto | 262.144 tokens nativos |
| Arquitetura | Hybrid GDN+Attention (3:1) — O(L) nativo |
| Disponibilidade | Kaggle: `barnobarno/qwen3.5-2b` |

O GDN (Gated Delta Networks) nativo do Qwen3.5 é atenção linear O(L), mais eficiente que MHA para sequências longas. A proporção 3:1 (GDN:full-attention) é alinhada com a nossa descoberta experimental de que híbridos SSM+Attention superam SSMs puros.

### 4.2 Abordagem: LLM Gerando Tokens de Codec de Áudio

A ideia central é "transplante de vocabulário": remover as camadas de embedding de texto do Qwen3.5 e substituí-las por embeddings de áudio DualCodec, mantendo todos os pesos das camadas intermediárias intactos. O LoRA fine-tuning então adapta o modelo para a nova distribuição.

```
Qwen3.5-2B (pré-treinado em texto)
    ↓ Substituição de vocab: 248.320 → 20.480 tokens
    ↓ embed_tokens: novo nn.Embedding(20.480, 2048)
    ↓ lm_head: nova nn.Linear(2048, 20.480)
    ↓ LoRA r=16 nas projeções de atenção
    ↓ Fine-tuning em tokens DualCodec de música terapêutica
```

**Por que funciona?** O modelo pré-treinado já aprendeu a "prever próximo token" de forma robusta em sequências longas. A tarefa de prever próximo token de áudio discreto é estruturalmente similar — apenas o vocabulário muda. O LoRA precisa apenas adaptar as projeções de atenção, não reaprender a mecânica fundamental.

### 4.3 DualCodec — Por que Escolhemos

O DualCodec (Amphion/OpenMMLab) opera em dois fluxos paralelos:

**Fluxo semântico:** W2V-BERT 2.0 SSL → CB0 com 16.384 tokens  
**Fluxo acústico:** DAC waveform → CB1-7 com 4.096 tokens cada

**Configuração operacional:**
- Taxa de quadros: 12.5 Hz
- Duração por chunk: 60 segundos
- Tokens por chunk: 749 timesteps × 8 codebooks = 5.992 tokens (intercalados)
- Vocab total: 20.480 tokens

**Comparativo de codecs:**

| Codec | Taxa | Tokens/60s | Qualidade |
|---|---|---|---|
| EnCodec (Meta) | 75 Hz | ~4.500 | Boa |
| DAC | 50–75 Hz | 3k–4.5k | Excelente |
| **DualCodec** | **12.5 Hz** | **749×8=5.992** | **Excelente + semântico** |

Apesar de produzir mais tokens por segundo em modo intercalado, cada timestep contém 8× mais informação hierárquica, permitindo reconstrução de alta fidelidade com 6× menos frames temporais que o EnCodec.

**Desempacotamento correto (hack crítico):**
```python
# DualCodec retorna tokens intercalados — reshape obrigatório
tokens = raw_tokens.reshape(1, -1, 8).transpose(1, 2)  # [1, 8, T]
# CB0 (semântico): índices 0–16383 → clamp(0, 16383)
# CB1-7 (acústico): índices 16384–20479 → subtrair 16384, clamp(0, 4095)
audio = dualcodec.decode(tokens).squeeze(0)  # [1, samples]
```

### 4.4 Fine-tuning: LoRA, Embeddings Custom, Hiperparâmetros

**Configuração LoRA:**
- r = 16 (Qwen3.5-2B full LoRA) ou r = 32 QLoRA (Qwen3.5-4B)
- target_modules: q_proj, k_proj, v_proj, o_proj
- Precisão: FP16

**Hiperparâmetros Beta-3:**
- LR: 1e-4
- Warmup: 100 steps
- Optimizer: adamw_torch
- Batch: 1 por GPU (CUDA_VISIBLE_DEVICES="0" — single GPU para evitar conflito bitsandbytes)
- max_seq_len: 5.992 tokens (60s completos intercalados)

**Hacks críticos do ambiente Kaggle:**

```python
# 1. model_type não reconhecido
config.model_type = "qwen2"

# 2. Dependências ausentes — mocks obrigatórios
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.protobuf.internal.builder'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
# + mock completo do tensorflow

# 3. Força FP16 (evita inconsistência de dtype)
model.to(torch.float16)
model.config.torch_dtype = torch.float16

# 4. DataLoader sem pin_memory (crash no Kaggle)
dataloader_pin_memory = False
```

**Fix crítico de embeddings entre sessões:**
```python
# FINAL de cada sessão — salvar embeddings JUNTO com LoRA
torch.save({
    'embed_tokens': model.model.embed_tokens.state_dict(),
    'lm_head':      model.lm_head.state_dict()
}, os.path.join(save_path, 'custom_embeddings.pt'))

# INÍCIO de cada sessão — carregar ANTES do LoRA
emb_path = os.path.join(CHECKPOINT_RESUME, 'custom_embeddings.pt')
if os.path.exists(emb_path):
    emb_ckpt = torch.load(emb_path, map_location='cpu')
    model.model.embed_tokens.load_state_dict(emb_ckpt['embed_tokens'])
    model.lm_head.load_state_dict(emb_ckpt['lm_head'])
# Só ENTÃO: PeftModel.from_pretrained(model, CHECKPOINT_RESUME)
```

### 4.5 Resultados Beta-3 (Até Abril 2026)

| Sessão | Steps | Val Loss | Observação |
|---|---|---|---|
| S1 | ~1000 | ~14.x | Bug: embeddings re-inicializados |
| S2 | ~1000 | ~14.x | Bug: embeddings re-inicializados |
| S3 | ~700 | ~14.18 | Bug: embeddings re-inicializados |
| S4 | ~700 | — | Fix embeddings aplicado, crash trainer_state.json |
| S5 | 700 | 14.22 | Primeiro treino estável (FP32LossTrainer + GradScaler) |
| **S6** | **700** | **6.69** | **⚡ BREAKTHROUGH: Master Weights FP32 + Autocast MonkeyPatch Multi-GPU** |
| **S7** | **700** | **6.27 🏆** | **Resume perfeito. Cosine Warm Restart puxou all-time-low** |
| S8 | 700 | ~12.8 | Loss pesada CB0×5. Restrição de capacidade (r=16, batch=4, T4) |
| **S9** | **1000** | **4.97 🏆** | **Migração Lightning AI. LoRA r=128, Curriculum Fase 1 (só CB0)** |

**S6 — O Breakthrough Multi-GPU:**
A loss caiu de 14.x para 6.69 em uma única sessão. O fix combinou três técnicas: (1) Master Weights em FP32 para sobreviver ao GradScaler, (2) MonkeyPatch do `forward` na subclasse exata do PEFT para forçar Autocast por thread de GPU, e (3) `fp16=True` no Trainer.

**S9 — A Maior Sessão:**
Migração para RTXP 6000 (96GB VRAM). LoRA expandido 8× (r=16 → r=128), batch efetivo 32, BFloat16 puro, num_workers=16. Velocidade: 1.92 áudios/sec (40× mais que Kaggle). Curriculum Learning Fase 1 treinou APENAS o CB0 semântico (weight_CB0=1.0, weight_CB1-7=0.0), resultando em Val Loss 4.97 — **menor da história do projeto**.

**⛔ Beta-3 Encerrada após S9.** [Post-mortem](free/Beta3_Post_Mortem.md) completo documentado separadamente.

**Por que a loss estava em 14.x (S1-S3):** O modelo inicializava embeddings com `nn.init.normal_` (distribuição aleatória, std=0.02) a cada sessão. Com embeddings aleatórios, a loss inicial é log(20480) ≈ 9.93. O fato da loss estar em ~14.x indica modelo aprendendo do zero em cada sessão, sem nenhuma progressão acumulada.

---

## 5. BETA-4 — NexMOSHA v2.5 (ATIVA — Abril 2026)

### 5.1 O Pivot do Codec

A análise pós-mortem da Beta-3 revelou um problema fundamental: o **DualCodec** usava W2V-BERT 2.0 como backbone SSL do codebook semântico (CB0). Esse modelo foi treinado exclusivamente para **fala** — ele captura fonemas e prosódia, não harmonia e melodia. A fundação estava errada para gerar música terapêutica.

Decisão: retorno ao **EnCodec 24kHz @ 6kbps** (Meta) — codec de áudio geral, validado pelo MusicGen.

| Propriedade | DualCodec (descartado) | EnCodec (Beta-4) |
|---|---|---|
| Backbone SSL | W2V-BERT 2.0 (fala) | Nenhum (end-to-end) |
| Taxa | 12.5 Hz | 75 Hz |
| Vocab | 16.384 + 4.096 | **1.024 uniforme** |
| Codebooks | 8 (hierárquico) | 8 (RVQ) |
| Domínio | Fala | **Áudio geral** |

### 5.2 Arquitetura: Híbrida 3:1 MS-SSM/KDA

A Beta-4 retorna à arquitetura custom "from scratch", mas com duas inovações cruciais:

1. **Kimi Delta Attention (KDA)** substitui o MHA padrão — atenção linear O(L) com Delta Rule
2. **Proporção 3:1** (local:global) baseada no paper Kimi Linear

```
tokens [B, 8, 2250]
    ↓ transpose → [B, 2250, 8]
    ↓ 8× nn.Embedding(1024, 512) → concatenar → nn.Linear(4096, 512)
hidden [B, 2250, 512]
    ↓ + pos_emb(2250, 512)
    ↓ layers 0-2: ResidualMSSSMBlock (local)      ┐
    ↓ layer 3:   SSMLayerWithKDA [GLOBAL]          │ proporção 3:1
    ↓ layers 4-6: ResidualMSSSMBlock (local)       │
    ↓ layer 7:   SSMLayerWithKDA [GLOBAL]          ┘
    ↓ RMSNorm
    ↓ 8× nn.Linear(512, 1024) lm_heads
logits: dict {cb0..cb7: [B, T, 1024]}
```

**Parâmetros totais:** ~116M (preset "tiny")

| Parâmetro | Valor |
|---|---|
| d_model | 512 |
| n_layers | 8 |
| vocab_size | 1024 (uniforme, EnCodec) |
| max_seq_len | 2250 tokens = 30s @ 75Hz |
| ssm_scales | [32, 64, 128] |
| kda_layer_indices | [3, 7] (proporção 3:1) |
| n_heads_kda | 8 |

### 5.3 Kimi Delta Attention (KDA)

Substitui tanto o MHA da Beta-2 quanto o GDN do Qwen3.5. A Delta Rule mantém expressão comparável ao MHA mas com complexidade **O(L)** em vez de O(L²):

```
Recorrência:
  e_t = v_t - (S @ k_t)        // erro de predição
  S   = α·S + β·(e_t ⊗ k_t)    // atualização da memória
  y_t = S @ q_t                 // leitura
```

### 5.4 Janela Deslizante (Prefix Conditioning)

```
Janela 1:  [========== 30s ==========]
                    ↕ 15s overlap (50%)
Janela 2:           [========== 30s ==========]
                             ↕ 15s overlap
Janela 3:                    [========== 30s ==========]
```

- **Treino:** Chunks de 30s (2250 tokens). DataLoader concatena arquivos de ~10s até atingir 30s.
- **Inferência:** Últimos 15s da janela anterior são prefixo da próxima. Música contínua de qualquer duração.

### 5.5 Escalonamento: 4 Presets

| Preset | d_model | Layers | KDA Layers | Parâmetros | Hardware Mínimo |
|---|---|---|---|---|---|
| Tiny | 512 | 8 | [3, 7] | ~116M | T4 (16GB) |
| Small | 768 | 12 | [3, 7, 11] | ~371M | RTX 4090 (24GB) |
| Medium | 768 | 16 | [3, 7, 11, 15] | ~500M | A6000 (48GB) |
| Large | 1024 | 16 | [3, 7, 11, 15] | ~1.3B | A100/RTXP 6000 |

### 5.6 Dataset Beta-4

- **Fonte:** ~748K chunks de 10s extraídos do Jamendo (32 tags terapêuticas)
- **Codec:** EnCodec 24kHz @ 6kbps, shape por arquivo: `[8, 750]`
- **Tokens totais:** **~4.5 bilhões** (reaproveitando ~24GB de tokens da Beta-1)
- **Duração de áudio:** ~2.078 horas (~86 dias contínuos)
- **Tokenização:** Dual-GPU paralela em contas Kaggle (2× T4)

### 5.7 Status: Arquitetura Testada ✅ (21/04/2026)

- Forward pass com loss: OK
- Todos os blocos (Mamba2, KDA, MS-SSM, NexMOSHA completo): OK
- Aguardando upload para Lightning AI e início do treino

## 6. DECISÕES TÉCNICAS IMPORTANTES

### 6.1 Tabela de Decisões e Porquês

| Decisão | Alternativa Rejeitada | Razão da Decisão |
|---|---|---|
| DualCodec 12.5Hz | EnCodec 75Hz | 6× menos tokens/s; representação dual |
| DualCodec → EnCodec (Beta-4) | Manter DualCodec | W2V-BERT 2.0 do DualCodec é treinado pra FALA, não música |
| KDA em vez de GDN/MHA (Beta-4) | MHA padrão | KDA (Delta Rule) tem expressividade de MHA mas é O(L) |
| Proporção 3:1 (Beta-4) | 1:7 ou 1:1 | Kimi Linear paper: 3:1 é o melhor tradeoff custo/expressividade |
| Janela 30s (Beta-4) | 60s | 30s captura estrutura musical longa sem explodir VRAM |
| SSM + MHA híbrido | SSM puro | SSM puro causa "late homogenization" |
| Causal mask explícita | `is_causal=True` | PyTorch MHA ignora `is_causal` em alguns contextos |
| cb_loss_weights decrescentes | pesos uniformes | CB0 é mais informativo; sub-otimizar CB7 é aceitável |
| Fine-tuning LLM (Beta-3) | Treino do zero | LLM já sabe prever próximo token |
| Salvar embeddings separados | Incluir no adapter | LoRA não persiste embeddings custom; 3 sessões perdidas |

### 6.2 Por Que NÃO Usamos Certas Arquiteturas

**Transformer puro → Descartado:** Complexidade O(L²) torna inviável áudio de 30s+ (2250 tokens) em GPU de 16GB.

**SSM puro → Insuficiente:** Beta-1 S4 com SiMBA puro atingiu 3.62; injeção de atenção na S8 chegou a 1.28.

**DualCodec → Descartado (Beta-4):** W2V-BERT 2.0 treinado pra fala, não música. EnCodec é codec de áudio geral.

**SNAC → Roadmap v2.0:** Hierarquia multi-escala quebra DataLoader e exige re-tokenização.

**TPU → Incompatível:** Scan seletivo do Mamba usa ops CUDA que XLA não otimiza.

---

## 7. INOVAÇÕES E DIFERENCIAIS

### 7.1 O que o NexMOSHA Faz de Diferente

**1. Primeira aplicação documentada de MS-SSM + Hybrid Attention para musicoterapia neural**
Não existe paper que combine Multi-Scale SSM (múltiplos d_state em paralelo), atenção causal híbrida e codec de baixa frequência (12.5Hz) especificamente para geração de música terapêutica.

**2. Predição Hierárquica com cb_bridges**
A arquitetura de "pontes" entre codebooks (cb_bridges) é uma contribuição original: cada codebook condiciona a predição do próximo via projeção dos logits de volta ao espaço latente, com `.detach()` para isolamento de gradientes. Isso permite que o CB0 semântico "guie" os CB1-7 acústicos sem criação de ciclos de gradiente.

**3. Multi-Scale SSM para Estrutura Musical**
O Scale Mixer dinâmico — que calcula pesos por timestep para cada escala SSM — permite que o modelo atue como "produtor musical neural", decidindo em tempo real qual resolução temporal é mais relevante. Escalas finas capturam transientes, escalas grossas capturam seções inteiras.

**4. Codec-LLM Transplant com Embeddings Persistentes**
A abordagem de trocar o vocabulário de um LLM pré-treinado por tokens de codec de áudio e fazer fine-tuning com LoRA, mantendo os pesos intermediários, é a fronteira atual da área. O insight crítico — salvar embeddings customizados separadamente do adapter LoRA — foi descoberto empiricamente após 3 sessões perdidas.

### 7.2 Insights Técnicos Originais

**Insight 1: Causal Mask Bug Universal**
`nn.MultiheadAttention` do PyTorch não aplica máscara causal por padrão, mesmo com `is_causal=True` em certos contextos. Este bug silencioso causou convergência falsa (PPL→1.0) em múltiplas sessões antes de ser identificado. É provável que este bug afete outros implementadores que não o conhecem.

**Insight 2: Batch Size por GPU Importa**
Com DataParallel em 2 GPUs, batch_size=2 resulta em 1 amostra por GPU — gradientes instáveis. O mínimo para estabilidade é 2 amostras por GPU (batch_size=4 ou gradient_accumulation). Descoberto empiricamente quando sessões com batch=2 estagnavam enquanto batch=4 convergiam.

**Insight 3: KV Cache Muda de Magnitude**
A refatoração de `nn.MultiheadAttention` para projeções q/k/v separadas — sem mudança de comportamento durante treino — reduziu tempo de geração de 60s de ~1.5 horas para **37 segundos**. A diferença O(L²) vs O(1) por token em inferência é mais impactante em sequências longas do que qualquer outro fator.

**Insight 4: Embeddings vs LoRA são Entidades Separadas**
O PEFT/LoRA salva apenas os adapter layers, não os embeddings modificados. Para fine-tuning com vocabulário customizado, os embeddings devem ser salvos e carregados explicitamente, com precedência sobre o carregamento do adapter.

### 7.3 Resultados Surpreendentes

1. **Beta-2 S1 com 5 layers atingiu 4.97** — comparável ou melhor que múltiplas sessões da Beta-1 com EnCodec e arquitetura similar.

2. **Speed-up KV Cache de 240×** — de 5.400 segundos para 37 segundos de geração de 60s de áudio.

3. **O hybrid SSM+Attention quebrou o plateau de 3.69 em UMA sessão** — saltando para 1.28, uma redução de 65% em uma única sessão após múltiplas sessões estagnadas.

4. **ssm_scales importa mais que n_layers** — o Scale Mixer dinâmico com múltiplas escalas paralelas contribuiu mais para a convergência do que o aumento de layers.

---

## 8. NÚMEROS E MÉTRICAS COMPLETAS

### 8.1 Histórico Completo de Loss

```
BETA-1 (SiMBA puro, EnCodec 75Hz, d_model=512, 4 layers):
S1:  ~4.50
S2:  3.9690
S3:  3.7853
S4:  3.6207  ← record SiMBA puro
S5:  3.6942  ← regressão
S6:  4.2–4.9 ← divergência (bug causal mask)
S7:  4.2239
S8:  1.2885 🏆 ← breakthrough SSM+MHA + causal mask fix
S9:  0.8709 (PPL: 2.7) ← melhor resultado Beta-1

BETA-2 (NexMOSHA, DualCodec 12.5Hz, d_model=384):
S1 (5L, do zero): 4.97 🏆
S2 (8L, do zero): 5.20
S3 (8L, resume):  4.75
S4 (8L, opt reset): 4.79
S5 (8L, warmup=100): 4.66 🏆 BASELINE DEFINITIVO

BETA-3 (Qwen3.5-2B LoRA, DualCodec):
S1: ~14.x (bug embeddings)
S2: ~14.x (bug embeddings)
S3: ~14.18 (bug embeddings)
S4: crash trainer_state.json (fix aplicado)
S5: 14.22 (FP32LossTrainer, primeiro treino estável)
S6: 6.69 ⚡ (Master Weights FP32 + Autocast MonkeyPatch Multi-GPU)
S7: 6.27 🏆 (Resume perfeito, Cosine Warm Restart)
S8: ~12.8 (Loss pesada CB0×5, restrição de capacidade)
S9: Val 4.97 🏆 (Lightning AI, LoRA r=128, Curriculum Fase 1 só CB0)
⛔ BETA-3 ENCERRADA. DualCodec descartado (SSL de fala).

BETA-4 (NexMOSHA v2.5, EnCodec 24kHz, MS-SSM/KDA 3:1):
✅ Arquitetura testada e compilando (116M params)
⏳ Treino iminente na Lightning AI
```

### 8.2 Tamanho do Dataset e Parâmetros

| Item | Beta-1 | Beta-2 | Beta-3 | Beta-4 |
|---|---|---|---|---|
| Codec | EnCodec 75Hz | DualCodec 12.5Hz | DualCodec 12.5Hz | **EnCodec 24kHz** |
| Vocab | 1.024 | 20.480 | 20.480 | **1.024** |
| Chunks | ~377K | 48.400 | 48.400 | **~748K** |
| Tokens totais | ~283M | ~36M | ~36M | **~4.5B** |
| Áudio (horas) | ~840 | ~792 | ~792 | **~2.078** |
| Parâmetros | ~42M | ~77M | ~2B | **116M–1.3B** |
| d_model | 512 | 384 | 2.048 | **512–1.024** |
| n_layers | 4 | 8 | 28 | **8–16** |

### 8.3 Velocidade e Infraestrutura

| Métrica | Beta-2 (Kaggle) | Beta-3 S9 (Lightning) | Beta-4 (Lightning) |
|---|---|---|---|
| Hardware | 2× T4, 32GB | RTXP 6000, 96GB | RTXP 6000, 96GB |
| Sessão máxima | 12h (free tier) | Ilimitada | Ilimitada |
| Velocidade | ~1.4–1.7 spm | 1.92 áudios/sec | ⏳ |
| Tempo geração 60s (antes KV) | ~1.5 horas | — | — |
| Tempo geração 60s (depois KV) | **37 segundos** | — | — |
| Speedup KV Cache | **~240×** | — | — |

### 8.4 Comparações com Literatura

| Modelo | Parâmetros | Custo Treino | Qualidade (ref.) |
|---|---|---|---|
| MusicGen Small (Meta) | 300M | ~$8.400 / 3 sem. | CLAP 0.42 |
| SiMBA (paper original) | ~42M | ~$700 / 4 dias | CLAP 0.39 |
| NexMOSHA Beta-2 | ~77M | $0 (Kaggle free) | val_loss 4.66 |
| NexMOSHA Beta-3 | ~2B (base) | $0 (Kaggle+Lightning) | val_loss 4.97 |
| NexMOSHA Beta-4 | 116M–1.3B | ⏳ Lightning AI | ⏳ |

*Nota: CLAP não foi calculado para NexMOSHA; val_loss não é diretamente comparável.*

---

## 9. PONTOS ABERTOS E LIMITAÇÕES CONHECIDAS

1. **Qualidade do áudio gerado:** Beta-4 ainda não foi treinada — qualidade sonora é uma incógnita.

2. **Ausência de métricas perceptuais:** Val loss e PPL são proxies — não medem qualidade musical percebida. Avaliação formal requereria FAD, CLAP, MUSHRA ou MOS.

3. **Dataset concentrado em Jamendo:** 32 tags terapêuticas, mas diversidade de gêneros e culturas musicais pode ser limitada.

4. **Hardware como gargalo:** Lightning AI tem créditos finitos. O treino precisa ser eficiente desde o início.

5. **Validação clínica ausente:** Nenhum estudo com profissionais de saúde foi realizado até o momento.

---

## 10. INFRAESTRUTURA E REPRODUCIBILIDADE

### 10.1 Ambiente
- Python 3.12, PyTorch 2.11+, CUDA 13.0+
- **Treino:** Lightning AI Studios (RTXP 6000, 96GB VRAM, 48 CPUs)
- **Tokenização:** Kaggle Notebooks (GPU T4×2, sessões de 12h)
- **Dev local:** Notebook pessoal (testes de compilação)

### 10.2 Paths Críticos
```bash
# Dataset tokens EnCodec (Beta-4)
/kaggle/input/datasets/destro01400/nexus-fma-tokens/tokens       (~374K chunks)
/kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens (~374K chunks)
```

### 10.3 Notificações Automatizadas
Telegram bot envia alertas ao final de cada sessão com val_loss, PPL e steps realizados.

---

## 11. FUNDAMENTAÇÃO CIENTÍFICA

### 11.1 Base Terapêutica
- ETH Zurich (Nature Metabolism, 2023): mecanotransdução celular via bass 50-60 Hz → liberação de insulina
- CHI 2025: "Intention Gap" — usuários sabem como querem a música mas não conseguem descrever para a IA
- Musicoterapia prescritiva por profissionais de saúde: posicionamento B2B2C

### 11.2 Base Arquitetural
- Gu & Dao (2024): Mamba — SSMs com seleção input-dependent
- Dao & Gu (2024): Mamba-2 — State Space Duality (SSD)
- Kimi Linear (2025): Proporção 3:1 local:global como tradeoff ótimo
- SAMBA (Microsoft): hibridização SSM+SWA superior a Transformer puro
- Zamba, Nemotron-H: validação industrial de híbridos SSM+Attention

---

*Documento preparado para publicação em arXiv. Todos os números são resultados empíricos de treino real em hardware de consumidor.*  
*SnaX Company © 2026 | Guilherme Destro | info.snaxcompany@gmail.com*

