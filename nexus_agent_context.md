# NEXUS-AUDIO — Documento de Contexto para Agente
## SnaX Company | Atualizado: Abril 2026

---

## 1. VISÃO GERAL DO PROJETO

**Nexus-Audio** é uma plataforma B2B2C de musicoterapia neural desenvolvida pela SnaX Company. O modelo gera música terapêutica personalizada com IA, prescrita por profissionais de saúde. A abordagem é conservadora em termos científicos — o produto é posicionado como ferramenta auxiliar sob supervisão profissional, não como tratamento autônomo.

**CEO:** Guilherme Destro, 18 anos, Cacoal/RO  
**Contato:** info.snaxcompany@gmail.com  
**Compute:** Kaggle free tier (2× Tesla T4, sessões de 12h)  
**Codec:** DualCodec 12.5Hz — vocabulário de 20.480 tokens (16.384 semântico + 4.096 acústico × 7)

---

## 2. HISTÓRICO DE ARQUITETURAS

### Beta-1 — SiMBATherapeutic v1 (ENCERRADA)
- **Arquitetura:** SiMBA puro (MS-SSM, 4 layers, d_model=512)
- **Codec:** EnCodec 75Hz — vocab=1024, chunks de 10s (750 tokens)
- **Dataset:** ~377k chunks, ~283M tokens
- **Bug crítico descoberto:** `nn.MultiheadAttention` não aplica causal mask por padrão. Sem ela, o modelo "vê o futuro" → overfitting instantâneo (loss → 0.018, PPL → 1.0). Fix: passar máscara triangular explícita via `generate_square_subsequent_mask`.
- **Breakthrough S8:** Injeção de `AttentionBlock` após layer 2 → loss 1.2885, S9: 0.8709 (PPL 2.7)
- **Motivo do encerramento:** Teto atingido; migração para DualCodec e arquitetura maior.
- **Checkpoints:** `/kaggle/input/models/destro01400/nexus-audio/pytorch/beta-1/9/checkpoints/`

**Histórico de loss Beta-1:**
```
S1: ~4.50 → S2: 3.9690 → S3: 3.7853 → S4: 3.6207
S5: 3.6942 → S6: 4.2–4.9 → S7: 4.2239
S8: 1.2885 🏆 (breakthrough SSM+MHA) → S9: 0.8709 (PPL 2.7)
```

---

### Beta-2 — NexMOSHA (BASELINE DE REFERÊNCIA)
**Nome:** Multi-scale cOntextual State space Hybrid Attention

#### Arquitetura Final
| Parâmetro | Valor |
|---|---|
| d_model | 384 |
| n_layers | 8 |
| n_heads | 6 |
| vocab_size_semantic (CB0) | 16.384 |
| vocab_size_acoustic (CB1–7) | 4.096 |
| max_seq_len | 749 tokens = ~60s |
| ssm_scales | [16, 64, 256] |
| attn_layer | 4 (meio da rede) |
| cb_loss_weights | [3.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5] |
| Parâmetros totais | ~77M |

**Fluxo de dados:**
```
tokens [B, 8, 749]
    ↓ embed_tokens (CB0: Embedding(16384,384) | CB1-7: Embedding(4096,384))
hidden [B, 749, 384]
    ↓ layers 1-3: ResidualMSSSMBlock
    ↓ layer 4:   SSMLayerWithAttn (SSM → AttentionBlock causal obrigatória)
    ↓ layers 5-8: ResidualMSSSMBlock
    ↓ RMSNorm
    ↓ lm_heads hierárquicos com cb_bridges
logits: lista de [B, T, vocab_sizes[i]]
```

#### Dataset Beta-2
- 48.400 chunks × 749 tokens = ~36M tokens
- Paths: `/kaggle/input/datasets/destro01400/nexus-audio-dataset-beta-2/tokens`
- Complementar: `/kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens`

#### Histórico de Treino Beta-2
| Sessão | Config | Best Val Loss |
|---|---|---|
| S1 | 5 layers, do zero | 4.97 |
| S2 | 8 layers, do zero | 5.20 |
| S3 | 8 layers, resume S2 | 4.75 |
| S4 | 8 layers, resume S3 | 4.79 |
| **S5** | **8 layers, resume S3, warmup=100** | **4.66 🏆** |

**Beta-2 atingiu teto em ~4.66. É o baseline de referência.**

#### KV Cache (implementado na Beta-2)
- `AttentionBlock` refatorado com q/k/v_proj separados
- Treino: comportamento idêntico
- Inferência: O(1) por token — geração de 60s caiu de ~1.5h → **37 segundos**
- Conversor automático de pesos antigos (in_proj_weight) → novo formato

#### Gerador VIP Beta-2
- Auto-Seed via Jamendo API (`client_id: e92f52d5`, tags terapêuticas)
- Temperature schedule (temp_inicial → temp_final)
- Batch de múltiplas variações
- Notificações Telegram
- Seed forçado como mono `[1,1,T]` antes de tokenizar (DualCodec exige mono)
- `SEED_DURATION_S = 5`

---

### Beta-3 — Qwen3.5 Fine-tuning (ATIVA)

#### Decisão de Migração
Abandonar treino do zero e fazer fine-tuning de LLM pré-treinado. Motivo: convergência em 3-5 sessões vs 20-30 do zero. A ideia é "transplantar" os embeddings do modelo de texto para tokens de áudio DualCodec.

#### Stack Técnico Beta-3
| Item | Valor |
|---|---|
| Modelo base | Qwen3.5-2B ou 4B |
| Kaggle path (2B) | `/kaggle/input/models/barnobarno/qwen3.5-2b/transformers/unsloth/1` |
| Kaggle dataset (4B) | `abebe9849/qwen35-4b` |
| Fine-tuning | LoRA r=16 (2B) ou QLoRA r=32 (4B) |
| Vocab | 248.320 → 20.480 (embed trocado) |
| Precisão | FP16, adamw_torch |
| Hardware | T4×2 |

#### Hacks Críticos do Ambiente Kaggle (OBRIGATÓRIOS)
```python
# 1. Contornar registro não reconhecido do Qwen3.5
config.model_type = "qwen2"

# 2. Mocks de importação (evitar crashes de dependências)
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.protobuf.internal.builder'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
# + mock completo de tensorflow

# 3. "Plástica matemática" — desempacotamento correto dos tokens DualCodec
tokens = raw_tokens.reshape(1, -1, 8).transpose(1, 2)
# CB0: clamp 0–16383
# CB1–7: subtrair 16384, clamp 0–4095
audio = codec.decode(tokens).squeeze(0)

# 4. Força dtype FP16
model.to(torch.float16)
model.config.torch_dtype = torch.float16

# 5. Evitar bitsandbytes com DataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 6. Evitar pin_memory crash
dataloader_pin_memory = False
```

#### Bug Crítico Identificado nas S1–S3 (JÁ CORRIGIDO)
**Causa raiz:** `nn.init.normal_` nos embeddings `embed_tokens` e `lm_head` era executado ANTES de carregar o LoRA, re-inicializando os embeddings a cada sessão. Resultado: loss ~14.x em todas as sessões, sem progresso.

**Fix implementado:**
```python
# No FINAL de cada sessão — salvar embeddings junto com LoRA:
torch.save({
    'embed_tokens': model.model.embed_tokens.state_dict(),
    'lm_head':      model.lm_head.state_dict()
}, os.path.join(save_path, 'custom_embeddings.pt'))

# No INÍCIO de cada sessão seguinte — carregar ANTES do LoRA:
emb_path = os.path.join(CHECKPOINT_RESUME, 'custom_embeddings.pt')
if os.path.exists(emb_path):
    emb_ckpt = torch.load(emb_path, map_location='cpu')
    model.model.embed_tokens.load_state_dict(emb_ckpt['embed_tokens'])
    model.lm_head.load_state_dict(emb_ckpt['lm_head'])
# Só DEPOIS carregar o LoRA com PeftModel.from_pretrained()
```

#### Histórico Beta-3
| Sessão | Steps | Val Loss | Observação |
|---|---|---|---|
| S1 | ~1000 | ~14.x | Bug embeddings |
| S2 | ~1000 | ~14.x | Bug embeddings |
| S3 | ~700 | ~14.18 | Bug embeddings |
| S4+ | — | ⏳ | Fix implementado, rodando |

---

## 3. PATHS IMPORTANTES

```
# Tokens dataset
/kaggle/input/datasets/destro01400/nexus-audio-dataset-beta-2/tokens
/kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens

# Modelos base
/kaggle/input/models/barnobarno/qwen3.5-2b/transformers/unsloth/1  (Qwen3.5-2B)
abebe9849/qwen35-4b  (Qwen3.5-4B, dataset Kaggle)

# DualCodec
/kaggle/input/datasets/destro01400/dualcodec-tokenizer/ckpt/

# Checkpoints Beta-2 (baseline)
# best val_loss=4.66 (S5) — salvo no Kaggle output da sessão
```

---

## 4. INFRAESTRUTURA

- **Notificações:** Telegram bot `7781713605:AAFzHiduYdHUMAWuw56MY5VHq4LRbty4hZQ`, chat_id `6908956487`
- **Jamendo API:** client_id `e92f52d5`, tags terapêuticas, retry com exponential backoff
- **Tokenização paralela:** Rodar em contas secundárias do Kaggle durante downtime de GPU

---

## 5. DECISÕES ARQUITETURAIS E PORQUÊS

| Decisão | Por quê |
|---|---|
| DualCodec 12.5Hz em vez de EnCodec 75Hz | 6× menos tokens por segundo → sequências menores → viável em T4 |
| SSM + MHA híbrido | SSM puro entra em "late homogenization" (embrumation). MHA preserva contexto global |
| MHA na layer 4 de 8 | SSMs processam primeiro, atenção integra no meio, SSMs refinam no final |
| Causal mask explícita no MHA | `nn.MultiheadAttention` NÃO aplica mask por padrão — lição da Beta-1 |
| cb_loss_weights decrescentes | CB0 semântico é mais importante; CB1–7 acústicos têm peso menor |
| Fine-tuning LLM em vez de treino do zero | Beta-2 levou 5 sessões para atingir teto em 4.66. Fine-tuning deve convergir em 3-5 sessões |
| adamw_torch em vez de adamw_hf | bitsandbytes incompatível com CUDA 12.8 no Kaggle |
| TPU descartado | XLA tem conflito com selective scan (grafo estático) |

---

## 6. LIÇÕES APRENDIDAS CRÍTICAS

1. **Causal mask é obrigatória** no `nn.MultiheadAttention` — `is_causal=True` sozinho é insuficiente
2. **Batch size por GPU:** mínimo 2 amostras/GPU para gradientes estáveis (DataParallel divide o batch)
3. **Embeddings devem ser salvos** junto com LoRA entre sessões — não são parte do adapter
4. **Scheduler não pode resetar** entre sessões — salvar e carregar optimizer + scheduler completos
5. **ssm_scales = [8, 16, 32]** no Kaggle T4 — [16, 64, 256] causa OOM com 8 layers
6. **Avaliar val a cada 1000 steps** (não 500) — economiza ~50% do tempo de sessão

---

## 7. PRÓXIMOS PASSOS

- [ ] Confirmar val_loss da Beta-3 S4 com fix de embeddings
- [ ] Avaliar convergência vs baseline Beta-2 (4.66)
- [ ] Testar Unsloth com Qwen3.5-2B (2× mais rápido, 70% menos VRAM)
- [ ] Gemma 4 E2B: aguardar bugs dia-zero serem resolvidos (~2-3 semanas)
- [ ] Implementar GDN na layer 4 do NexMOSHA (futuro)
- [ ] KV Cache para Beta-3 na geração
- [ ] Coletar 30+ respostas de validação de profissionais de saúde
- [ ] Rodada de investimento: angel R$80–100k por 15–20%

---

## 8. HISTÓRICO COMPLETO DE LOSS

```
Beta-1: S1~4.50 → S2:3.9690 → S3:3.7853 → S4:3.6207 (record SiMBA puro)
        S5:3.6942 → S6:4.2–4.9 → S7:4.2239
        S8:1.2885 🏆 (breakthrough SSM+MHA) → S9:0.8709 (PPL 2.7)

Beta-2: S1:4.97 → S2:5.20 → S3:4.75 → S4:4.79 → S5:4.66 🏆 (teto)

Beta-3: S1/S2/S3: ~14.x (bug embeddings) → S4+: ⏳ (fix aplicado)
```

---

## 9. CONTEXTO DE MERCADO

- **Mercado:** USD 3-4B (2024) → USD 20-54B projetado (2030)
- **Diferencial técnico:** SSMs O(L) viabilizam edge computing; modelos Transformer tradicionais são O(L²)
- **Diferencial de negócio:** B2B2C com prescrição profissional — evita regulação direta de dispositivo médico
- **Posicionamento científico:** Linguagem conservadora ("estudos preliminares", "auxiliar terapêutico") — overclaiming causa problemas regulatórios
- **Base científica:** ETH Zurich 2023 (Nature Metabolism) demonstrou mecanismo molecular música→controle glicêmico via mecanotransdução celular
