# 📦 POST-MORTEM COMPLETO: BETA-3 (Qwen3.5 Fine-tuning)
## NexMOSHA — SnaX Company | Encerrada em Abril 2026

---

## 1. RESUMO EXECUTIVO

A Beta-3 foi uma abordagem de **fine-tuning de LLM pré-treinado** (Qwen3.5-2B/4B) para geração de tokens de áudio DualCodec. A ideia central era aproveitar a capacidade de "next-token prediction" de um modelo de linguagem já treinado e ensiná-lo a "falar" na linguagem dos tokens DualCodec, em vez de treinar um modelo do zero.

**Resultado:** A abordagem produziu avanços técnicos valiosos e atingiu a menor val_loss da história do projeto (4.9697 na S9), mas foi encerrada por decisão estratégica — o overhead de hacks, incompatibilidades de ambiente e complexidade operacional superou os benefícios em relação ao modelo próprio (NexMOSHA Beta-2).

**Decisão final:** Retorno à Beta-2 (NexMOSHA, arquitetura própria SSM+MHA, 77M params).

---

## 2. CRONOLOGIA COMPLETA DAS SESSÕES

| Sessão | Steps | Val Loss | Plataforma | Observação |
|--------|-------|----------|------------|------------|
| S1 | ~1000 | ~14.x | Kaggle 2×T4 | ❌ Bug dos embeddings — re-inicializava a cada sessão |
| S2 | ~1000 | ~14.x | Kaggle 2×T4 | ❌ Bug dos embeddings (mesmo bug, sem diagnóstico) |
| S3 | ~700 | ~14.18 | Kaggle 2×T4 | ❌ Bug dos embeddings identificado |
| S4 | ~700 | — | Kaggle 2×T4 | Fix de embeddings aplicado, crash `trainer_state.json` |
| S5 | 700 | 14.22 | Kaggle 2×T4 | ✅ Primeiro treino estável! `FP32LossTrainer` + `GradScaler` patch |
| S6 | 700 | **6.69** | Kaggle 2×T4 | ⚡ **SUCESSO ESTRUTURAL!** Master Weights FP32 + Monkey-Patch Autocast no DP |
| S7 | 700 | **6.27** 🏆 | Kaggle 2×T4 | ✅ Resume perfeito. Cosine Warm Restart puxou o all-time-low Kaggle |
| S8 | ~700 | ~12.8 | Kaggle 2×T4 | ⚠️ Loss Ponderada (CB0 ×5). Loss travou — restrição física r=16/batch=4 |
| **S9** | **1000** | **4.9697** 🏆🏆 | Lightning AI RTXP 6000 | ⚡ **MENOR VAL_LOSS DA HISTÓRIA!** Curriculum Learning Fase 1 (CB0 only) |

### Progressão da Loss:
```
S1/S2/S3: ~14.x (bug embeddings — nenhum aprendizado real)
S4: crash (fix aplicado, infra quebrou)
S5: 14.22 (estável, mas ainda alto — underflow de gradientes)
S6: 6.69 ⚡ (Master Weights + MonkeyPatch = fim do underflow!)
S7: 6.27 🏆 (Resume Perfeito + SGDR)
S8: 12.8 (Loss Ponderada inflou, capacity wall do LoRA r=16)
S9: Train 10.3958, Val 4.9697 🏆🏆 (Lightning AI + Curriculum Learning)
```

---

## 3. STACK TÉCNICO

| Item | Valor |
|------|-------|
| Modelo base | Qwen3.5-2B (Apache 2.0) |
| Kaggle path (2B) | `/kaggle/input/models/barnobarno/qwen3.5-2b/transformers/unsloth/1` |
| Kaggle dataset (4B) | `abebe9849/qwen35-4b` |
| Fine-tuning (Kaggle) | LoRA r=16, alpha=32, target=`[q,k,v,o_proj]` |
| Fine-tuning (Lightning) | LoRA r=128, alpha=256, target=`[q,k,v,o,gate,up,down_proj]` |
| Vocab | Troca completa: 248.320 → 20.480 (embed_tokens + lm_head substituídos) |
| Precisão Kaggle | FP16 forçado + Master Weights FP32 para LoRA |
| Precisão Lightning | BFloat16 nativo |
| Hardware Kaggle | 2× Tesla T4 (16GB cada) via DataParallel |
| Hardware Lightning | NVIDIA RTXP 6000 (96GB VRAM), 48 CPUs |
| Optimizer | `adamw_torch` (bitsandbytes incompatível com CUDA 12.8) |
| Dataset | DualCodec tokens: 48.400 chunks × 749 tokens (~36M tokens) |

---

## 4. BUGS DESCOBERTOS E CORRIGIDOS

### 4.1 Bug Crítico: Embeddings Re-inicializados (S1-S3)

**Sintoma:** Loss travada em ~14.x por 3 sessões consecutivas. Nenhum aprendizado.

**Causa raiz:** Os embeddings customizados (`embed_tokens` e `lm_head`) eram criados com `nn.init.normal_` a cada sessão, ANTES de carregar o LoRA. Como embeddings não fazem parte do adapter LoRA, cada sessão recomeçava com embeddings aleatórios.

**Fix:**
```python
# SALVAR no final de cada sessão:
torch.save({
    'embed_tokens': model.model.embed_tokens.state_dict(),
    'lm_head':      model.lm_head.state_dict()
}, os.path.join(save_path, 'custom_embeddings.pt'))

# CARREGAR no início da sessão seguinte (ANTES do LoRA!):
emb_ckpt = torch.load(emb_path, map_location='cpu')
model.model.embed_tokens.load_state_dict(emb_ckpt['embed_tokens'])
model.lm_head.load_state_dict(emb_ckpt['lm_head'])
# Só DEPOIS carregar o LoRA
```

**Lição:** Embeddings customizados SEMPRE devem ser persistidos separadamente em fine-tuning com PEFT.

---

### 4.2 Bug: DataParallel + Autocast + Master Weights (S5-S6)

**Sintoma:** Loss travada, crashes de dtype (`float != c10::Half`), NaN/Inf nos logits.

**Causa raiz (em 3 camadas):**
1. `DataParallel` do PyTorch executa forwards em sub-threads, que **não herdam** o contexto `torch.amp.autocast` do Trainer
2. Master Weights FP32 (necessários para GradScaler não sofrer underflow) colidem com model layers FP16 nas sub-threads
3. PEFT cria subclasses dinâmicas (`PeftModelForCausalLM`) — monkey-patch na classe pai `PeftModel` era ignorado pela MRO do Python

**Fix (Class-level Monkey-Patch):**
```python
# 1. Promove LoRA e embeddings para FP32 (Master Weights)
for name, param in model.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.to(torch.float32)

# 2. Captura a classe REAL (subclasse dinâmica do PEFT)
ModelClass = type(model)

if not hasattr(ModelClass, '_original_forward_saved'):
    ModelClass._original_forward_saved = ModelClass.forward

    def autocast_forward(self, *args, **kwargs):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            return ModelClass._original_forward_saved(self, *args, **kwargs)

    ModelClass.forward = autocast_forward
```

**Impacto:** Loss despencou de 14.2 → 6.69 na S6.

---

### 4.3 Bug: Prefixo PEFT nos State Dict Keys

**Sintoma:** `Missing key "weight"` ao carregar embeddings salvos após o LoRA ser acoplado.

**Causa raiz:** Quando embeddings são salvos APÓS o LoRA, as chaves ficam com prefixo `modules_to_save.default.weight` em vez de simplesmente `weight`.

**Fix:**
```python
def unwrap_peft(state_dict):
    if "modules_to_save.default.weight" in state_dict:
        return {"weight": state_dict["modules_to_save.default.weight"]}
    return state_dict
```

---

### 4.4 Bug: LoRA Congelado na Inferência

**Sintoma:** Modelo não gera nada diferente do baseline.

**Fix:** Adicionar `is_trainable=True` ao carregar LoRA para inferência:
```python
model = PeftModel.from_pretrained(model, CHECKPOINT_RESUME, is_trainable=True)
```

---

## 5. HACKS OBRIGATÓRIOS DO AMBIENTE KAGGLE

Estes eram necessários em TODAS as sessões Kaggle. São workarounds de incompatibilidades do ambiente, não bugs do modelo:

```python
# 1. Qwen3.5 não registrado no HuggingFace do Kaggle
config.model_type = "qwen2"

# 2. Mocks de importação (evitar crashes de deps)
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.protobuf.internal.builder'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
# + mock completo de tensorflow

# 3. Desempacotamento correto dos tokens DualCodec p/ inferência
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

---

## 6. DESCOBERTAS TÉCNICAS VALIOSAS

### 6.1 Curriculum Learning Faseado por Sessão

**A grande inovação da Beta-3.** Inspirado em AudioLM e VALL-E (que usam predição estagiada: primeiro semântica, depois acústica), adaptamos o conceito para fine-tuning de Causal LM com DualCodec.

**O problema:** Num Transformer causal treinando com 8 codebooks linearizados, o modelo gasta igual "energia atencional" errando ruído acústico (CB7) e errando estrutura musical (CB0). Isso cria um teto artificial (platô ~6.x).

**A solução em 3 fases:**

| Fase | Sessão | Pesos CB0 | Pesos CB1-7 | Objetivo |
|------|--------|-----------|-------------|----------|
| 1 | S9 | 1.0 | 0.0 | "Aprender a gramática" — só semântica |
| 2 | S10 | 3.0 | 1.0 | "Adicionar timbre" — acústica com defesa semântica |
| 3 | S11+ | 1.0 | 1.0 | "Refinamento" — equalização total |

**Implementação:** Máscara via `(shift_labels < 16384)` no `CurriculumTrainer.compute_loss()`.

**Resultado:** Val Loss caiu de 6.27 (S7, Kaggle) → 4.9697 (S9, Lightning). **Redução de 21%.**

> ⚠️ **ESTA TÉCNICA É VÁLIDA PARA QUALQUER MODELO COM CODEBOOKS HIERÁRQUICOS** — não é exclusiva de LLMs fine-tunados. Deve ser testada na Beta-2 (NexMOSHA).

---

### 6.2 Scheduler — Cosine vs Warm Restarts

**Descoberta (via ExaNet):** `CosineWithWarmRestarts` (SGDR) é prejudicial em fine-tuning avançado. Quando o modelo está num vale de loss (~6.x), picos de LR "chutam" o otimizador para fora do mínimo local.

**Recomendação:** `cosine` padrão com warmup moderado (15%) e LR máximo baixo (`3e-5`).

---

### 6.3 Escalabilidade Lightning AI

A migração do Kaggle (2×T4 16GB = 32GB) para Lightning AI (1×RTXP 6000 96GB) trouxe:

| Métrica | Kaggle | Lightning AI | Ganho |
|---------|--------|--------------|-------|
| LoRA Rank | r=16 | r=128 | 8× mais capacidade |
| Batch Efetivo | 4 | 32 | 8× maiores |
| Throughput | 0.048 áudio/s | 1.92 áudio/s | **40×** |
| Precisão | FP16 + hacks | BFloat16 nativo | Sem hacks |
| DataParallel hacks | Obrigatório | Desnecessário | Simplificação total |

---

### 6.4 Unsloth com Qwen3.5 — DESCARTADO

**Problemas encontrados:**
1. Carrega Qwen3.5 como modelo multimodal (`Qwen3_5ForConditionalGeneration`), não CausalLM
2. Estrutura de módulos diferente: `model.model.language_model.layers.X`
3. Forçou float32 ("float16 won't work") → dobrou VRAM, matou velocidade
4. Config usa `text_config.hidden_size` em vez de `hidden_size` direto
5. LoRA checkpoint keys incompatíveis
6. ETA de 36h para 2100 steps (vs ~10h no notebook original)

**Conclusão:** O overhead do Unsloth anula o benefício. O notebook manual com `config.model_type = "qwen2"` é superior.

---

### 6.5 Gemma 4 — Avaliação

**Lançado:** 2 Abril 2026.

- E2B/E4B suportam áudio como **entrada** (ASR), NÃO como geração de tokens discretos
- Unsloth tem suporte dia-zero
- Bugs dia-zero: PEFT não reconhece `Gemma4ClippableLinear`, `mm_token_type_ids` exigido mesmo em texto-only
- **Ainda precisaria do mesmo hack de substituição de embeddings**

**Status:** Aguardando estabilização. Pode ser irrelevante se a Beta-2 progredir bem.

---

## 7. NOTEBOOKS PRESERVADOS

| Notebook | Localização | Descrição |
|----------|-------------|-----------|
| `train-beta-3.ipynb` | raiz do repo | Notebook de treino S6 (Kaggle) — versão final com todos os fixes |
| `notebooks/nex-generate-beta-3.ipynb` | `notebooks/` | Notebook de geração/inferência Beta-3 |
| `notebooks/train-beta-3-final.ipynb` | `notebooks/` (se existir) | Versão limpa mencionada no post-mortem do Unsloth |

---

## 8. PATHS E CHECKPOINTS

```
# Modelo base Qwen3.5-2B (Kaggle)
/kaggle/input/models/barnobarno/qwen3.5-2b/transformers/unsloth/1

# Modelo 4B (Kaggle dataset)
abebe9849/qwen35-4b

# Dataset DualCodec (compartilhado com Beta-2)
/kaggle/input/datasets/destro01400/nexus-audio-dataset-beta-2/tokens
/kaggle/input/datasets/snaxcompany/nexus-dataset-complementar-1/tokens

# DualCodec codec
/kaggle/input/datasets/destro01400/dualcodec-tokenizer/ckpt/

# Checkpoints S6/S7/S9 — salvos nos outputs das respectivas sessões Kaggle/Lightning
# Cada checkpoint contém: adapter_model.safetensors + custom_embeddings.pt
```

---

## 9. O QUE LEVAR PARA A BETA-2 (LIÇÕES TRANSFERÍVEIS)

### ✅ Aplicar na Beta-2:
1. **Curriculum Learning Faseado** — A técnica funciona com QUALQUER modelo que usa codebooks hierárquicos. Implementar no `SiMBATherapeutic.forward()` com pesos dinâmicos por fase
2. **Cosine scheduler (sem restarts)** — Usar `cosine` padrão com warmup e LR decrescente
3. **Batch efetivo maior** — Se usar Lightning AI, aproveitar batch=32 em vez de batch=4
4. **BFloat16** — Usar bfloat16 nativo em vez de fp16 + GradScaler (se hardware suportar)
5. **Avaliação por codebook separada** — Monitorar loss de cada CB individualmente para diagnosticar gargalos

### ❌ NÃO se aplica à Beta-2:
1. Hacks de DataParallel/Autocast (modelo próprio não usa PEFT)
2. Class-level monkey-patching (não há subclasses dinâmicas)
3. Troca de vocab/embeddings (Beta-2 tem vocabulário nativo)
4. Mocks de importação Kaggle (problema do HuggingFace Transformers, não do PyTorch puro)

---

## 10. RESULTADOS DE INFERÊNCIA (GERAÇÃO DE ÁUDIO)

Os áudios gerados na inferência do checkpoint S6/S7 demonstraram:
- **Alta variação de ZCR (0.05)** — sinal de conteúdo musical real, não ruído estático
- **Estrutura musical e componentes vocais emergentes**
- **Abandono completo do ruído estatístico passivo** das sessões anteriores

Isso valida que o fine-tuning FUNCIONOU — o modelo aprendeu a gerar tokens DualCodec com estrutura musical coerente. A decisão de encerrar não é por falha técnica, mas por eficiência operacional.

---

## 11. CONCLUSÃO

A Beta-3 foi um experimento bem-sucedido em termos de aprendizado técnico, gerando inovações que serão diretamente aplicadas na Beta-2. As principais contribuições são:

1. **Curriculum Learning Faseado** — A técnica mais valiosa, transferível diretamente
2. **Diagnóstico profundo de DataParallel + FP16** — Documentação que serve para qualquer projeto multi-GPU
3. **Validação empírica de fine-tuning de LLM para áudio** — Provou que é possível, mas opera em faixa de complexidade desnecessária quando se tem arquitetura própria
4. **Benchmark de Lightning AI** — Dados de escalabilidade para futura migração de hardware

**A Beta-3 fecha com honras. O conhecimento gerado aquí não morre — ele renasce na Beta-2.** 🏆

---

*Relatório gerado em 08/04/2026 — SnaX Company*
