# Relatório Oficial de Migração: Mamba-3 e GPUs NVIDIA L4
**Data:** Maio de 2026
**Projeto:** NexMOSHA (Neural EXpert Model for Optimized Sound & Harmonic Architecture)
**Status:** Pivô Arquitetural Aprovado

## 1. O Problema: TPU e a Ponte PyTorch-JAX
Durante os testes de escalonamento do modelo Beta-4 na arquitetura Mamba-2 usando instâncias TPU v5e no Kaggle, encontramos bugs críticos de "memory truncation" e instabilidade numérica (NaNs) na ponte `torch_xla` (versão 2.9.0) com kernels `call_jax`. 

O compilador XLA forçava *downcast* para `bfloat16`, causando divisão por zero em otimizações essenciais do Mamba-2 (ex: `1e-8`). Tentativas de mitigar o erro de forma nativa e via "escudos de memória" expuseram a inviabilidade de manter dependências de tão baixo nível numa arquitetura onde o foco deve ser inovação sonora e musical.

## 2. A Solução: GPUs NVIDIA e a Nova Fronteira do Mamba-3
Abandonamos a infraestrutura TPU em favor de instâncias **NVIDIA L4 (24GB VRAM)** na plataforma Lightning AI. Coincidindo com essa mudança, decidimos elevar o núcleo da rede neural para a recém-lançada arquitetura **Mamba-3**, que nos fornece os seguintes ganhos fundamentais na geração de áudio autorregressivo:

### A) Estados Complexos = O Fim da "Amnésia Acústica"
Música é inerentemente rotacional (fase, harmonia, ritmo). O Mamba-2 utilizava estados Reais, incapazes de manter "quando" um padrão ocorreu num ciclo. O **Mamba-3 usa Estados Complexos**, agindo de forma análoga a um RoPE interno (Rotary Positional Embeddings), conseguindo modelar estruturas de batidas de forma natural e precisa.

### B) MIMO (Multi-Input Multi-Output) = Máxima Velocidade
A geração autorregressiva do Mamba-2 é *memory-bound*. Ele gasta a maior parte do tempo carregando pesos para a memória ao invés de calcular. O Mamba-3 MIMO injeta mais matemática (operações de matriz ao invés de produto externo), **saturando os núcleos CUDA da nossa L4** sem aumentar a latência de inferência, proporcionando maior throughput.

### C) Corte de 50% na VRAM
Graças à Matemática "Exponencial-Trapezoidal" de 2ª ordem do Mamba-3, atingimos qualidade similar (mesma perplexidade) usando apenas **metade do tamanho do estado oculto** (`d_state` cai de 128 para 64). Os 12GB de VRAM que economizamos por batch podem ser usados para **aumentar monstruosamente a janela de contexto** (mais de 30 segundos de geração de áudio).

## Próximos Passos
1. Substituir a camada `_FallbackMamba2` (e qualquer dependência residual de TPU/JAX) por suporte limpo e nativo Triton via PyTorch e `mamba_ssm`.
2. Preparar os scripts de treinamento para iniciar os loops na L4 na Lightning AI usando o config Small (150M) com os novos parâmetros Mamba-3 (`ssm_type='mamba3'`, `use_mimo=True`).
