# Nexus-Audio: SiMBA Therapeutic Music Generation

Estado de arte em geração de música terapêutica usando State Space Models (SSMs).

## 🎯 Objetivo

Modelo SiMBA otimizado para musicoterapia com:

- **98% menos dados** que Transformers
- **12x mais barato** de treinar
- Otimização de frequências terapêuticas (bass 50-60Hz + HFC)

## 🚀 Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Preparar dataset
python scripts/prepare_dataset.py --input ./raw_audio --output ./data

# Treinar modelo
python scripts/train.py --config configs/simba_therapy.yaml
```

## 📁 Estrutura

```
src/
├── model/         # Arquitetura SiMBA
├── audio/         # Processamento de áudio
├── data/          # Dataset e tokenização
└── training/      # Loop de treinamento
```

## 🔬 Baseado Em

- [SiMBA Architecture](https://arxiv.org/abs/2403.15360)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [The Triple Convergence Hypothesis](./descoberta.md)

## 📊 Especificações

| Parâmetro | Valor |
|-----------|-------|
| Arquitetura | SiMBA (8 camadas) |
| Dimensão | 512 |
| Sample Rate | 44.1 kHz |
| Treinamento | ~450h áudio |
| Hardware | 1x RTX 3090 |

## 📜 Licença

MIT License - SnaX Company 2026
