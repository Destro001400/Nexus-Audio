# Guia Completo: Treinando Seu Modelo SSM de Áudio
## Do Zero ao Deploy — Nexus Audio (SnaX Company)

**Por:** SnaX Company Research Team  
**Data:** Janeiro 2026  
**Nível:** Intermediário a Avançado  
**Tempo Estimado:** 2-4 semanas (dependendo do dataset)

---

## 📋 Índice

1. [Fundamentos de SSMs](#1-fundamentos-de-ssms)
2. [Arquitetura SiMBA para Áudio](#2-arquitetura-simba-para-áudio)
3. [Setup do Ambiente](#3-setup-do-ambiente)
4. [Preparação de Dados](#4-preparação-de-dados)
5. [Implementação do Modelo](#5-implementação-do-modelo)
6. [Treinamento](#6-treinamento)
7. [Fine-Tuning e LoRA](#7-fine-tuning-e-lora)
8. [Inferência e Deploy](#8-inferência-e-deploy)
9. [Otimização e Quantização](#9-otimização-e-quantização)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Fundamentos de SSMs

### 1.1 O Que São State Space Models?

SSMs são uma família de modelos que processam sequências (como áudio, texto, vídeo) usando uma representação de "estado" que evolui ao longo do tempo.

**Comparação Rápida:**

```
TRANSFORMER:
Input → Self-Attention (olha pra TODA a sequência anterior) → Output
Problema: O(L²) — quanto mais longo, exponencialmente mais lento

SSM (Mamba):
Input → State Compression (resume histórico em vetor fixo) → Output
Vantagem: O(L) — cresce linearmente, processa sequências infinitas
```

**Por que funciona pra áudio:**
- Áudio é uma sequência MUITO longa (44100 samples/segundo)
- Transformers explodem em memória com áudio >30 segundos
- SSMs processam horas de áudio com memória constante

---

### 1.2 A Arquitetura Mamba

Mamba é a arquitetura SSM state-of-the-art (2023-2024). SiMBA é uma versão simplificada otimizada especificamente para música.

**Componentes Principais:**

```python
# Pseudo-código da arquitetura Mamba
class MambaBlock:
    def __init__(self):
        self.input_projection = Linear(d_model, d_state)
        self.ssm = SelectiveSSM(d_state)  # O coração do modelo
        self.output_projection = Linear(d_state, d_model)
        
    def forward(self, x):
        # x shape: (batch, length, d_model)
        
        # 1. Projetar input pro espaço de estado
        z = self.input_projection(x)
        
        # 2. Processar com SSM (aqui a mágica acontece)
        h = self.ssm(z)
        
        # 3. Projetar de volta
        y = self.output_projection(h)
        
        return y
```

**SelectiveSSM** é onde a complexidade O(L) acontece:
- Mantém um "estado oculto" de tamanho fixo (d_state, ex: 16-64)
- A cada novo input, atualiza o estado
- Usa o estado pra prever o próximo output
- Não precisa olhar pra trás — tudo tá no estado

---

### 1.3 SiMBA: Simplificado para Música

SiMBA adiciona otimizações específicas pra áudio:

1. **Channel Mixing**: Mistura informação entre diferentes bandas de frequência
2. **Quantização de Tokens**: Trabalha com tokens discretos (não raw audio)
3. **Multi-Scale Processing**: Processa áudio em múltiplas resoluções

---

## 2. Arquitetura SiMBA para Áudio

### 2.1 Pipeline Completo

```
RAW AUDIO (waveform)
    ↓
[ENCODER] → EnCodec ou DAC
    ↓
DISCRETE TOKENS (sequência de inteiros)
    ↓
[EMBEDDING LAYER]
    ↓
TOKEN EMBEDDINGS (vetores contínuos)
    ↓
[SIMBA BLOCKS] × N (8-12 camadas)
    ↓
HIDDEN STATES
    ↓
[PREDICTION HEAD]
    ↓
PREDICTED NEXT TOKEN (logits)
    ↓
[DECODER] → EnCodec ou DAC Decoder
    ↓
GENERATED AUDIO (waveform)
```

---

### 2.2 Diagrama Detalhado

```
┌─────────────────────────────────────────────────┐
│              INPUT: Audio Waveform               │
│          [44100 samples/sec × duration]          │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            NEURAL AUDIO CODEC (EnCodec)          │
│  - Comprime áudio 40x (1s = ~75 tokens)          │
│  - RVQ com 4-8 codebooks                         │
│  - Output: sequência de tokens [t1, t2, ..., tN] │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            EMBEDDING LAYER                       │
│  - Converte tokens → vetores densos              │
│  - Dimension: 256-512                            │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          SIMBA BLOCK 1                           │
│  ┌───────────────────────────────────┐           │
│  │ 1. Input Projection               │           │
│  │ 2. Selective SSM (O(L) magic)     │           │
│  │ 3. Channel Mixing                 │           │
│  │ 4. Output Projection + Residual   │           │
│  └───────────────────────────────────┘           │
└───────────────────┬─────────────────────────────┘
                    ↓
          ... (repeat 7-11 mais vezes) ...
                    ↓
┌─────────────────────────────────────────────────┐
│          SIMBA BLOCK N                           │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          PREDICTION HEAD                         │
│  - Linear layer → vocabulary size (1024-2048)    │
│  - Softmax → probabilidade de cada token         │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          SAMPLING STRATEGY                       │
│  - Top-k, Top-p, Temperature                     │
│  - Seleciona próximo token                       │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          DECODER (EnCodec Decoder)               │
│  - Tokens → waveform                             │
│  - Output: 44100 samples/sec                     │
└─────────────────────────────────────────────────┘
```

---

## 3. Setup do Ambiente

### 3.1 Hardware Necessário

**Mínimo (para experimentos):**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 32GB
- Storage: 500GB SSD

**Recomendado (para treinamento sério):**
- GPU: NVIDIA RTX 3090 ou 4090 (24GB VRAM)
- RAM: 64GB
- Storage: 1TB NVMe SSD

**Ideal (para escala):**
- GPU: NVIDIA A100 (40GB ou 80GB)
- RAM: 128GB+
- Storage: 2TB+ NVMe SSD

---

### 3.2 Software Stack

```bash
# Sistema operacional: Ubuntu 22.04 LTS recomendado

# Python 3.10 ou 3.11
python --version  # verificar

# Criar ambiente virtual
python -m venv nexus_env
source nexus_env/bin/activate  # Linux/Mac
# ou
nexus_env\Scripts\activate  # Windows

# Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar bibliotecas essenciais
pip install transformers
pip install accelerate
pip install datasets
pip install soundfile
pip install librosa
pip install einops
pip install wandb  # para tracking de experimentos

# Instalar EnCodec (codec de áudio neural)
pip install encodec

# Instalar Mamba (implementação oficial)
pip install mamba-ssm
pip install causal-conv1d>=1.1.0

# Verificar instalação
python -c "import torch; print(torch.cuda.is_available())"
# Deve retornar True se GPU está disponível
```

---

### 3.3 Estrutura de Diretórios

```
nexus-audio/
├── data/
│   ├── raw/              # Áudios originais
│   ├── processed/        # Tokens processados
│   └── metadata.json     # Metadados dos áudios
├── models/
│   ├── checkpoints/      # Modelos salvos durante treinamento
│   └── final/            # Modelo final
├── configs/
│   └── simba_config.yaml # Configuração do modelo
├── src/
│   ├── model.py          # Definição do modelo
│   ├── dataset.py        # Dataset loader
│   ├── train.py          # Script de treinamento
│   └── inference.py      # Script de inferência
├── notebooks/
│   └── exploration.ipynb # Jupyter para testes
└── requirements.txt
```

---

## 4. Preparação de Dados

### 4.1 Datasets Recomendados

**Para começar (Creative Commons, gratuito):**

1. **Free Music Archive (FMA)**
   - 106.574 faixas, 161 gêneros
   - ~1000 horas de áudio
   - Licença: Creative Commons
   - Download: https://github.com/mdeff/fma

2. **MusicCaps (Google)**
   - 5.521 músicas com descrições textuais
   - Qualidade alta
   - Ideal para text-to-music
   - Download: https://www.kaggle.com/datasets/googleai/musiccaps

3. **Jamendo (via MTG)**
   - ~55.000 faixas
   - Licença CC
   - Download: https://mtg.github.io/mtg-jamendo-dataset/

**Se tiver orçamento (licenciado):**

4. **AudioSet (Google)**
   - 2 milhões de clips
   - 632 classes de áudio
   - Precisa negociar licença

---

### 4.2 Script de Preprocessamento

```python
# src/preprocess.py

import os
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm
import json

class AudioPreprocessor:
    def __init__(self, model_name="facebook/encodec_24khz", 
                 target_sr=24000, bandwidth=6.0):
        """
        Args:
            model_name: qual modelo EnCodec usar
            target_sr: taxa de amostragem alvo (24kHz recomendado)
            bandwidth: qualidade (1.5, 3.0, 6.0, 12.0, 24.0 kbps)
        """
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self.target_sr = target_sr
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def encode_audio_file(self, audio_path):
        """
        Converte arquivo de áudio → tokens discretos
        
        Returns:
            tokens: tensor de shape (n_codebooks, n_frames)
        """
        # Carregar áudio
        wav, sr = torchaudio.load(audio_path)
        
        # Converter para target sample rate e mono se necessário
        wav = convert_audio(wav, sr, self.target_sr, self.model.channels)
        
        # Normalizar
        wav = wav.unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            wav = wav.cuda()
        
        # Encode
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        
        # encoded_frames é uma lista de tuplas (codes, scale)
        # codes tem shape (batch, n_codebooks, n_frames)
        codes = encoded_frames[0][0]  # Pegar apenas os codes
        
        return codes.squeeze(0).cpu()  # Remove batch dim, volta pra CPU
    
    def process_dataset(self, input_dir, output_dir, max_duration=30):
        """
        Processa todos os áudios de uma pasta
        
        Args:
            input_dir: pasta com áudios .wav, .mp3, .flac
            output_dir: onde salvar os tokens
            max_duration: duração máxima em segundos (pra não estourar memória)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(
                [f for f in os.listdir(input_dir) if f.endswith(ext)]
            )
        
        metadata = []
        
        for audio_file in tqdm(audio_files, desc="Processando áudios"):
            try:
                audio_path = os.path.join(input_dir, audio_file)
                
                # Checar duração antes de processar
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                
                if duration > max_duration:
                    print(f"Skipping {audio_file}: muito longo ({duration:.1f}s)")
                    continue
                
                # Encode
                tokens = self.encode_audio_file(audio_path)
                
                # Salvar tokens
                output_path = os.path.join(
                    output_dir, 
                    audio_file.replace(os.path.splitext(audio_file)[1], '.pt')
                )
                torch.save(tokens, output_path)
                
                # Metadata
                metadata.append({
                    'file': audio_file,
                    'tokens_file': os.path.basename(output_path),
                    'duration': duration,
                    'n_frames': tokens.shape[1],
                    'n_codebooks': tokens.shape[0]
                })
                
            except Exception as e:
                print(f"Erro processando {audio_file}: {e}")
                continue
        
        # Salvar metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nProcessamento completo!")
        print(f"Total processado: {len(metadata)} arquivos")
        print(f"Tokens salvos em: {output_dir}")

# Uso
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(bandwidth=6.0)  # 6 kbps
    
    preprocessor.process_dataset(
        input_dir="data/raw/fma_small",
        output_dir="data/processed/fma_tokens",
        max_duration=30  # 30 segundos máximo
    )
```

**Rodar:**
```bash
python src/preprocess.py
```

---

### 4.3 Dataset PyTorch

```python
# src/dataset.py

import torch
from torch.utils.data import Dataset
import json
import os

class AudioTokenDataset(Dataset):
    def __init__(self, tokens_dir, sequence_length=1024, stride=512):
        """
        Dataset de tokens de áudio
        
        Args:
            tokens_dir: pasta com arquivos .pt de tokens
            sequence_length: tamanho da sequência de treinamento
            stride: quanto pular entre sequências (overlap se < seq_length)
        """
        self.tokens_dir = tokens_dir
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Carregar metadata
        with open(os.path.join(tokens_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Criar índice de sequências
        self.sequences = []
        for item in self.metadata:
            tokens_file = os.path.join(tokens_dir, item['tokens_file'])
            n_frames = item['n_frames']
            
            # Quantas sequências cabem neste arquivo?
            n_sequences = (n_frames - sequence_length) // stride + 1
            
            for i in range(max(1, n_sequences)):
                start_idx = i * stride
                end_idx = start_idx + sequence_length
                
                if end_idx <= n_frames:
                    self.sequences.append({
                        'file': tokens_file,
                        'start': start_idx,
                        'end': end_idx
                    })
        
        print(f"Dataset criado: {len(self.sequences)} sequências")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        
        # Carregar tokens
        tokens = torch.load(seq_info['file'])
        
        # Extrair sequência
        # tokens shape: (n_codebooks, n_frames)
        sequence = tokens[:, seq_info['start']:seq_info['end']]
        
        # Para treinamento autoregressivo:
        # input = tokens[:-1], target = tokens[1:]
        input_seq = sequence[:, :-1]
        target_seq = sequence[:, 1:]
        
        return {
            'input': input_seq,  # (n_codebooks, seq_len-1)
            'target': target_seq  # (n_codebooks, seq_len-1)
        }

# Teste
if __name__ == "__main__":
    dataset = AudioTokenDataset(
        tokens_dir="data/processed/fma_tokens",
        sequence_length=1024,
        stride=512
    )
    
    sample = dataset[0]
    print("Input shape:", sample['input'].shape)
    print("Target shape:", sample['target'].shape)
```

---

## 5. Implementação do Modelo

### 5.1 SiMBA Block (Coração do Modelo)

```python
# src/model.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class SiMBABlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        Bloco SiMBA (Simplified Mamba para áudio)
        
        Args:
            d_model: dimensão do modelo (ex: 512)
            d_state: dimensão do estado SSM (ex: 16)
            d_conv: tamanho do kernel convolucional (ex: 4)
            expand: fator de expansão (ex: 2)
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Layer norm antes de processar
        self.norm = nn.LayerNorm(d_model)
        
        # Mamba SSM (o coração)
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Channel mixing (específico para áudio)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer norm após channel mixing
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: (batch, length, d_model)
        """
        # Residual connection 1: SSM
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = x + residual
        
        # Residual connection 2: Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x
```

---

### 5.2 Modelo Completo

```python
class Nexus Audio(nn.Module):
    def __init__(
        self,
        vocab_size=1024,  # Tamanho do vocabulário de tokens
        n_codebooks=4,    # Quantos codebooks do EnCodec
        d_model=512,      # Dimensão do modelo
        n_layers=8,       # Número de blocos SiMBA
        d_state=16,       # Dimensão do estado SSM
        d_conv=4,         # Tamanho do conv kernel
        expand=2          # Fator de expansão
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.d_model = d_model
        
        # Embedding layer para cada codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])
        
        # Projeção combinada dos embeddings
        self.input_projection = nn.Linear(d_model * n_codebooks, d_model)
        
        # Stack de blocos SiMBA
        self.blocks = nn.ModuleList([
            SiMBABlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)
        
        # Prediction heads (um para cada codebook)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_codebooks)
        ])
    
    def forward(self, x):
        """
        x: (batch, n_codebooks, length)
        returns: (batch, n_codebooks, length, vocab_size)
        """
        batch_size, n_codebooks, length = x.shape
        
        # Embed cada codebook separadamente
        embeddings = []
        for i in range(n_codebooks):
            emb = self.embeddings[i](x[:, i, :])  # (batch, length, d_model)
            embeddings.append(emb)
        
        # Concatenar embeddings de todos os codebooks
        x = torch.cat(embeddings, dim=-1)  # (batch, length, d_model * n_codebooks)
        
        # Projetar para d_model
        x = self.input_projection(x)  # (batch, length, d_model)
        
        # Passar pelos blocos SiMBA
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm_final(x)
        
        # Predict próximo token para cada codebook
        logits = []
        for i in range(n_codebooks):
            logit = self.output_heads[i](x)  # (batch, length, vocab_size)
            logits.append(logit)
        
        # Stack logits
        logits = torch.stack(logits, dim=1)  # (batch, n_codebooks, length, vocab_size)
        
        return logits
    
    def generate(self, prompt_tokens, max_new_tokens=1000, temperature=1.0, top_k=50):
        """
        Geração autoregressiva
        
        Args:
            prompt_tokens: (n_codebooks, length) tokens iniciais
            max_new_tokens: quantos novos tokens gerar
            temperature: controla aleatoriedade (1.0 = normal, <1.0 = mais determinístico)
            top_k: considera apenas top-k tokens mais prováveis
        """
        self.eval()
        
        # Adicionar batch dimension
        x = prompt_tokens.unsqueeze(0)  # (1, n_codebooks, length)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(x)  # (1, n_codebooks, length, vocab_size)
                
                # Pegar logits do último timestep
                logits = logits[:, :, -1, :] / temperature  # (1, n_codebooks, vocab_size)
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, :, [-1]]] = -float('Inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(
                    probs.view(-1, self.vocab_size), 
                    num_samples=1
                ).view(1, self.n_codebooks, 1)  # (1, n_codebooks, 1)
                
                # Append ao histórico
                x = torch.cat([x, next_tokens], dim=2)
        
        return x.squeeze(0)  # Remove batch dim

# Teste
if __name__ == "__main__":
    model = NexusAudio(
        vocab_size=1024,
        n_codebooks=4,
        d_model=512,
        n_layers=8
    )
    
    # Teste forward
    x = torch.randint(0, 1024, (2, 4, 100))  # (batch=2, codebooks=4, length=100)
    logits = model(x)
    print("Output shape:", logits.shape)  # Deve ser (2, 4, 100, 1024)
    
    # Teste generate
    prompt = torch.randint(0, 1024, (4, 50))  # (codebooks=4, length=50)
    generated = model.generate(prompt, max_new_tokens=100)
    print("Generated shape:", generated.shape)  # Deve ser (4, 150)
```

---

## 6. Treinamento

### 6.1 Script de Treinamento

```python
# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os

from model import NexusAudio
from dataset import AudioTokenDataset

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=100,
        warmup_steps=1000,
        save_dir="models/checkpoints",
        log_wandb=True
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
            eta_min=learning_rate * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Wandb
        if log_wandb:
            wandb.init(
                project="nexus-audio",
                config={
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "model_params": sum(p.numel() for p in model.parameters())
                }
            )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)  # (batch, n_codebooks, seq_len)
            targets = batch['target'].to(self.device)
            
            # Forward
            logits = self.model(inputs)  # (batch, n_codebooks, seq_len, vocab_size)
            
            # Calculate loss para cada codebook
            loss = 0
            for cb in range(logits.shape[1]):  # Para cada codebook
                cb_logits = logits[:, cb, :, :]  # (batch, seq_len, vocab_size)
                cb_targets = targets[:, cb, :]    # (batch, seq_len)
                
                # Flatten pra calcular loss
                cb_logits = cb_logits.reshape(-1, cb_logits.size(-1))
                cb_targets = cb_targets.reshape(-1)
                
                loss += self.criterion(cb_logits, cb_targets)
            
            # Average loss entre codebooks
            loss = loss / logits.shape[1]
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Log
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                logits = self.model(inputs)
                
                loss = 0
                for cb in range(logits.shape[1]):
                    cb_logits = logits[:, cb, :, :].reshape(-1, logits.size(-1))
                    cb_targets = targets[:, cb, :].reshape(-1)
                    loss += self.criterion(cb_logits, cb_targets)
                
                loss = loss / logits.shape[1]
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        wandb.log({"val/loss": avg_loss, "val/epoch": epoch})
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss
        }
        
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(epoch)
            if val_loss is not None:
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss)

# Uso
if __name__ == "__main__":
    # Criar datasets
    train_dataset = AudioTokenDataset(
        tokens_dir="data/processed/fma_tokens",
        sequence_length=1024
    )
    
    # Criar modelo
    model = NexusAudio(
        vocab_size=1024,
        n_codebooks=4,
        d_model=512,
        n_layers=8
    )
    
    # Criar trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=100
    )
    
    # Treinar!
    trainer.train()
```

**Rodar treinamento:**
```bash
python src/train.py
```

---

## 7. Fine-Tuning e LoRA

### 7.1 LoRA (Low-Rank Adaptation)

LoRA permite adaptar o modelo pra tarefas específicas com **<0.015% de novos parâmetros**.

```python
# src/lora.py

import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        """
        LoRA layer
        
        Args:
            in_features: input dimension
            out_features: output dimension
            rank: rank da decomposição (4, 8, 16 típico)
            alpha: scaling factor
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        
        # Matrizes de baixo rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling
        self.scaling = alpha / rank
    
    def forward(self, x):
        # x shape: (..., in_features)
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class NexusAudioWithLoRA(nn.Module):
    def __init__(self, base_model, rank=4, alpha=16):
        """
        Adiciona LoRA no modelo base
        """
        super().__init__()
        
        self.base_model = base_model
        
        # Congelar base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Adicionar LoRA em cada output head
        self.lora_layers = nn.ModuleList([
            LoRALayer(
                in_features=base_model.d_model,
                out_features=base_model.vocab_size,
                rank=rank,
                alpha=alpha
            )
            for _ in range(base_model.n_codebooks)
        ])
    
    def forward(self, x):
        # Base model frozen
        with torch.no_grad():
            # Passar pelos blocos e norm
            batch_size, n_codebooks, length = x.shape
            
            embeddings = []
            for i in range(n_codebooks):
                emb = self.base_model.embeddings[i](x[:, i, :])
                embeddings.append(emb)
            
            x = torch.cat(embeddings, dim=-1)
            x = self.base_model.input_projection(x)
            
            for block in self.base_model.blocks:
                x = block(x)
            
            x = self.base_model.norm_final(x)
        
        # Prediction com LoRA (trainável)
        logits = []
        for i in range(n_codebooks):
            # Output head original (frozen)
            base_logit = self.base_model.output_heads[i](x)
            
            # LoRA adaptation (trainável)
            lora_logit = self.lora_layers[i](x)
            
            # Combine
            logit = base_logit + lora_logit
            logits.append(logit)
        
        logits = torch.stack(logits, dim=1)
        return logits

# Uso
if __name__ == "__main__":
    # Carregar modelo base
    base_model = NexusAudio(vocab_size=1024, n_codebooks=4, d_model=512, n_layers=8)
    base_model.load_state_dict(torch.load("models/final/nexus_audio.pt"))
    
    # Criar modelo com LoRA
    model_lora = NexusAudioWithLoRA(base_model, rank=4, alpha=16)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params (LoRA): {trainable_params:,}")
    print(f"Percentage: {100 * trainable_params / total_params:.4f}%")
    
    # Fine-tune apenas LoRA!
    # (usar mesmo script de treinamento)
```

---

## 8. Inferência e Deploy

### 8.1 Script de Inferência

```python
# src/inference.py

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

from model import NexusAudio

class AudioGenerator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        
        # Carregar modelo
        self.model = NexusAudio(
            vocab_size=1024,
            n_codebooks=4,
            d_model=512,
            n_layers=8
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        
        # Carregar codec
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.0)
        self.codec = self.codec.to(device)
        self.codec.eval()
    
    def generate_from_prompt(
        self,
        prompt_audio_path=None,
        duration_seconds=10,
        temperature=1.0,
        top_k=50
    ):
        """
        Gera áudio continuando de um prompt ou do zero
        """
        if prompt_audio_path:
            # Encode prompt
            wav, sr = torchaudio.load(prompt_audio_path)
            wav = convert_audio(wav, sr, 24000, 1)
            wav = wav.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                encoded = self.codec.encode(wav)
                prompt_tokens = encoded[0][0].squeeze(0)  # (n_codebooks, n_frames)
        else:
            # Start from random or silence
            prompt_tokens = torch.zeros(4, 10, dtype=torch.long, device=self.device)
        
        # Calcular quantos tokens precisa gerar
        # EnCodec 24kHz gera ~75 tokens/segundo
        target_tokens = int(duration_seconds * 75)
        tokens_to_generate = target_tokens - prompt_tokens.shape[1]
        
        # Generate
        with torch.no_grad():
            generated_tokens = self.model.generate(
                prompt_tokens,
                max_new_tokens=tokens_to_generate,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode tokens → waveform
        generated_tokens = generated_tokens.unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            decoded = self.codec.decode([(generated_tokens, None)])
        
        audio = decoded[0].squeeze(0).cpu()  # (channels, samples)
        
        return audio, 24000  # waveform, sample rate
    
    def save_audio(self, waveform, sample_rate, output_path):
        torchaudio.save(output_path, waveform, sample_rate)
        print(f"Áudio salvo: {output_path}")

# Uso
if __name__ == "__main__":
    generator = AudioGenerator("models/final/nexus_audio.pt")
    
    # Gerar 10 segundos de áudio
    audio, sr = generator.generate_from_prompt(
        prompt_audio_path=None,  # Começar do zero
        duration_seconds=10,
        temperature=1.0,
        top_k=50
    )
    
    generator.save_audio(audio, sr, "generated_audio.wav")
```

---

## 9. Otimização e Quantização

### 9.1 Quantização Int8

Para rodar em mobile ou dispositivos com pouca memória:

```python
# src/quantize.py

import torch
from torch.quantization import quantize_dynamic

from model import NexusAudio

def quantize_model(model_path, output_path):
    """
    Quantiza modelo pra Int8 (reduz tamanho 75%)
    """
    # Carregar modelo
    model = NexusAudio(vocab_size=1024, n_codebooks=4, d_model=512, n_layers=8)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Quantização dinâmica
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantizar apenas Linear layers
        dtype=torch.qint8
    )
    
    # Salvar
    torch.save(quantized_model.state_dict(), output_path)
    
    # Comparar tamanhos
    original_size = os.path.getsize(model_path) / (1024 ** 2)  # MB
    quantized_size = os.path.getsize(output_path) / (1024 ** 2)
    
    print(f"Modelo original: {original_size:.2f} MB")
    print(f"Modelo quantizado: {quantized_size:.2f} MB")
    print(f"Redução: {100 * (1 - quantized_size/original_size):.1f}%")

# Uso
quantize_model(
    "models/final/nexus_audio.pt",
    "models/final/nexus_audio_int8.pt"
)
```

---

## 10. Troubleshooting

### 10.1 Problemas Comuns

**Problema: Out of Memory (OOM)**
```
Soluções:
- Reduzir batch_size (começar com 2 ou 1)
- Reduzir sequence_length (1024 → 512)
- Usar gradient accumulation:
  
  accumulation_steps = 4
  for step, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      
      if (step + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
```

**Problema: Loss não diminui**
```
Checagens:
1. Learning rate muito alto? Testar 1e-5
2. Dados corretos? Verificar que targets são shifted
3. Modelo muito pequeno? Aumentar d_model ou n_layers
4. Gradient clipping muito agressivo? Aumentar threshold
```

**Problema: Geração produz áudio ruidoso**
```
Soluções:
- Temperature muito alto → reduzir pra 0.7-0.9
- Top-k muito permissivo → reduzir pra 20-30
- Modelo undertrained → treinar mais epochs
- Codec bandwidth muito baixo → usar 12.0 kbps ao invés de 6.0
```

---

## 🎓 Próximos Passos

Agora você tem **TUDO** que precisa pra treinar seu modelo Nexus Audio!

**Checklist:**
- [ ] Setup do ambiente (Python, PyTorch, CUDA)
- [ ] Download de dataset (FMA ou Jamendo)
- [ ] Preprocessing (converter áudio → tokens)
- [ ] Treinar modelo base (~100 epochs, ~4 dias em RTX 3090)
- [ ] Fine-tune com LoRA pra tarefa específica
- [ ] Quantizar pra Int8 se precisar deploy mobile
- [ ] Testar geração de áudio

**Recursos Adicionais:**
- Paper SiMBA: arxiv.org/abs/2507.06674
- Mamba repo: github.com/state-spaces/mamba
- EnCodec: github.com/facebookresearch/encodec

**Suporte:**
Se tiver dúvidas, pode me chamar! Vou te ajudar em qualquer etapa. 🚀

---

*Criado com 💙 pela SnaX Company Research Team*