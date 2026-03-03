"""
Trainer para o Modelo SiMBA Music
CORRIGIDO v2: bug do labels=None (loss zerada) corrigido

Funcionalidades:
- Training loop com mixed precision
- Gradient accumulation
- Checkpointing
- Logging TensorBoard/W&B
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """
    Trainer para o modelo SiMBA de geração musical.

    Args:
        model: SiMBATherapeutic ou SiMBAMusic
        train_dataloader: Dados de treino
        val_dataloader: Dados de validação (opcional)
        config: Configurações de treino
        output_dir: Diretório para checkpoints e logs
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader=None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Config padrão
        self.config = {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "max_steps": 100000,
            "fp16": True,
            "gradient_checkpointing": False,
            "log_every_n_steps": 100,
            "save_every_n_steps": 1000,
            "eval_every_n_steps": 500,
            "max_grad_norm": 1.0,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "min_lr": 3e-5,
        }
        if config:
            self.config.update(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["max_steps"] - self.config["warmup_steps"],
            eta_min=self.config["min_lr"],
        )

        self.use_amp = self.config["fp16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        if self.config["gradient_checkpointing"]:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.logger = self._setup_logging()

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> AdamW:
        """Cria otimizador AdamW com weight decay seletivo."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "_no_weight_decay" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config["learning_rate"],
            betas=tuple(self.config["betas"]),
        )

    def _setup_logging(self):
        """Configura backends de logging."""
        logger = {}

        if TENSORBOARD_AVAILABLE:
            logger["tensorboard"] = SummaryWriter(
                log_dir=self.output_dir / "tensorboard"
            )

        if WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
            wandb.init(
                project="nexus-audio",
                config=self.config,
                dir=str(self.output_dir),
            )
            logger["wandb"] = wandb

        return logger

    def _log(self, metrics: Dict[str, float], step: int):
        """Loga métricas em todos os backends."""
        if "tensorboard" in self.logger:
            for k, v in metrics.items():
                self.logger["tensorboard"].add_scalar(k, v, step)

        if "wandb" in self.logger:
            wandb.log(metrics, step=step)

    def _warmup_lr(self, step: int) -> float:
        """Calcula learning rate com warmup linear."""
        warmup_steps = self.config["warmup_steps"]
        if step < warmup_steps:
            return self.config["learning_rate"] * step / warmup_steps
        else:
            return self.scheduler.get_last_lr()[0]

    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepara batch para forward pass.

        CORRIGIDO: garante que labels sempre seja fornecido para calcular a loss.
        Antes o Trainer passava labels=None, resultando em loss=0 silenciosamente.
        """
        if "tokens" in batch:
            # Batch de tokens pré-computados (ex: do FastDataset do Kaggle)
            tokens = batch["tokens"].to(self.device)
            return {"tokens": tokens, "labels": tokens}  # self-supervised

        elif "waveform" in batch:
            # Batch de waveforms brutos
            waveform = batch["waveform"].to(self.device)
            return {"waveform": waveform}
            # Nota: para waveform, o modelo precisa tokenizar internamente
            # e então calcular a loss. Isso depende da impl. do modelo.

        else:
            raise ValueError(f"Batch deve conter 'tokens' ou 'waveform'. Keys: {batch.keys()}")

    def train(self):
        """Loop principal de treino."""
        print(f"Iniciando treino em {self.device}")
        print(f"Parâmetros do modelo: {self.model.count_parameters():,}")
        print(f"Steps de treino: {self.config['max_steps']:,}")

        self.model.train()
        accumulation_steps = self.config["gradient_accumulation_steps"]

        data_iter = iter(self.train_dataloader)

        pbar = tqdm(
            total=self.config["max_steps"],
            desc="Treinando",
            initial=self.global_step,
        )

        accumulated_loss = 0.0
        start_time = time.time()

        while self.global_step < self.config["max_steps"]:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # CORRIGIDO: prepara batch com labels garantidos
            prepared = self._prepare_batch(batch)

            with autocast(enabled=self.use_amp):
                outputs = self.model(**prepared)
                loss = outputs.get("loss")

                # CORRIGIDO: agora loss nunca deve ser None se _prepare_batch
                # funcionou corretamente. Mas mantemos a guarda por segurança.
                if loss is None:
                    raise RuntimeError(
                        "Model retornou loss=None. Verifique se labels foi fornecido. "
                        "Isso indica um bug no forward() do modelo."
                    )

                loss = loss / accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            if (self.global_step + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"],
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)  # Mais rápido!

                if self.global_step >= self.config["warmup_steps"]:
                    self.scheduler.step()
                else:
                    lr = self._warmup_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

            if (self.global_step + 1) % self.config["log_every_n_steps"] == 0:
                avg_loss = accumulated_loss * accumulation_steps
                elapsed = time.time() - start_time
                steps_per_sec = self.config["log_every_n_steps"] / elapsed

                metrics = {
                    "train/loss": avg_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/steps_per_sec": steps_per_sec,
                }
                self._log(metrics, self.global_step)

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    spm=f"{steps_per_sec * 60:.1f}",
                )

                accumulated_loss = 0.0
                start_time = time.time()

            if (
                self.val_dataloader is not None
                and (self.global_step + 1) % self.config["eval_every_n_steps"] == 0
            ):
                val_loss = self.evaluate()
                self._log({"val/loss": val_loss}, self.global_step)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

                self.model.train()

            if (self.global_step + 1) % self.config["save_every_n_steps"] == 0:
                self.save_checkpoint(f"step_{self.global_step}")

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        self.save_checkpoint("final")
        print("Treino concluído!")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Avalia no conjunto de validação."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Avaliando"):
            prepared = self._prepare_batch(batch)

            with autocast(enabled=self.use_amp):
                outputs = self.model(**prepared)
                loss = outputs.get("loss")
                if loss is not None:
                    total_loss += loss.item()

            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, name: str):
        """Salva checkpoint do modelo."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint salvo: {path}")

    def load_checkpoint(self, path: str):
        """Carrega checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Checkpoint carregado do step {self.global_step}")
