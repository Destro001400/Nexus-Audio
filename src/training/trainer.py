"""
Trainer for SiMBA Music Model

Handles:
- Training loop with mixed precision
- Gradient accumulation
- Checkpointing
- Logging to TensorBoard/W&B
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
    Trainer for SiMBA music generation model.
    
    Features:
    - Mixed precision training (fp16/bf16)
    - Gradient accumulation for large effective batch sizes
    - Learning rate scheduling with warmup
    - Checkpoint saving and resuming
    - Logging to TensorBoard and W&B
    
    Args:
        model: SiMBAMusic model
        train_dataloader: Training data
        val_dataloader: Validation data
        config: Training configuration dict
        output_dir: Directory for checkpoints and logs
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
        
        # Default config
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
            
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["max_steps"] - self.config["warmup_steps"],
            eta_min=self.config["min_lr"],
        )
        
        # Setup mixed precision
        self.use_amp = self.config["fp16"] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient checkpointing
        if self.config["gradient_checkpointing"]:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                
        # Logging
        self.logger = self._setup_logging()
        
        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate weight decay for different parameter types
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
        """Setup logging backend."""
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
        """Log metrics to all backends."""
        if "tensorboard" in self.logger:
            for k, v in metrics.items():
                self.logger["tensorboard"].add_scalar(k, v, step)
                
        if "wandb" in self.logger:
            wandb.log(metrics, step=step)
            
    def _warmup_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        warmup_steps = self.config["warmup_steps"]
        
        if step < warmup_steps:
            return self.config["learning_rate"] * step / warmup_steps
        else:
            return self.scheduler.get_last_lr()[0]
            
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training steps: {self.config['max_steps']:,}")
        
        self.model.train()
        accumulation_steps = self.config["gradient_accumulation_steps"]
        
        # Create data iterator
        data_iter = iter(self.train_dataloader)
        
        # Progress bar
        pbar = tqdm(
            total=self.config["max_steps"],
            desc="Training",
            initial=self.global_step,
        )
        
        accumulated_loss = 0.0
        start_time = time.time()
        
        while self.global_step < self.config["max_steps"]:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
                
            # Move to device
            waveform = batch["waveform"].to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(waveform=waveform, labels=None)
                
                # Self-supervised loss (reconstruction + next token)
                # For now, use the internal loss calculation
                loss = outputs.get("loss")
                
                if loss is None:
                    # If no explicit loss, use dummy loss for testing
                    logits = outputs["logits"]
                    loss = torch.tensor(0.0, device=self.device)
                    
                loss = loss / accumulation_steps
                
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (self.global_step + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"],
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                # LR scheduling
                if self.global_step >= self.config["warmup_steps"]:
                    self.scheduler.step()
                else:
                    # Manual warmup
                    lr = self._warmup_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                        
            # Logging
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
                )
                
                accumulated_loss = 0.0
                start_time = time.time()
                
            # Evaluation
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
                
            # Save checkpoint
            if (self.global_step + 1) % self.config["save_every_n_steps"] == 0:
                self.save_checkpoint(f"step_{self.global_step}")
                
            self.global_step += 1
            pbar.update(1)
            
        pbar.close()
        self.save_checkpoint("final")
        print("Training complete!")
        
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            waveform = batch["waveform"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(waveform=waveform)
                loss = outputs.get("loss", torch.tensor(0.0))
                
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / max(n_batches, 1)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
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
        print(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Loaded checkpoint from step {self.global_step}")
