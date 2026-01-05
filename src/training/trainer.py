"""
Advanced Training Pipeline with:
- Learning rate scheduling
- Early stopping
- Checkpointing
- Logging (MLflow/Wandb)
- Mixed precision training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional, Callable
import time


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class RLMTrainer:
    """Advanced trainer for Regression Language Models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'auto',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'lr': [],
            'epoch_times': []
        }
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Calculate MAE
            with torch.no_grad():
                mae = torch.abs(outputs - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        return {'loss': avg_loss, 'mae': avg_mae}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_batches = len(self.val_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        r2 = 1 - (np.sum((all_targets - all_predictions) ** 2) / 
                  np.sum((all_targets - all_targets.mean()) ** 2))
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train(
        self,
        n_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        save_best: bool = True,
        verbose: bool = True
    ):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            if verbose:
                print(f"\n{'-'*70}")
                print(f"Epoch {epoch+1}/{n_epochs} Summary ({epoch_time:.2f}s)")
                print(f"{'-'*70}")
                print(f"   Train Loss: {train_metrics['loss']:.4f} | Train MAE: {train_metrics['mae']:.4f}")
                print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val MAE:   {val_metrics['mae']:.4f}")
                print(f"   Val RMSE:   {val_metrics['rmse']:.4f} | Val R²:    {val_metrics['r2']:.4f}")
                print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"{'-'*70}\n")
            
            # Save best model
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt', is_best=True)
                print(f"Saved best model (Val Loss: {val_metrics['loss']:.4f})")
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_metrics['loss']):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save final model and history
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Total Training Time: {sum(self.history['epoch_times']):.2f}s")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {path}")
    
    def save_history(self):
        """Save training history"""
        path = self.log_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {path}")


# Training configuration factory
def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict
) -> RLMTrainer:
    """Create trainer with configuration"""
    
    # Loss function
    if config.get('loss_type') == 'hybrid':
        from src.models.rlm import HybridLoss
        criterion = HybridLoss(alpha=config.get('loss_alpha', 0.5))
    elif config.get('loss_type') == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
        betas=config.get('betas', (0.9, 0.999))
    )
    
    # Scheduler
    scheduler_type = config.get('scheduler_type', 'onecycle')
    if scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            epochs=config.get('n_epochs', 50),
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = RLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get('device', 'auto'),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        log_dir=config.get('log_dir', 'logs'),
        use_amp=config.get('use_amp', True)
    )
    
    return trainer


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
