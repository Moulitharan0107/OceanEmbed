"""
OceanEmbed Model Training Pipeline
Complete training loop with validation, early stopping, and model evaluation.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.ocean_model import OceanEmbedModel, OceanEmbedLoss, create_model


class OceanEmbedTrainer:
    """
    Handles the full training lifecycle:
    1. Data preparation (split, normalize)
    2. Training with early stopping
    3. Evaluation on held-out test set
    4. Model checkpointing
    5. Metrics computation and logging
    """
    
    def __init__(self, model: OceanEmbedModel = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TRAINER] Using device: {self.device}")
        
        self.model = model or create_model()
        self.model = self.model.to(self.device)
        
        self.criterion = OceanEmbedLoss(monotonicity_weight=0.1)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.metrics = {}
        self.training_history = []
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = None,
        val_ratio: float = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split and normalize data into train/val/test sets.
        
        Args:
            X: (n_samples, n_features) surface features
            y: (n_samples, n_depths) temperature profiles
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_ratio = train_ratio or config.MODEL_CONFIG["train_split"]
        val_ratio = val_ratio or config.MODEL_CONFIG["val_split"]
        
        print(f"\n[DATA] Preparing {len(X)} samples...")
        
        # Train/val/test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=1 - train_ratio - val_ratio, random_state=42
        )
        
        val_fraction = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction, random_state=42
        )
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Normalize features
        X_train_norm = self.scaler_X.fit_transform(X_train).astype(np.float32)
        X_val_norm = self.scaler_X.transform(X_val).astype(np.float32)
        X_test_norm = self.scaler_X.transform(X_test).astype(np.float32)
        
        # Normalize targets
        y_train_norm = self.scaler_y.fit_transform(y_train).astype(np.float32)
        y_val_norm = self.scaler_y.transform(y_val).astype(np.float32)
        y_test_norm = self.scaler_y.transform(y_test).astype(np.float32)
        
        # Store raw test data for evaluation
        self.X_test_raw = X_test
        self.y_test_raw = y_test
        
        # Create DataLoaders
        batch_size = config.MODEL_CONFIG["batch_size"]
        
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_norm),
                torch.tensor(y_train_norm),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_norm),
                torch.tensor(y_val_norm),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test_norm),
                torch.tensor(y_test_norm),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Store SST input for boundary loss (first feature)
        self.X_train_sst = X_train[:, 0].astype(np.float32)
        self.X_val_sst = X_val[:, 0].astype(np.float32)
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        patience: int = None,
    ) -> Dict:
        """
        Train the model with early stopping.
        
        Returns:
            Training history dict
        """
        num_epochs = num_epochs or config.MODEL_CONFIG["num_epochs"]
        patience = patience or config.MODEL_CONFIG["patience"]
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.MODEL_CONFIG["learning_rate"],
            weight_decay=1e-4,
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "learning_rate": [],
        }
        
        print(f"\n{'='*60}")
        print(f"Training OceanEmbed Model")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Patience: {patience}")
        print(f"Batch size: {config.MODEL_CONFIG['batch_size']}")
        print(f"Learning rate: {config.MODEL_CONFIG['learning_rate']}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # ---- Training phase ----
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss, loss_dict = self.criterion(pred, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss_dict["total"])
            
            scheduler.step()
            
            avg_train_loss = np.mean(train_losses)
            
            # ---- Validation phase ----
            val_loss, val_rmse = self._evaluate_loader(val_loader)
            
            # Record history
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            history["learning_rate"].append(current_lr)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs or on first/last
            if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {val_rmse:.4f}°C | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model = self.model.to(self.device)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s ({epoch+1} epochs)")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        self.training_history = history
        return history
    
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a DataLoader. Returns (avg_loss, avg_rmse)."""
        self.model.eval()
        all_losses = []
        all_rmses = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.model(batch_X)
                loss, loss_dict = self.criterion(pred, batch_y)
                all_losses.append(loss_dict["total"])
                
                # Denormalize for RMSE in original units
                pred_np = self._denormalize_y(pred.cpu().numpy())
                true_np = self._denormalize_y(batch_y.cpu().numpy())
                rmse = np.sqrt(np.mean((pred_np - true_np) ** 2))
                all_rmses.append(rmse)
        
        return np.mean(all_losses), np.mean(all_rmses)
    
    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize temperature profiles back to original units."""
        if y_norm.ndim == 1:
            y_norm = y_norm.reshape(1, -1)
        return self.scaler_y.inverse_transform(y_norm)
    
    def _denormalize_X(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize features back to original units."""
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
        return self.scaler_X.inverse_transform(X_norm)
    
    def evaluate(self, test_loader: DataLoader = None) -> Dict:
        """
        Comprehensive evaluation on held-out test set.
        
        Computes:
        - Overall RMSE and R²
        - Per-depth-level RMSE and R²
        - Feature importance (gradient-based)
        
        Returns metrics dict.
        """
        if test_loader is None:
            # Recreate test loader
            X_test_norm = self.scaler_X.transform(self.X_test_raw).astype(np.float32)
            y_test_norm = self.scaler_y.transform(self.y_test_raw).astype(np.float32)
            
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_test_norm),
                    torch.tensor(y_test_norm),
                ),
                batch_size=config.MODEL_CONFIG["batch_size"],
                shuffle=False,
            )
        
        print(f"\n{'='*60}")
        print("Model Evaluation on Held-Out Test Set")
        print(f"{'='*60}")
        
        self.model.eval()
        all_preds_norm = []
        all_true_norm = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                pred = self.model(batch_X)
                all_preds_norm.append(pred.cpu().numpy())
                all_true_norm.append(batch_y.numpy())
        
        preds_norm = np.concatenate(all_preds_norm, axis=0)
        trues_norm = np.concatenate(all_true_norm, axis=0)
        
        # Denormalize
        preds = self._denormalize_y(preds_norm)
        trues = self._denormalize_y(trues_norm)
        
        # Overall metrics
        overall_rmse = np.sqrt(np.mean((preds - trues) ** 2))
        ss_res = np.sum((trues - preds) ** 2)
        ss_tot = np.sum((trues - np.mean(trues, axis=0)) ** 2)
        overall_r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        print(f"\nOverall Metrics:")
        print(f"  RMSE: {overall_rmse:.3f}°C")
        print(f"  R²:   {overall_r2:.4f}")
        
        # Per-depth metrics
        depth_rmse = []
        depth_r2 = []
        
        print(f"\n{'Depth (m)':>10} {'RMSE (°C)':>12} {'R²':>10}")
        print("-" * 35)
        
        for j, depth in enumerate(config.MODEL_DEPTH_LEVELS):
            rmse_j = np.sqrt(np.mean((preds[:, j] - trues[:, j]) ** 2))
            ss_res_j = np.sum((trues[:, j] - preds[:, j]) ** 2)
            ss_tot_j = np.sum((trues[:, j] - np.mean(trues[:, j])) ** 2)
            r2_j = 1 - ss_res_j / (ss_tot_j + 1e-10)
            
            depth_rmse.append(float(rmse_j))
            depth_r2.append(float(r2_j))
            
            print(f"  {depth:>7d}m  {rmse_j:>10.3f}  {r2_j:>8.4f}")
        
        # Feature importance via gradient-based attribution
        feature_importance = self._compute_feature_importance()
        
        # Compile metrics
        self.metrics = {
            "overall_rmse": float(overall_rmse),
            "overall_r2": float(overall_r2),
            "n_test_samples": len(preds),
            "depth_rmse": depth_rmse,
            "depth_r2": depth_r2,
            "depth_levels": config.DEPTH_LEVELS,
            "feature_importance": feature_importance,
            "timestamp": datetime.now().isoformat(),
            "model_params": sum(p.numel() for p in self.model.parameters()),
        }
        
        return self.metrics
    
    def _compute_feature_importance(self) -> Dict[str, float]:
        """
        Compute gradient-based feature importance.
        Measures how much each input feature affects the output via gradient magnitude.
        """
        self.model.eval()
        
        # Use a batch of test data
        X_test_norm = self.scaler_X.transform(self.X_test_raw[:100]).astype(np.float32)
        X_tensor = torch.tensor(X_test_norm, requires_grad=True).to(self.device)
        
        pred = self.model(X_tensor)
        
        # Compute gradients
        pred.sum().backward()
        
        # Feature importance = mean absolute gradient
        grads = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        
        # Normalize to sum to 1
        total = grads.sum() + 1e-10
        importance = {feat: float(grads[i] / total) for i, feat in enumerate(config.FEATURES)}
        
        print(f"\nFeature Importance (gradient-based):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 50)
            print(f"  {feat:>5s}: {imp:.3f} {bar}")
        
        return importance
    
    def save(self, path: str = None):
        """Save model checkpoint and scalers."""
        path = path or os.path.join(config.CHECKPOINT_DIR, "ocean_embed_best.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scaler_X_mean": self.scaler_X.mean_,
            "scaler_X_scale": self.scaler_X.scale_,
            "scaler_y_mean": self.scaler_y.mean_,
            "scaler_y_scale": self.scaler_y.scale_,
            "metrics": self.metrics,
            "training_history": self.training_history,
            "config": config.MODEL_CONFIG,
        }
        
        torch.save(checkpoint, path)
        print(f"\n[SAVE] Model saved to {path}")
        
        # Also save metrics as JSON
        metrics_path = os.path.join(config.SAMPLE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[SAVE] Metrics saved to {metrics_path}")
    
    @classmethod
    def load(cls, path: str = None) -> "OceanEmbedTrainer":
        """Load a saved model checkpoint."""
        path = path or os.path.join(config.CHECKPOINT_DIR, "ocean_embed_best.pt")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        model = create_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        
        trainer = cls(model=model)
        
        # Restore scalers
        trainer.scaler_X.mean_ = checkpoint["scaler_X_mean"]
        trainer.scaler_X.scale_ = checkpoint["scaler_X_scale"]
        trainer.scaler_y.mean_ = checkpoint["scaler_y_mean"]
        trainer.scaler_y.scale_ = checkpoint["scaler_y_scale"]
        
        trainer.metrics = checkpoint.get("metrics", {})
        trainer.training_history = checkpoint.get("training_history", {})
        
        print(f"[LOAD] Model loaded from {path}")
        return trainer


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str = None,
    num_epochs: int = None,
) -> Tuple[OceanEmbedTrainer, Dict]:
    """
    Convenience function: prepare data, train, evaluate, save.
    
    Args:
        X: (n_samples, n_features) surface features
        y: (n_samples, n_depths) temperature profiles
        save_path: path to save checkpoint
        num_epochs: override number of epochs
        
    Returns:
        (trainer, metrics)
    """
    trainer = OceanEmbedTrainer()
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(X, y)
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=num_epochs or config.MODEL_CONFIG["num_epochs"],
    )
    
    # Evaluate on held-out test set
    metrics = trainer.evaluate(test_loader)
    
    # Save
    trainer.save(save_path)
    
    return trainer, metrics


if __name__ == "__main__":
    from scripts.data_ingestion import run_full_ingestion
    
    # Load data
    X, y, status = run_full_ingestion()
    
    # Train
    trainer, metrics = train_model(X, y)
    
    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.3f}°C")
    print(f"Overall R²:   {metrics['overall_r2']:.4f}")
    print(f"Test samples: {metrics['n_test_samples']}")
