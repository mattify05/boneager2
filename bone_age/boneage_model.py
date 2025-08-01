#!/usr/bin/env python3
"""
🦴 Complete Resumable Bone Age Training System
=============================================

Features:
- ✅ Automatic checkpoint saving after every epoch
- ✅ Resume training from any point
- ✅ Safe interruption with Ctrl+C
- ✅ Optimized for laptop training (CPU + 16GB RAM)
- ✅ Progress tracking and history preservation
- ✅ Best model auto-saving
- ✅ Memory efficient processing

Usage:
    # Start new training
    python bone_age_training.py --csv dataset.csv
    
    # Resume training
    python bone_age_training.py --resume --csv dataset.csv
    
    # Check progress
    python bone_age_training.py --status
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import gc
import psutil
import time
import json
from datetime import datetime, timedelta
import warnings
import shutil
warnings.filterwarnings('ignore')

# ============================================================================
# CHECKPOINT MANAGEMENT SYSTEM
# ============================================================================

class CheckpointManager:
    """Advanced checkpoint management for resumable training"""
    
    def __init__(self, checkpoint_dir="checkpoints", project_name="bone_age"):
        self.checkpoint_dir = checkpoint_dir
        self.project_name = project_name
        self.project_dir = os.path.join(checkpoint_dir, project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Files
        self.config_file = os.path.join(self.project_dir, "training_config.json")
        self.progress_file = os.path.join(self.project_dir, "training_progress.json")
        self.best_model_file = os.path.join(self.project_dir, "best_model.pth")
        self.latest_checkpoint_file = os.path.join(self.project_dir, "latest_checkpoint.pth")
        
    def save_config(self, config):
        """Save training configuration"""
        config_copy = config.copy()
        # Convert any non-serializable objects to strings
        for key, value in config_copy.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_copy[key] = str(value)
        
        with open(self.config_file, 'w') as f:
            json.dump(config_copy, f, indent=2)
        print(f"💾 Config saved to: {self.config_file}")
    
    def load_config(self):
        """Load training configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_mae, 
                       train_loss, val_loss, history, total_training_time=0, is_best=False):
        """Save complete training checkpoint"""
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_mae': val_mae,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history,
            'total_training_time': total_training_time,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        # Save regular checkpoint
        checkpoint_file = os.path.join(self.project_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save(checkpoint_data, checkpoint_file)
        
        # Save latest checkpoint (for easy resuming)
        torch.save(checkpoint_data, self.latest_checkpoint_file)
        
        # Save best model separately (lightweight)
        if is_best:
            best_model_data = {
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'total_training_time': total_training_time
            }
            torch.save(best_model_data, self.best_model_file)
            print(f"🏆 New best model saved! MAE: {val_mae:.2f} months")
        
        # Save progress info
        progress = {
            'current_epoch': epoch,
            'best_mae': val_mae if is_best else getattr(self, 'current_best_mae', float('inf')),
            'total_epochs_trained': epoch,
            'last_checkpoint': checkpoint_file,
            'total_training_time': total_training_time,
            'training_start_time': getattr(self, 'training_start_time', datetime.now().isoformat()),
            'last_update': datetime.now().isoformat(),
            'status': 'training'
        }
        
        # Update best MAE if this is the best
        if is_best:
            self.current_best_mae = val_mae
            progress['best_mae'] = val_mae
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"💾 Checkpoint saved: epoch {epoch}, MAE: {val_mae:.2f}")
        
        # Clean up old checkpoints (keep last 5)
        self._cleanup_old_checkpoints(keep_last=5)
        
        return checkpoint_file
    
    def _cleanup_old_checkpoints(self, keep_last=5):
        """Keep only the last N checkpoints to save disk space"""
        checkpoints = glob.glob(os.path.join(self.project_dir, "checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_last:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last]:
                try:
                    os.remove(checkpoint)
                except:
                    pass
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint to resume from"""
        if os.path.exists(self.latest_checkpoint_file):
            return self.latest_checkpoint_file
        
        # Fallback: find highest epoch checkpoint
        checkpoints = glob.glob(os.path.join(self.project_dir, "checkpoint_epoch_*.pth"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            return checkpoints[-1]
        
        return None
    
    def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler=None):
        """Load checkpoint and restore training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return training info
        resume_info = {
            'start_epoch': checkpoint['epoch'] + 1,  # Start from next epoch
            'best_mae': checkpoint['val_mae'],
            'history': checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_mae': []}),
            'last_train_loss': checkpoint.get('train_loss', 0),
            'last_val_loss': checkpoint.get('val_loss', 0),
            'total_training_time': checkpoint.get('total_training_time', 0)
        }
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Resuming from epoch {resume_info['start_epoch']}")
        print(f"🎯 Previous best MAE: {resume_info['best_mae']:.2f} months")
        
        # Update our tracking
        self.current_best_mae = checkpoint.get('val_mae', float('inf'))
        self.training_start_time = checkpoint.get('timestamp', datetime.now().isoformat())
        
        return resume_info
    
    def get_training_progress(self):
        """Get current training progress information"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None
    
    def should_resume(self):
        """Check if there's a training session to resume"""
        latest_checkpoint = self.find_latest_checkpoint()
        progress = self.get_training_progress()
        
        if latest_checkpoint and progress:
            print(f"\n🔍 Found existing training session:")
            print(f"  📊 Last epoch: {progress.get('current_epoch', 'Unknown')}")
            print(f"  🎯 Best MAE: {progress.get('best_mae', 'Unknown'):.2f} months")
            
            # Calculate training time
            total_time = progress.get('total_training_time', 0)
            if total_time > 0:
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                print(f"  ⏱️  Training time: {hours}h {minutes}m")
            
            print(f"  📅 Last update: {progress.get('last_update', 'Unknown')[:19]}")
            print(f"  💾 Checkpoint: {os.path.basename(latest_checkpoint)}")
            
            return True
        return False
    
    def load_existing_model(self, model_path, model):
        """Load weights from an existing model (like best_bone_age_model.pth)"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Existing model not found: {model_path}")
        
        print(f"📂 Loading existing model: {model_path}")
        
        # Load the existing model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Try to extract model state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            existing_mae = checkpoint.get('val_mae', 'Unknown')
            existing_epoch = checkpoint.get('epoch', 'Unknown')
        else:
            # Assume the whole file is the state dict
            model_state = checkpoint
            existing_mae = 'Unknown'
            existing_epoch = 'Unknown'
        
        # Load weights with flexibility for architecture differences
        try:
            model.load_state_dict(model_state, strict=True)
            print(f"✅ Loaded existing model weights perfectly!")
        except RuntimeError as e:
            print(f"⚠️  Model architecture mismatch. Trying flexible loading...")
            # Load compatible weights only
            model_dict = model.state_dict()
            compatible_weights = {}
            
            for name, param in model_state.items():
                if name in model_dict and model_dict[name].shape == param.shape:
                    compatible_weights[name] = param
                else:
                    print(f"  ⚠️  Skipping incompatible layer: {name}")
            
            model_dict.update(compatible_weights)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(compatible_weights)}/{len(model_state)} compatible layers")
        
        print(f"📊 Existing model info:")
        print(f"  🎯 Previous MAE: {existing_mae}")
        print(f"  📅 Previous epoch: {existing_epoch}")
        print(f"  🔄 Continuing training from this model...")
        
        return {
            'previous_mae': existing_mae,
            'previous_epoch': existing_epoch,
            'loaded_successfully': True
        }
    
    def mark_training_completed(self, final_mae, total_time):
        """Mark training as completed"""
        progress = self.get_training_progress() or {}
        progress.update({
            'status': 'completed',
            'final_mae': final_mae,
            'total_training_time': total_time,
            'completion_time': datetime.now().isoformat()
        })
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

# ============================================================================
# PREPROCESSING SYSTEM
# ============================================================================

class LaptopOptimizedPreprocessor:
    """Preprocessing optimized for laptop training - faster but still effective"""
    
    def __init__(self, image_size=384):
        self.image_size = image_size
        
        # Training augmentations - optimized for speed
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            
            # Essential augmentations for bone age
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.7),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
            A.HorizontalFlip(p=0.5),  # Hands can be mirrored
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            
            # Light geometric augmentations
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=8, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transform (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str, is_training: bool = True):
        """Fast preprocessing for laptop training"""
        try:
            # Load image efficiently
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if is_training:
                transformed = self.train_transform(image=image)
            else:
                transformed = self.val_transform(image=image)
            
            return transformed['image']
        
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            # Return dummy image to prevent crashes
            return torch.zeros((3, self.image_size, self.image_size))

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LightweightBoneAgeModel(nn.Module):
    """Laptop-friendly model - faster training, still accurate"""
    
    def __init__(self, backbone='efficientnet_b0', pretrained=True, dropout=0.3):
        super(LightweightBoneAgeModel, self).__init__()
        
        # Use smaller, faster backbone for laptop training
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128)
        )
        
        # Gender embedding (important for bone age)
        self.gender_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        
        # Combined features
        combined_dim = 128 + 16  # image features + gender embedding
        
        # Age regression head
        self.age_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, gender):
        # Extract image features
        image_features = self.backbone(x)
        processed_features = self.feature_processor(image_features)
        
        # Process gender information
        gender_features = self.gender_embedding(gender.unsqueeze(1))
        
        # Combine image and gender features
        combined = torch.cat([processed_features, gender_features], dim=1)
        
        # Predict age
        age_pred = self.age_head(combined)
        
        return {'age': age_pred.squeeze()}

# ============================================================================
# DATASET CLASS
# ============================================================================

class BoneAgeDataset(Dataset):
    """Memory-efficient dataset for bone age training"""
    
    def __init__(self, image_paths, ages_months, genders, preprocessor, is_training=True):
        self.image_paths = image_paths
        self.ages_months = ages_months
        self.genders = genders
        self.preprocessor = preprocessor
        self.is_training = is_training
        
        print(f"📊 Dataset created: {len(self.image_paths)} images")
        print(f"   Training mode: {is_training}")
        print(f"   Age range: {min(ages_months):.1f} - {max(ages_months):.1f} months")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = self.preprocessor.preprocess_image(self.image_paths[idx], self.is_training)
        
        return {
            'image': image,
            'age_months': torch.tensor(self.ages_months[idx], dtype=torch.float32),
            'gender': torch.tensor(self.genders[idx], dtype=torch.float32),
            'image_path': self.image_paths[idx]
        }

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ResumableTrainer:
    """Trainer with full checkpoint resuming capabilities"""
    
    def __init__(self, checkpoint_manager, device='cpu'):
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.process = psutil.Process(os.getpid())
        
    def monitor_resources(self):
        """Monitor system resources"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = psutil.virtual_memory().percent
        
        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'memory_percent': memory_percent
        }
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train single epoch with progress tracking"""
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                ages = batch['age_months'].to(self.device)
                genders = batch['gender'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images, genders)
                loss = criterion(outputs['age'], ages)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}'
                })
                
                # Memory cleanup every 50 batches
                if batch_idx % 50 == 0:
                    gc.collect()
                    
                # Resource monitoring every 100 batches
                if batch_idx % 100 == 0:
                    resources = self.monitor_resources()
                    if resources['memory_percent'] > 85:
                        print(f"\n⚠️  High memory usage: {resources['memory_percent']:.1f}%")
                        gc.collect()
            
            except Exception as e:
                print(f"\n❌ Error in batch {batch_idx}: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, epoch_time
    
    def validate(self, model, val_loader, criterion):
        """Fast validation"""
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    images = batch['image'].to(self.device)
                    ages = batch['age_months'].to(self.device)
                    genders = batch['gender'].to(self.device)
                    
                    outputs = model(images, genders)
                    loss = criterion(outputs['age'], ages)
                    
                    total_loss += loss.item()
                    all_predictions.extend(outputs['age'].cpu().numpy())
                    all_targets.extend(ages.cpu().numpy())
                
                except Exception as e:
                    print(f"❌ Error in validation batch: {e}")
                    continue
        
        if len(all_predictions) == 0:
            return {'val_loss': float('inf'), 'mae': float('inf'), 'r2': 0.0}
        
        avg_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return {
            'val_loss': avg_loss,
            'mae': mae,
            'r2': r2,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def run_training(self, model, train_loader, val_loader, config, resume_from=None):
        """Main training loop with full resuming capability"""
        
        print(f"\n🚀 Initializing training...")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
        criterion = nn.SmoothL1Loss()  # Robust loss function
        
        # Initialize training state
        start_epoch = 0
        best_mae = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'epoch_times': []}
        total_training_time = 0
        
        # Resume from checkpoint if provided
        if resume_from:
            resume_info = self.checkpoint_manager.load_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            start_epoch = resume_info['start_epoch']
            best_mae = resume_info['best_mae']
            history = resume_info['history']
            total_training_time = resume_info.get('total_training_time', 0)
            print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            print(f"🆕 Starting fresh training")
        
        print(f"\n📋 Training Configuration:")
        print(f"  🎯 Target epochs: {config['epochs']}")
        print(f"  📏 Image size: {config['image_size']}")
        print(f"  📦 Batch size: {config['batch_size']}")
        print(f"  🎓 Learning rate: {config['learning_rate']}")
        print(f"  ⏳ Early stopping patience: {config['patience']}")
        
        # Estimate training time
        if not resume_from:
            print(f"\n⏱️  Time Estimates (per epoch):")
            print(f"  📊 Training: 25-45 minutes")
            print(f"  📊 Validation: 3-5 minutes")
            print(f"  📊 Total per epoch: ~30-50 minutes")
            print(f"  📊 Full training: ~{config['epochs'] * 0.5:.1f}-{config['epochs'] * 0.8:.1f} hours")
        
        print(f"\n💡 Remember: You can safely stop with Ctrl+C anytime!")
        print(f"💡 Resume later with: python {__file__} --resume --csv <your_csv>")
        print("="*60)
        
        # Training loop
        try:
            for epoch in range(start_epoch, config['epochs']):
                epoch_start_time = time.time()
                
                print(f"\n🔄 Epoch {epoch + 1}/{config['epochs']}")
                print("-" * 50)
                
                # Train
                train_loss, train_time = self.train_epoch(
                    model, train_loader, optimizer, criterion, epoch + 1
                )
                
                # Validate
                val_metrics = self.validate(model, val_loader, criterion)
                
                # Update history
                epoch_total_time = time.time() - epoch_start_time
                total_training_time += epoch_total_time
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_mae'].append(val_metrics['mae'])
                history['epoch_times'].append(epoch_total_time)
                
                # Step scheduler
                scheduler.step()
                
                # Check if best model
                is_best = val_metrics['mae'] < best_mae
                if is_best:
                    best_mae = val_metrics['mae']
                    patience_counter = 0
                    print(f"🎉 New best model!")
                else:
                    patience_counter += 1
                
                # Print epoch results
                print(f"\n📊 Epoch {epoch + 1} Results:")
                print(f"  🔸 Train Loss: {train_loss:.4f}")
                print(f"  🔸 Val MAE: {val_metrics['mae']:.2f} months ({val_metrics['mae']/12:.2f} years)")
                print(f"  🔸 Val R²: {val_metrics['r2']:.4f}")
                print(f"  🔸 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"  🔸 Epoch Time: {epoch_total_time/60:.1f} minutes")
                print(f"  🔸 Total Time: {total_training_time/3600:.1f} hours")
                
                # Estimate remaining time
                if epoch > start_epoch:
                    avg_epoch_time = total_training_time / (epoch + 1 - start_epoch)
                    remaining_epochs = config['epochs'] - (epoch + 1)
                    estimated_remaining = (remaining_epochs * avg_epoch_time) / 3600
                    print(f"  🔸 Est. Remaining: {estimated_remaining:.1f} hours")
                
                # Resource monitoring
                resources = self.monitor_resources()
                print(f"  🔸 System: CPU {resources['cpu_percent']:.1f}%, RAM {resources['memory_percent']:.1f}%")
                
                # Save checkpoint after every epoch
                self.checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics['mae'],
                    train_loss, val_metrics['val_loss'], history, 
                    total_training_time, is_best
                )
                
                print(f"💾 Progress saved. Safe to stop and resume anytime!")
                
                # Early stopping check
                if patience_counter >= config['patience']:
                    print(f"\n⏹️  Early stopping triggered!")
                    print(f"📊 No improvement for {config['patience']} epochs")
                    print(f"🏆 Best MAE: {best_mae:.2f} months")
                    break
                
                # Memory cleanup
                gc.collect()
        
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Training interrupted by user at epoch {epoch + 1}")
            print(f"💾 Progress saved successfully!")
            print(f"🔄 Resume anytime with: python {__file__} --resume --csv <your_csv>")
            print(f"🏆 Current best MAE: {best_mae:.2f} months")
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print(f"💾 Progress saved. You can resume from the last checkpoint.")
            print(f"🏆 Current best MAE: {best_mae:.2f} months")
        
        # Mark training as completed
        self.checkpoint_manager.mark_training_completed(best_mae, total_training_time)
        
        print(f"\n🎉 Training session completed!")
        print(f"🏆 Best MAE achieved: {best_mae:.2f} months ({best_mae/12:.2f} years)")
        print(f"⏱️  Total training time: {total_training_time/3600:.1f} hours")
        
        # Show comparison to benchmarks
        print(f"\n📊 Performance Comparison:")
        if best_mae < 6:
            print(f"  🥇 Excellent! Near RSNA challenge winner level (4.2-4.5 months)")
        elif best_mae < 8:
            print(f"  🥈 Very good! Competitive performance")
        elif best_mae < 12:
            print(f"  🥉 Good! Solid results for laptop training")
        else:
            print(f"  📈 Room for improvement - consider longer training or more data")
        
        return best_mae

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_training_status(project_name="bone_age"):
    """Show current training status"""
    checkpoint_manager = CheckpointManager(project_name=project_name)
    progress = checkpoint_manager.get_training_progress()
    
    if not progress:
        print("❌ No training sessions found")
        return
    
    print(f"📊 Training Status for Project: {project_name}")
    print("="*50)
    
    # Basic info
    print(f"Status: {progress.get('status', 'Unknown').upper()}")
    print(f"Current Epoch: {progress.get('current_epoch', 'Unknown')}")
    print(f"Best MAE: {progress.get('best_mae', 'Unknown'):.2f} months")
    
    # Time info
    total_time = progress.get('total_training_time', 0)
    if total_time > 0:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        print(f"Training Time: {hours}h {minutes}m")
    
    print(f"Last Update: {progress.get('last_update', 'Unknown')[:19]}")
    
    # Show available checkpoints
    checkpoints_dir = f"checkpoints/{project_name}"
    if os.path.exists(checkpoints_dir):
        checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint_epoch_*.pth"))
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"\nAvailable Checkpoints: {len(checkpoints)}")
        for checkpoint in checkpoints[-3:]:  # Show last 3
            epoch = int(checkpoint.split('_')[-1].split('.')[0])
            print(f"  📁 Epoch {epoch:3d}: {os.path.basename(checkpoint)}")
        
        # Best model
        best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"  🏆 Best model: best_model.pth")

def create_sample_dataset(csv_path, sample_size=1000, output_path="sample_dataset.csv"):
    """Create a smaller sample dataset for testing"""
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📊 Original dataset: {len(df)} images")
    
    if len(df) <= sample_size:
        print(f"📊 Dataset is already small enough ({len(df)} <= {sample_size})")
        return
    
    # Stratified sampling by gender and age groups
    df['age_group'] = pd.cut(df.iloc[:, 1], bins=5, labels=False)
    sample_df = df.groupby(['age_group', df.iloc[:, 2]], group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 10))
    ).reset_index(drop=True)
    
    # Ensure we don't exceed sample_size
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=42)
    
    # Remove helper column
    sample_df = sample_df.drop('age_group', axis=1)
    
    sample_df.to_csv(output_path, index=False)
    print(f"✅ Sample dataset created: {output_path}")
    print(f"📊 Sample size: {len(sample_df)} images")
    print(f"📈 Age range: {sample_df.iloc[:, 1].min():.1f} - {sample_df.iloc[:, 1].max():.1f} months")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training interface with full functionality"""
    parser = argparse.ArgumentParser(
        description='🦴 Complete Resumable Bone Age Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start new training
    python bone_age_training.py --csv dataset.csv
    
    # Continue training your existing model
    python bone_age_training.py --csv dataset.csv --existing_model best_bone_age_model.pth
    
    # Resume training
    python bone_age_training.py --resume --csv dataset.csv
    
    # Check training status
    python bone_age_training.py --status
    
    # Create sample dataset for testing
    python bone_age_training.py --sample dataset.csv --size 1000
        """
    )
    
    parser.add_argument('--csv', type=str, help='Path to CSV dataset file')
    parser.add_argument('--existing_model', type=str, help='Path to existing model to continue training (e.g., best_bone_age_model.pth)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--status', action='store_true', help='Show training status')
    parser.add_argument('--sample', type=str, help='Create sample dataset from CSV')
    parser.add_argument('--size', type=int, default=1000, help='Sample size for --sample')
    parser.add_argument('--project', type=str, default='bone_age', help='Project name')
    parser.add_argument('--config', type=str, help='Custom config JSON file')
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.status:
        show_training_status(args.project)
        return
    
    if args.sample:
        create_sample_dataset(args.sample, args.size)
        return
    
    # Main training interface
    print("🦴 Complete Resumable Bone Age Training System")
    print("="*60)
    print("✨ Features:")
    print("  • Automatic checkpoint saving after every epoch")
    print("  • Resume training from any point")
    print("  • Safe interruption with Ctrl+C")
    print("  • Optimized for laptop training (CPU + 16GB RAM)")
    print("  • Progress tracking and history preservation")
    print("  • Best model auto-saving")
    print("  • Memory efficient processing")
    print()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(project_name=args.project)
    
    # Check for existing training session
    latest_checkpoint = None
    if checkpoint_manager.should_resume() or args.resume:
        if args.resume:
            latest_checkpoint = checkpoint_manager.find_latest_checkpoint()
            if latest_checkpoint:
                print(f"✅ Will resume from: {os.path.basename(latest_checkpoint)}")
            else:
                print("❌ No checkpoint found to resume from")
                return
        else:
            resume_choice = input("🔄 Resume from latest checkpoint? (y/n): ").strip().lower()
            if resume_choice == 'y':
                latest_checkpoint = checkpoint_manager.find_latest_checkpoint()
                print(f"✅ Will resume from: {os.path.basename(latest_checkpoint)}")
            else:
                print("🆕 Starting fresh training")
    
    # Get dataset
    if args.csv:
        csv_file = args.csv
    else:
        csv_file = input("📄 Enter CSV file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(csv_file):
        print(f"❌ Dataset not found: {csv_file}")
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded config from: {args.config}")
    else:
        # Default laptop-optimized configuration
        config = {
            'image_size': 384,          # Smaller for faster training
            'batch_size': 8,            # Small batch size for 16GB RAM
            'epochs': 25,               # Reasonable number of epochs
            'learning_rate': 1e-3,      # Good starting learning rate
            'weight_decay': 1e-4,       # Regularization
            'patience': 6,              # Early stopping patience
            'backbone': 'efficientnet_b0',  # Fastest EfficientNet
            'num_workers': 2,           # Conservative for laptop CPU
            'test_size': 0.15           # Validation split
        }
    
    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Save configuration
    checkpoint_manager.save_config(config)
    
    # Load and analyze dataset
    df = pd.read_csv(csv_file)
    print(f"\n📊 Dataset Analysis:")
    print(f"  📁 Total images: {len(df)}")
    print(f"  📈 Age range: {df.iloc[:, 1].min():.1f} - {df.iloc[:, 1].max():.1f} months")
    print(f"  👥 Gender distribution: {df.iloc[:, 2].value_counts().to_dict()}")
    
    # Handle large datasets
    if len(df) > 10000:
        print(f"\n⚠️  Large dataset detected ({len(df)} images)")
        print(f"💡 For laptop training, consider using a sample first:")
        print(f"   python {__file__} --sample {csv_file} --size 4000")
        
        use_full = input("Use full dataset? (y/n, n for sample): ").strip().lower()
        if use_full != 'y':
            sample_size = min(4000, len(df))
            df = df.sample(n=sample_size, random_state=42)
            print(f"📉 Using sample of {len(df)} images")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, 
        test_size=config.get('test_size', 0.15), 
        random_state=42, 
        stratify=df.iloc[:, 2]  # Stratify by gender
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"  🎓 Training: {len(train_df)} images")
    print(f"  ✅ Validation: {len(val_df)} images")
    
    # Create preprocessor and datasets
    preprocessor = LaptopOptimizedPreprocessor(image_size=config['image_size'])
    
    train_dataset = BoneAgeDataset(
        train_df.iloc[:, 0].tolist(),
        train_df.iloc[:, 1].tolist(),
        train_df.iloc[:, 2].tolist(),
        preprocessor,
        is_training=True
    )
    
    val_dataset = BoneAgeDataset(
        val_df.iloc[:, 0].tolist(),
        val_df.iloc[:, 1].tolist(),
        val_df.iloc[:, 2].tolist(),
        preprocessor,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=False,  # Disable for CPU training
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=False
    )
    
    print(f"📦 Data loaders created:")
    print(f"  🎓 Train batches: {len(train_loader)}")
    print(f"  ✅ Val batches: {len(val_loader)}")
    
    # Create model and trainer
    device = 'cpu'  # Force CPU for laptop training
    model = LightweightBoneAgeModel(
        backbone=config.get('backbone', 'efficientnet_b0'),
        pretrained=True,
        dropout=0.3
    )
    
    # Load existing model if provided
    existing_model_info = None
    if args.existing_model and not latest_checkpoint:  # Only if not resuming from checkpoint
        try:
            existing_model_info = checkpoint_manager.load_existing_model(args.existing_model, model)
            print(f"🎯 Will continue training from your existing model")
            print(f"📈 Goal: Improve beyond {existing_model_info['previous_mae']} months MAE")
        except Exception as e:
            print(f"❌ Error loading existing model: {e}")
            print(f"💡 Starting fresh training instead...")
            existing_model_info = None
    
    trainer = ResumableTrainer(checkpoint_manager, device)
    
    # Final confirmation
    if not latest_checkpoint:
        print(f"\n🚨 FINAL CONFIRMATION:")
        print(f"  📊 Training {len(df)} images for up to {config['epochs']} epochs")
        print(f"  ⏱️  Estimated time: {config['epochs'] * 0.5:.1f}-{config['epochs'] * 0.8:.1f} hours")
        print(f"  💾 Checkpoints will be saved to: checkpoints/{args.project}/")
        if existing_model_info:
            print(f"  🔄 Continuing from existing model (Previous MAE: {existing_model_info['previous_mae']})")
        else:
            print(f"  🆕 Starting fresh training")
        print(f"  🛑 You can safely stop anytime with Ctrl+C")
        
        proceed = input("\n🚀 Start training? (y/n): ").strip().lower()
        if proceed != 'y':
            print("👋 Training cancelled")
            return
    
    # Run training
    try:
        best_mae = trainer.run_training(
            model, train_loader, val_loader, config, latest_checkpoint
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Final best MAE: {best_mae:.2f} months")
        print(f"💾 Best model saved in: checkpoints/{args.project}/best_model.pth")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print(f"💾 Check checkpoints/{args.project}/ for saved progress")

if __name__ == "__main__":
    main()