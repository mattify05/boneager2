import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
import shutil
from datetime import datetime
import hashlib
import random
from collections import defaultdict
import gc

warnings.filterwarnings('ignore')

@dataclass
class TrainingSession:
    """Track training sessions and used images"""
    session_id: str
    used_images: List[str]
    epoch_start: int
    epoch_end: int
    best_mae: float
    timestamp: str
    stage: str

class SessionManager:
    """Manages training sessions to avoid image repetition"""
    
    def __init__(self, session_file: str = "training_sessions.json"):
        self.session_file = session_file
        self.sessions = []
        self.load_sessions()
    
    def load_sessions(self):
        """Load previous training sessions"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = [TrainingSession(**session) for session in data]
                print(f"📚 Loaded {len(self.sessions)} previous training sessions")
            except Exception as e:
                print(f"⚠️ Error loading sessions: {e}")
                self.sessions = []
        else:
            print("🆕 No previous sessions found, starting fresh")
    
    def save_sessions(self):
        """Save training sessions"""
        data = [
            {
                'session_id': s.session_id,
                'used_images': s.used_images,
                'epoch_start': s.epoch_start,
                'epoch_end': s.epoch_end,
                'best_mae': s.best_mae,
                'timestamp': s.timestamp,
                'stage': s.stage
            }
            for s in self.sessions
        ]
        with open(self.session_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_used_images(self) -> set:
        """Get all previously used images"""
        used = set()
        for session in self.sessions:
            used.update(session.used_images)
        return used
    
    def get_available_images(self, all_images: List[str], stage: str = None) -> List[str]:
        """Get images that haven't been used yet"""
        used = self.get_used_images()
        available = [img for img in all_images if img not in used]
        
        print(f"📊 Image usage:")
        print(f"  Total images: {len(all_images)}")
        print(f"  Previously used: {len(used)}")
        print(f"  Available for training: {len(available)}")
        
        return available
    
    def add_session(self, session: TrainingSession):
        """Add a new training session"""
        self.sessions.append(session)
        self.save_sessions()
    
    def get_next_stage(self) -> str:
        """Determine the next training stage"""
        if not self.sessions:
            return "foundation"
        
        stages = [s.stage for s in self.sessions]
        if "foundation" not in stages:
            return "foundation"
        elif "intermediate" not in stages:
            return "intermediate"
        elif "advanced" not in stages:
            return "advanced"
        else:
            return "refinement"

class CurriculumDataset(Dataset):
    """Dataset with curriculum learning - easier samples first"""
    
    def __init__(self, image_paths: List[str], ages_months: List[float], 
                 genders: List[int], transform=None, is_training=True, stage="foundation"):
        self.image_paths = image_paths
        self.ages_months = ages_months
        self.genders = genders
        self.transform = transform
        self.is_training = is_training
        self.stage = stage
        
        # Sort by curriculum difficulty
        self._sort_by_curriculum()
        
        # Create augmentation pipeline based on stage
        self._create_augmentations()
    
    def _sort_by_curriculum(self):
        """Sort samples by training difficulty"""
        print(f"📖 Applying curriculum learning for stage: {self.stage}")
        
        # Calculate difficulty based on age variance and extremes
        difficulties = []
        mean_age = np.mean(self.ages_months)
        
        for i, age in enumerate(self.ages_months):
            # Difficulty factors:
            # 1. Distance from mean age (extreme ages are harder)
            age_difficulty = abs(age - mean_age) / mean_age
            
            # 2. Very young or very old are harder
            extreme_difficulty = 0
            if age < 24 or age > 180:  # < 2 years or > 15 years
                extreme_difficulty = 0.5
            
            # 3. Add some randomness to avoid overfitting to difficulty metric
            random_factor = np.random.random() * 0.1
            
            total_difficulty = age_difficulty + extreme_difficulty + random_factor
            difficulties.append((total_difficulty, i))
        
        # Sort by difficulty
        difficulties.sort(key=lambda x: x[0])
        
        # Reorder all arrays based on difficulty
        sorted_indices = [idx for _, idx in difficulties]
        self.image_paths = [self.image_paths[i] for i in sorted_indices]
        self.ages_months = [self.ages_months[i] for i in sorted_indices]
        self.genders = [self.genders[i] for i in sorted_indices]
        
        print(f"✅ Sorted {len(self.image_paths)} samples by curriculum difficulty")
    
    def _create_augmentations(self):
        """Create augmentation pipeline based on training stage"""
        if self.is_training:
            if self.stage == "foundation":
                # Gentle augmentations for foundation stage
                self.aug_transform = A.Compose([
                    A.Resize(384, 384),  # Smaller size for faster training
                    A.Rotate(limit=10, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.HorizontalFlip(p=0.3),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
            elif self.stage == "intermediate":
                # Moderate augmentations
                self.aug_transform = A.Compose([
                    A.Resize(448, 448),
                    A.Rotate(limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
            else:  # advanced/refinement
                # Strong augmentations for final stages
                self.aug_transform = A.Compose([
                    A.Resize(512, 512),
                    A.Rotate(limit=20, p=0.6),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.GaussianBlur(blur_limit=5, p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                    A.GridDistortion(p=0.2),
                    A.ElasticTransform(p=0.2),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
        else:
            # Validation - no augmentation, but size based on stage
            size = 384 if self.stage == "foundation" else (448 if self.stage == "intermediate" else 512)
            self.aug_transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image with error handling
        image_path = self.image_paths[idx]
        
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to 3-channel for pretrained models
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply augmentations
            if self.aug_transform:
                augmented = self.aug_transform(image=image)
                image = augmented['image']
            
            # Get labels
            age_months = torch.tensor(self.ages_months[idx], dtype=torch.float32)
            gender = torch.tensor(self.genders[idx], dtype=torch.float32)
            
            return {
                'image': image,
                'age_months': age_months,
                'gender': gender,
                'image_path': image_path
            }
        
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            # Return a dummy sample to prevent crashes
            dummy_image = torch.zeros((3, 384, 384))
            return {
                'image': dummy_image,
                'age_months': torch.tensor(120.0, dtype=torch.float32),  # 10 years
                'gender': torch.tensor(0.0, dtype=torch.float32),
                'image_path': image_path
            }

class ImprovedBoneAgeModel(nn.Module):
    """Enhanced model with better architecture and techniques"""
    
    def __init__(self, backbone='efficientnet_b3', pretrained=True, dropout=0.3, stage="foundation"):
        super(ImprovedBoneAgeModel, self).__init__()
        
        self.stage = stage
        
        # Load backbone with different configurations per stage
        if backbone == 'efficientnet_b3':
            from torchvision.models import efficientnet_b3
            self.backbone = efficientnet_b3(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b0':  # Lighter for foundation
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Adaptive feature dimensions based on stage
        if stage == "foundation":
            hidden_dim = 256
            intermediate_dim = 128
        elif stage == "intermediate":
            hidden_dim = 384
            intermediate_dim = 192
        else:  # advanced/refinement
            hidden_dim = 512
            intermediate_dim = 256
        
        # Feature processing with residual connections
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim)
        )
        
        # Age regression head with attention
        self.age_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(intermediate_dim, intermediate_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim//2),
            nn.Dropout(dropout/3),
            nn.Linear(intermediate_dim//2, 1)
        )
        
        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(intermediate_dim, intermediate_dim//4),
            nn.ReLU(),
            nn.Linear(intermediate_dim//4, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(intermediate_dim, intermediate_dim//4),
            nn.ReLU(),
            nn.Linear(intermediate_dim//4, 1),
            nn.Softplus()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        processed_features = self.feature_processor(features)
        
        # Predictions
        age_pred = self.age_head(processed_features)
        gender_pred = self.gender_head(processed_features)
        uncertainty = self.uncertainty_head(processed_features)
        
        return {
            'age': age_pred.squeeze(),
            'gender': gender_pred.squeeze(),
            'uncertainty': uncertainty.squeeze(),
            'features': processed_features  # For potential transfer learning
        }

class EnhancedTrainer:
    """Enhanced trainer with better training strategies"""
    
    def __init__(self, model, device='cuda', mixed_precision=True):
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.learning_rates = []
        
        # Memory optimization
        torch.backends.cudnn.benchmark = True
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, val_mae: float, 
                       optimizer, scheduler, session_id: str, stage: str, is_best=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': epoch,
            'val_mae': val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
            'learning_rates': self.learning_rates,
            'session_id': session_id,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'mixed_precision': self.mixed_precision
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_path.replace('.pth', '_best.pth')
            shutil.copy2(checkpoint_path, best_path)
            
            # Copy to root for easy access
            root_best_path = f'best_bone_age_model_{stage}.pth'
            shutil.copy2(checkpoint_path, root_best_path)
            print(f"💾 New best {stage} model saved! MAE: {val_mae:.2f} months")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Enhanced checkpoint loading"""
        if os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_maes = checkpoint.get('val_maes', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            
            return checkpoint
        return None
    
    def train_epoch(self, train_loader, optimizer, criterion_age, criterion_gender, 
                   age_weight=1.0, gender_weight=0.3, uncertainty_weight=0.1, 
                   gradient_accumulation_steps=1):
        """Enhanced training epoch with mixed precision and gradient accumulation"""
        self.model.train()
        total_loss = 0
        total_age_loss = 0
        num_batches = 0
        
        # For gradient accumulation
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            ages = batch['age_months'].to(self.device, non_blocking=True)
            genders = batch['gender'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    age_loss = criterion_age(outputs['age'], ages)
                    gender_loss = criterion_gender(outputs['gender'], genders)
                    uncertainty_loss = torch.mean(outputs['uncertainty'])
                    
                    total_loss_batch = (age_weight * age_loss + 
                                      gender_weight * gender_loss + 
                                      uncertainty_weight * uncertainty_loss)
                    
                    # Scale loss for gradient accumulation
                    total_loss_batch = total_loss_batch / gradient_accumulation_steps
            else:
                outputs = self.model(images)
                age_loss = criterion_age(outputs['age'], ages)
                gender_loss = criterion_gender(outputs['gender'], genders)
                uncertainty_loss = torch.mean(outputs['uncertainty'])
                
                total_loss_batch = (age_weight * age_loss + 
                                  gender_weight * gender_loss + 
                                  uncertainty_weight * uncertainty_loss)
                total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(total_loss_batch).backward()
            else:
                total_loss_batch.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += total_loss_batch.item() * gradient_accumulation_steps
            total_age_loss += age_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item() * gradient_accumulation_steps:.4f}',
                'Age MAE': f'{age_loss.item():.2f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(optimizer.param_groups[0]['lr'])
        
        return {
            'total_loss': avg_loss,
            'age_loss': total_age_loss / num_batches
        }
    
    def validate(self, val_loader, criterion_age, criterion_gender):
        """Enhanced validation with memory optimization"""
        self.model.eval()
        total_loss = 0
        age_predictions = []
        age_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device, non_blocking=True)
                ages = batch['age_months'].to(self.device, non_blocking=True)
                genders = batch['gender'].to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                age_loss = criterion_age(outputs['age'], ages)
                gender_loss = criterion_gender(outputs['gender'], genders)
                total_loss += (age_loss + 0.3 * gender_loss).item()
                
                age_predictions.extend(outputs['age'].cpu().numpy())
                age_targets.extend(ages.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        age_mae = mean_absolute_error(age_targets, age_predictions)
        age_r2 = r2_score(age_targets, age_predictions)
        
        self.val_losses.append(avg_loss)
        self.val_maes.append(age_mae)
        
        return {
            'val_loss': avg_loss,
            'age_mae': age_mae,
            'age_r2': age_r2,
            'predictions': age_predictions,
            'targets': age_targets
        }

def create_stage_specific_dataset(all_paths: List[str], all_ages: List[float], 
                                all_genders: List[int], available_paths: List[str], 
                                stage: str, max_samples: int = None) -> Tuple[List[str], List[float], List[int]]:
    """Create dataset specific to training stage"""
    
    # Filter to only available images
    available_set = set(available_paths)
    filtered_data = []
    
    for i, path in enumerate(all_paths):
        if path in available_set:
            filtered_data.append((path, all_ages[i], all_genders[i]))
    
    if not filtered_data:
        return [], [], []
    
    # Limit samples if specified
    if max_samples and len(filtered_data) > max_samples:
        # For foundation stage, use easier samples
        if stage == "foundation":
            # Sort by age (easier to predict middle ages)
            filtered_data.sort(key=lambda x: abs(x[1] - 120))  # Sort by distance from 10 years
        filtered_data = filtered_data[:max_samples]
    
    paths, ages, genders = zip(*filtered_data)
    return list(paths), list(ages), list(genders)

def get_stage_config(stage: str) -> Dict:
    """Get configuration for training stage"""
    configs = {
        "foundation": {
            "epochs": 15,
            "batch_size": 24,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "max_samples": 2000,
            "backbone": "efficientnet_b0",
            "gradient_accumulation": 2
        },
        "intermediate": {
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "weight_decay": 5e-5,
            "max_samples": 3000,
            "backbone": "efficientnet_b3",
            "gradient_accumulation": 2
        },
        "advanced": {
            "epochs": 25,
            "batch_size": 12,
            "learning_rate": 5e-5,
            "weight_decay": 1e-5,
            "max_samples": None,  # Use all available
            "backbone": "efficientnet_b3",
            "gradient_accumulation": 3
        },
        "refinement": {
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "weight_decay": 1e-6,
            "max_samples": None,
            "backbone": "efficientnet_b3",
            "gradient_accumulation": 4
        }
    }
    return configs.get(stage, configs["foundation"])

def train_stage(csv_file: str, stage: str, session_manager: SessionManager, 
               previous_model_path: str = None):
    """Train a specific stage"""
    
    print(f"\n🎯 Starting {stage.upper()} stage training")
    print("="*50)
    
    # Load full dataset
    if not os.path.exists(csv_file):
        print(f"❌ Dataset not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    all_image_paths = df.iloc[:, 0].tolist()
    all_ages_months = df.iloc[:, 1].tolist()
    all_genders = df.iloc[:, 2].tolist()
    
    # Get available images (not used before)
    available_paths = session_manager.get_available_images(all_image_paths, stage)
    
    if len(available_paths) < 50:  # Minimum threshold
        print(f"❌ Not enough unused images for {stage} stage: {len(available_paths)}")
        return None
    
    # Get stage configuration
    config = get_stage_config(stage)
    print(f"📋 Stage config: {config}")
    
    # Create stage-specific dataset
    stage_paths, stage_ages, stage_genders = create_stage_specific_dataset(
        all_image_paths, all_ages_months, all_genders, 
        available_paths, stage, config['max_samples']
    )
    
    print(f"📊 Using {len(stage_paths)} images for {stage} stage")
    
    # Split dataset
    train_paths, val_paths, train_ages, val_ages, train_genders, val_genders = train_test_split(
        stage_paths, stage_ages, stage_genders, 
        test_size=0.15, random_state=None, stratify=stage_genders  # No fixed seed!
    )
    
    print(f"📊 Stage split - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = CurriculumDataset(train_paths, train_ages, train_genders, 
                                    is_training=True, stage=stage)
    val_dataset = CurriculumDataset(val_paths, val_ages, val_genders, 
                                  is_training=False, stage=stage)
    
    # Create dataloaders with optimized settings
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=num_workers, 
                            pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImprovedBoneAgeModel(backbone=config['backbone'], stage=stage)
    
    # Load previous model if available
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"🔄 Loading previous model: {previous_model_path}")
        checkpoint = torch.load(previous_model_path, map_location=device, weights_only=False)
        
        # Try to load compatible weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Loaded compatible weights from previous stage")
        except Exception as e:
            print(f"⚠️ Could not load previous weights: {e}")
            print("🆕 Starting with fresh weights")
    
    # Create trainer
    trainer = EnhancedTrainer(model, device=device, mixed_precision=(device=='cuda'))
    
    # Setup optimizer with different strategies per stage
    if stage == "foundation":
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'], betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], 
                                                steps_per_epoch=len(train_loader), 
                                                epochs=config['epochs'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Loss functions
    criterion_age = nn.SmoothL1Loss()  # More robust than L1Loss
    criterion_gender = nn.BCELoss()
    
    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    patience = 8
    
    session_id = hashlib.md5(f"{stage}_{datetime.now()}".encode()).hexdigest()[:8]
    model_dir = f'models/{stage}_stage'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"🚀 Starting {stage} training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']} ({stage})")
        print("-" * 30)
        
        # Training
        train_metrics = trainer.train_epoch(
            train_loader, optimizer, criterion_age, criterion_gender,
            gradient_accumulation_steps=config['gradient_accumulation']
        )
        
        # Validation
        val_metrics = trainer.validate(val_loader, criterion_age, criterion_gender)
        
        # Learning rate step
        if stage == "foundation":
            scheduler.step()
        else:
            scheduler.step(val_metrics['val_loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val MAE: {val_metrics['age_mae']:.2f} months")
        print(f"Val R²: {val_metrics['age_r2']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint_path = f'{model_dir}/checkpoint_epoch_{epoch+1}.pth'
        is_best = val_metrics['age_mae'] < best_val_mae
        
        trainer.save_checkpoint(checkpoint_path, epoch, val_metrics['age_mae'], 
                              optimizer, scheduler, session_id, stage, is_best)
        
        # Early stopping
        if is_best:
            best_val_mae = val_metrics['age_mae']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏹️ Early stopping after {patience} epochs without improvement")
            break
        
        # Memory cleanup
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Record session
    session = TrainingSession(
        session_id=session_id,
        used_images=stage_paths,  # All images used in this stage
        epoch_start=0,
        epoch_end=epoch,
        best_mae=best_val_mae,
        timestamp=datetime.now().isoformat(),
        stage=stage
    )
    session_manager.add_session(session)
    
    print(f"✅ {stage.upper()} stage completed!")
    print(f"🎯 Best validation MAE: {best_val_mae:.2f} months")
    print(f"📚 Used {len(stage_paths)} images in this stage")
    
    # Return path to best model
    best_models = [f for f in os.listdir(model_dir) if f.endswith('_best.pth')]
    if best_models:
        return os.path.join(model_dir, best_models[0])
    return None

def main():
    """Main multi-stage training pipeline"""
    print("🦴 Enhanced Multi-Stage Bone Age Training System")
    print("="*60)
    print("🎯 Features:")
    print("  • No repeated images across sessions")
    print("  • Curriculum learning (easy → hard)")
    print("  • Progressive model complexity")
    print("  • Laptop-optimized training")
    print("  • Auto-resume capability")
    print()
    
    # Get dataset
    csv_file = input("📄 Enter CSV file path (or press Enter for 'bone_age_dataset.csv'): ").strip()
    if not csv_file:
        csv_file = "bone_age_dataset.csv"
    
    csv_file = csv_file.strip('"').strip("'")
    
    if not os.path.exists(csv_file):
        print(f"❌ Dataset not found: {csv_file}")
        return
    
    print(f"📊 Using dataset: {csv_file}")
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Determine next stage
    next_stage = session_manager.get_next_stage()
    print(f"🎯 Next training stage: {next_stage.upper()}")
    
    # Ask user for confirmation
    proceed = input(f"\n🚀 Start {next_stage} stage training? (y/n): ").lower().strip()
    if proceed != 'y':
        print("👋 Training cancelled")
        return
    
    # Find previous model if not foundation stage
    previous_model = None
    if next_stage != "foundation":
        # Look for best model from previous stages
        stage_order = ["foundation", "intermediate", "advanced", "refinement"]
        current_idx = stage_order.index(next_stage)
        
        for prev_stage in reversed(stage_order[:current_idx]):
            prev_model_path = f'best_bone_age_model_{prev_stage}.pth'
            if os.path.exists(prev_model_path):
                previous_model = prev_model_path
                print(f"🔄 Will transfer from: {previous_model}")
                break
    
    # Train the stage
    best_model_path = train_stage(csv_file, next_stage, session_manager, previous_model)
    
    if best_model_path:
        print(f"\n🎉 {next_stage.upper()} stage training completed!")
        print(f"💾 Best model saved: {best_model_path}")
        
        # Show next steps
        used_images = session_manager.get_used_images()
        df = pd.read_csv(csv_file)
        total_images = len(df)
        remaining_images = total_images - len(used_images)
        
        print(f"\n📊 Training Progress:")
        print(f"  Total images in dataset: {total_images}")
        print(f"  Images used so far: {len(used_images)}")
        print(f"  Remaining images: {remaining_images}")
        print(f"  Progress: {(len(used_images)/total_images)*100:.1f}%")
        
        if remaining_images > 50:
            next_next_stage = session_manager.get_next_stage()
            print(f"\n🔮 Next recommended stage: {next_next_stage.upper()}")
            print("   Run this script again to continue training!")
        else:
            print("\n🏆 All available images have been used!")
            print("   Consider getting more data or adjusting training strategy.")
        
        # Usage example
        print(f"\n💡 To use the trained model:")
        print(f"```python")
        print(f"# Load the model")
        print(f"from bone_age_predictor import BoneAgePredictor")
        print(f"predictor = BoneAgePredictor('{best_model_path}')")
        print(f"")
        print(f"# Predict on new image")
        print(f"result = predictor.predict_single_image('path/to/xray.jpg')")
        print(f"print(f'Predicted age: {{result.predicted_age_months:.1f}} months')")
        print(f"```")
    else:
        print("❌ Training failed or was interrupted")

if __name__ == "__main__":
    main()