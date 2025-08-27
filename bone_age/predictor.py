#!/usr/bin/env python3
"""
🦴 Bone Age Predictor with No Gender Option
==========================================

This version allows predictions without specifying gender.
Save as: predictor_flexible.py

Usage:
    py predictor_flexible.py
"""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
import warnings
import inspect
warnings.filterwarnings('ignore')

# Single source of truth for default relative location
DEFAULT_MODEL_REL = Path("models") / "best_bone_age_model.pth"

def _resolve_model_path(path_str: str | None) -> str:
    # Priority: explicit arg → env var → default
    env = os.getenv("BONEAGER_MODEL")
    raw = path_str or env or str(DEFAULT_MODEL_REL)

    p = Path(raw)
    if p.is_absolute():
        final = str(p)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        final = str((repo_root / p).resolve())

    print(f"    resolved -> {final}")
    return final

_PREDICTOR_CACHE = {}

def get_predictor(model_path: str | None = None, device: str = "auto") -> "FlexibleBoneAgePredictor":
    #Actual code
    resolved = _resolve_model_path(model_path)
    key = (resolved, device)
    if key not in _PREDICTOR_CACHE:
        print(f"Using model: {resolved}")
        _PREDICTOR_CACHE[key] = FlexibleBoneAgePredictor(model_path=resolved, device=device)
    return _PREDICTOR_CACHE[key]

@dataclass
class PredictionResult:
    predicted_age_months: float
    predicted_age_years: float
    confidence_score: float
    uncertainty: float
    gender_used: str  # "Female", "Male", "Unknown", or "Average"

class BoneAgeModel(nn.Module):
    """Same model architecture as training"""
    
    def __init__(self, pretrained=False, dropout=0.3):
        super(BoneAgeModel, self).__init__()
        
        # Use EfficientNet-B0 backbone (same as training)
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Feature processing (EXACT same as training)
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
        
        # Gender embedding (EXACT same as training)
        self.gender_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        
        # Age prediction head (EXACT same as training)
        combined_dim = 128 + 16
        self.age_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, gender):
        # Extract image features
        image_features = self.backbone(x)
        processed_features = self.feature_processor(image_features)
        
        # Process gender
        gender_features = self.gender_embedding(gender.unsqueeze(1))
        
        # Combine features
        combined = torch.cat([processed_features, gender_features], dim=1)
        
        # Predict age
        age_pred = self.age_head(combined)
        
        return {'age': age_pred.squeeze()}

class FlexibleBoneAgePredictor:
    """Bone age predictor that can work with or without gender"""
    
    def __init__(self, model_path, device="auto"):
    # Convert relative path to absolute path if needed
        self.model_path = _resolve_model_path(model_path)
    
        self.preprocessor = self._create_preprocessor()

        self.device = self._get_device(device)
    
        # Debug output
        print(f"🔍 Final model path: {self.model_path}")
        print(f"🔍 Path exists: {os.path.exists(self.model_path)}")
    
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}. "
                               f"Current working directory: {os.getcwd()}")
    
        self._load_model()
        
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _create_preprocessor(self):
        """Create preprocessing pipeline matching training"""
        return A.Compose([
            A.Resize(384, 384),  # Same size as training
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _load_model(self):
        """Load the trained model once, using the resolved path & device."""
        print(f"📂 Loading model from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # 1) Build the architecture
        self.model = BoneAgeModel(pretrained=False)  # pretrained=False for inference
        self.model.to(self.device)

        # 2) Load checkpoint (map to the already-resolved device)
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older torch doesn't support weights_only
            ckpt = torch.load(self.model_path, map_location=self.device)

        # 3) Extract a state_dict no matter how it was saved
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            val_mae = ckpt.get("val_mae", "Unknown")
            print(f"📊 Model MAE: {val_mae}")
        elif isinstance(ckpt, dict):
            # Some training scripts save a raw state_dict
            state_dict = ckpt
        else:
            raise RuntimeError("Unexpected checkpoint format (not a dict).")

        # 4) Normalize keys (strip 'module.' if saved with DataParallel/DDP)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # 5) Load weights
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ Missing keys in state_dict: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys in state_dict: {unexpected}")

        # 6) Finalize
        self.model.eval()
        print("✅ Model loaded successfully!")
        print(f"🔧 Device: {self.device}")
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image"""
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB for preprocessing
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply preprocessing
            transformed = self.preprocessor(image=image)
            return transformed['image'].unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_single_image(self, image_path: str, gender: Union[int, str, None] = None, 
                           use_tta: bool = False) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Predict bone age for a single image with flexible gender options
        
        Args:
            image_path: Path to X-ray image
            gender: Can be:
                - 0 or "female" or "f" for female
                - 1 or "male" or "m" for male  
                - None, "unknown", "both", or "average" for no gender
            use_tta: Whether to use Test Time Augmentation
        
        Returns:
            PredictionResult or List[PredictionResult] if gender is None
        """
        
        # Parse gender input
        parsed_gender = self._parse_gender_input(gender)
        
        if parsed_gender == "both":
            # Predict for both genders and return both results
            return self._predict_both_genders(image_path, use_tta)
        elif parsed_gender == "average":
            # Predict average of both genders
            return self._predict_average_gender(image_path, use_tta)
        else:
            # Single gender prediction
            gender_value = 0 if parsed_gender == "female" else 1
            gender_name = "Female" if parsed_gender == "female" else "Male"
            
            if use_tta:
                result = self._predict_with_tta(image_path, gender_value)
            else:
                result = self._predict_single(image_path, gender_value)
            
            result.gender_used = gender_name
            return result
    
    def _parse_gender_input(self, gender: Union[int, str, None]) -> str:
        """Parse various gender input formats"""
        if gender is None:
            return "both"  # Default to showing both predictions
        
        if isinstance(gender, str):
            gender_lower = gender.lower().strip()
            if gender_lower in ["f", "female", "girl", "woman"]:
                return "female"
            elif gender_lower in ["m", "male", "boy", "man"]:
                return "male"
            elif gender_lower in ["unknown", "none", "both", "?"]:
                return "both"
            elif gender_lower in ["average", "avg", "neutral"]:
                return "average"
            else:
                print(f"⚠️  Unknown gender '{gender}', using both genders")
                return "both"
        
        elif isinstance(gender, int):
            if gender == 0:
                return "female"
            elif gender == 1:
                return "male"
            else:
                print(f"⚠️  Invalid gender {gender}, using both genders")
                return "both"
        
        else:
            return "both"
    
    def _predict_both_genders(self, image_path: str, use_tta: bool) -> List[PredictionResult]:
        """Predict for both genders and return both results"""
        results = []
        
        for gender_value, gender_name in [(0, "Female"), (1, "Male")]:
            if use_tta:
                result = self._predict_with_tta(image_path, gender_value)
            else:
                result = self._predict_single(image_path, gender_value)
            
            result.gender_used = gender_name
            results.append(result)
        
        return results
    
    def _predict_average_gender(self, image_path: str, use_tta: bool) -> PredictionResult:
        """Predict average of both genders"""
        # Get predictions for both genders
        female_result = self._predict_with_tta(image_path, 0) if use_tta else self._predict_single(image_path, 0)
        male_result = self._predict_with_tta(image_path, 1) if use_tta else self._predict_single(image_path, 1)
        
        # Average the predictions
        avg_months = (female_result.predicted_age_months + male_result.predicted_age_months) / 2
        avg_confidence = (female_result.confidence_score + male_result.confidence_score) / 2
        avg_uncertainty = (female_result.uncertainty + male_result.uncertainty) / 2
        
        return PredictionResult(
            predicted_age_months=avg_months,
            predicted_age_years=avg_months / 12.0,
            confidence_score=avg_confidence,
            uncertainty=avg_uncertainty,
            gender_used="Average"
        )
    
    def _predict_single(self, image_path: str, gender: int) -> PredictionResult:
        """Single prediction for specific gender"""
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image_path).to(self.device)
            gender_tensor = torch.tensor([float(gender)], device=self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor, gender_tensor)
            
            age_months = outputs['age'].item()
            
            # Simple confidence based on reasonable age range
            confidence = 1.0
            if age_months < 0 or age_months > 300:  # Outside reasonable range
                confidence = 0.5
            
            uncertainty = 0.0  # Your model doesn't have uncertainty
            
            return PredictionResult(
                predicted_age_months=age_months,
                predicted_age_years=age_months / 12.0,
                confidence_score=confidence,
                uncertainty=uncertainty,
                gender_used=""  # Will be set by caller
            )
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            # Return dummy result
            return PredictionResult(0, 0, 0, 100, "Error")
    
    def _predict_with_tta(self, image_path: str, gender: int) -> PredictionResult:
        """Prediction with Test Time Augmentation"""
        try:
            # Create TTA transforms
            tta_transforms = [
                # Original
                A.Compose([
                    A.Resize(384, 384),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]),
                # Horizontal flip
                A.Compose([
                    A.Resize(384, 384),
                    A.HorizontalFlip(p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]),
                # Slight rotation
                A.Compose([
                    A.Resize(384, 384),
                    A.Rotate(limit=5, p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            ]
            
            predictions = []
            
            # Load base image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            gender_tensor = torch.tensor([float(gender)], device=self.device)
            
            with torch.no_grad():
                for transform in tta_transforms:
                    # Apply transform
                    transformed = transform(image=image)
                    image_tensor = transformed['image'].unsqueeze(0).to(self.device)
                    
                    # Predict
                    outputs = self.model(image_tensor, gender_tensor)
                    predictions.append(outputs['age'].item())
            
            # Aggregate predictions
            mean_age = np.mean(predictions)
            std_age = np.std(predictions)
            
            # Calculate confidence based on consistency
            confidence = max(0, 1 - std_age / 50)
            
            return PredictionResult(
                predicted_age_months=mean_age,
                predicted_age_years=mean_age / 12.0,
                confidence_score=confidence,
                uncertainty=std_age,
                gender_used=""  # Will be set by caller
            )
            
        except Exception as e:
            print(f"❌ Error in TTA prediction: {e}")
            return self._predict_single(image_path, gender)

def main():
    """Interactive prediction interface with flexible gender options"""
    print("🦴 Flexible Bone Age Predictor")
    print("="*40)
    
    # Get model path
    model_path = input("📂 Enter path to your trained model: ").strip().strip('"').strip("'")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        # Initialize predictor
        predictor = FlexibleBoneAgePredictor(model_path)
        
        print(f"\n💡 Gender Options:")
        print(f"  • 0, f, female    = Female")
        print(f"  • 1, m, male      = Male") 
        print(f"  • Enter/unknown   = Show both genders")
        print(f"  • average         = Average of both genders")
        print(f"  • Type 'quit' to exit")
        
        while True:
            print(f"\n" + "-"*40)
            
            # Get image path
            image_path = input("📸 Enter image path (or 'quit'): ").strip().strip('"').strip("'")
            
            if image_path.lower() == 'quit':
                break
            
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                continue
            
            # Get gender (flexible input)
            gender_input = input("👤 Enter gender (0/1/f/m/female/male/unknown/average or Enter for both): ").strip()
            
            if gender_input == "":
                gender_input = None
            
            # Ask about TTA
            use_tta = input("🔄 Use Test Time Augmentation? (y/n): ").strip().lower() == 'y'
            
            # Predict
            print(f"\n🔄 Predicting...")
            try:
                result = predictor.predict_single_image(image_path, gender_input, use_tta)
                
                # Handle different result types
                if isinstance(result, list):
                    # Multiple results (both genders)
                    print(f"\n📊 Results for both genders:")
                    print(f"="*50)
                    
                    for res in result:
                        print(f"\n🔹 {res.gender_used}:")
                        print(f"  🎯 Age: {res.predicted_age_months:.1f} months ({res.predicted_age_years:.1f} years)")
                        print(f"  📈 Confidence: {res.confidence_score:.2f}")
                        if res.uncertainty > 0:
                            print(f"  📊 Uncertainty: ±{res.uncertainty:.1f} months")
                    
                    # Show difference
                    if len(result) == 2:
                        diff = abs(result[0].predicted_age_months - result[1].predicted_age_months)
                        print(f"\n📏 Gender difference: {diff:.1f} months")
                        if diff > 12:
                            print(f"⚠️  Large gender difference detected!")
                        elif diff < 3:
                            print(f"✅ Small gender difference - either prediction likely accurate")
                
                else:
                    # Single result
                    print(f"\n📊 Results ({result.gender_used}):")
                    print(f"  🎯 Predicted Age: {result.predicted_age_months:.1f} months ({result.predicted_age_years:.1f} years)")
                    print(f"  📈 Confidence: {result.confidence_score:.2f}")
                    if result.uncertainty > 0:
                        print(f"  📊 Uncertainty: ±{result.uncertainty:.1f} months")
                
                # Age interpretation
                years = result.predicted_age_years if not isinstance(result, list) else np.mean([r.predicted_age_years for r in result])
                if years < 1:
                    print(f"  👶 Age group: Infant")
                elif years < 3:
                    print(f"  🧒 Age group: Toddler")
                elif years < 13:
                    print(f"  👧👦 Age group: Child")
                elif years < 20:
                    print(f"  🧑 Age group: Adolescent")
                else:
                    print(f"  👨👩 Age group: Adult")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")

def batch_predict_demo():
    """Demo with various gender options"""
    print("🔄 Batch Prediction Demo with Flexible Gender")
    
    model_path = input("📂 Enter model path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Example with different gender specifications
    test_cases = [
        ("image1.jpg", "female"),
        ("image2.jpg", "male"),
        ("image3.jpg", None),      # Will show both
        ("image4.jpg", "average"), # Will show average
    ]
    
    try:
        predictor = FlexibleBoneAgePredictor(model_path)
        
        print(f"\n📊 Batch Results:")
        for i, (path, gender) in enumerate(test_cases):
            print(f"\n{i+1}. {os.path.basename(path)} (Gender: {gender or 'Unknown'})")
            
            if os.path.exists(path):
                result = predictor.predict_single_image(path, gender, use_tta=False)
                
                if isinstance(result, list):
                    for res in result:
                        print(f"   {res.gender_used}: {res.predicted_age_years:.1f} years")
                else:
                    print(f"   {result.gender_used}: {result.predicted_age_years:.1f} years")
            else:
                print(f"   ❌ File not found")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    choice = input("Choose mode:\n1. Interactive prediction\n2. Batch demo\nEnter (1 or 2): ").strip()
    
    if choice == "2":
        batch_predict_demo()
    else:
        main()