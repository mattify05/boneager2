import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class BoneAgeModel(nn.Module):
    def __init__(self, backbone='efficientnet_b3', dropout=0.3):
        super(BoneAgeModel, self).__init__()
        
        if backbone == 'efficientnet_b3':
            from torchvision.models import efficientnet_b3
            self.backbone = efficientnet_b3(weights=None)  # Fixed deprecation warning
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.age_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.gender_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Dropout(dropout/2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        processed_features = self.feature_processor(features)
        
        age_pred = self.age_head(processed_features)
        gender_pred = self.gender_head(processed_features)
        uncertainty = self.uncertainty_head(processed_features)
        
        return {
            'age': age_pred.squeeze(),
            'gender': gender_pred.squeeze(),
            'uncertainty': uncertainty.squeeze()
        }

def load_model(model_path='best_bone_age_model.pth'):
    """Load the trained model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = BoneAgeModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # IMPORTANT: Set to eval mode to fix BatchNorm issue
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model, device

def preprocess_image(image_path):
    """Load and preprocess image"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Preprocessing
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

def predict_bone_age(image_path, model, device, monte_carlo_samples=10):
    """Predict bone age for an image"""
    print(f"🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Preprocess
    image_tensor = preprocess_image(image_path).to(device)
    
    predictions = []
    uncertainties = []
    gender_predictions = []
    
    # FIXED: Use model.eval() consistently and handle BatchNorm properly
    model.eval()
    
    with torch.no_grad():
        for i in range(monte_carlo_samples):
            # For uncertainty estimation, we'll use multiple forward passes
            # but keep the model in eval mode to avoid BatchNorm issues
            output = model(image_tensor)
            
            predictions.append(output['age'].cpu().item())
            uncertainties.append(output['uncertainty'].cpu().item())
            gender_predictions.append(output['gender'].cpu().item())
            
            # Add small noise for uncertainty estimation instead of using dropout
            if i < monte_carlo_samples - 1:  # Don't modify on last iteration
                # Add tiny amount of noise to input for uncertainty estimation
                noise = torch.randn_like(image_tensor) * 0.01
                image_tensor = image_tensor + noise
    
    # Calculate statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions) if len(predictions) > 1 else 0
    uncertainty_mean = np.mean(uncertainties)
    gender_prob = np.mean(gender_predictions)
    
    # Total uncertainty
    total_uncertainty = max(np.sqrt(pred_std**2 + uncertainty_mean**2), 3.0)  # Minimum 3 months uncertainty
    confidence = 1.0 / (1.0 + total_uncertainty / 12.0)
    
    # Age range
    age_range_min = max(0, pred_mean - 1.96 * total_uncertainty)
    age_range_max = pred_mean + 1.96 * total_uncertainty
    
    # Gender
    predicted_gender = "Male" if gender_prob > 0.5 else "Female"
    gender_confidence = max(gender_prob, 1 - gender_prob)
    
    # Determine developmental stage
    if pred_mean < 24:
        stage = "Infant/Toddler"
    elif pred_mean < 72:
        stage = "Early Childhood"
    elif pred_mean < 144:
        stage = "Middle Childhood"
    elif pred_mean < 192:
        stage = "Adolescence"
    else:
        stage = "Young Adult"
    
    # Results
    print(f"📊 BONE AGE PREDICTION")
    # print(f"{'='*50}")
    print(f"🦴 Age: {pred_mean:.1f} months ({pred_mean/12:.1f} years)")
    # print(f"📈 Confidence: {confidence:.1%}")
    # print(f"📏 Range: {age_range_min:.1f} - {age_range_max:.1f} months")
    # print(f"🎯 Uncertainty: ±{total_uncertainty:.1f} months")
    # print(f"👤 Gender: {predicted_gender} ({gender_confidence:.1%})")
    # print(f"🏷️  Stage: {stage}")
    # print(f"{'='*50}")
    
    return {
        'age_months': pred_mean,
        'age_years': pred_mean / 12,
        'confidence': confidence,
        'uncertainty': total_uncertainty,
        'age_range': (age_range_min, age_range_max),
        'gender': predicted_gender,
        'gender_confidence': gender_confidence,
        'stage': stage
    }

def main():
    # Load model
    try:
        model, device = load_model('best_bone_age_model.pth')
        if model is None:
            print("❌ Failed to load model")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Main loop for processing images
    while True:
        # Get image from user
        while True:  # Keep asking until valid image path or user wants to quit
            image_path = input("\nEnter image path (e.g., ./1500.png): ").strip()
            
            if os.path.exists(image_path):
                break  # Valid path found, exit inner loop
            else:
                print(f"❌ Image not found: {image_path}")
                # Ask if user wants to continue trying
                while True:
                    continue_choice = input("Try again? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes']:
                        break  # Break inner input validation loop, continue asking for image
                    elif continue_choice in ['n', 'no']:
                        print("👋 Goodbye!")
                        return  # Exit the entire function
                    else:
                        print("Please enter 'y' or 'n'")
                
                if continue_choice in ['n', 'no']:
                    return  # Exit if user doesn't want to continue
        
        # Make prediction with valid image path
        try:
            result = predict_bone_age(image_path, model, device)
        except Exception as e:
            print(f"❌ Error with model prediction: {e}")
        
        # Ask if user wants to analyze more images
        while True:
            batch = input("\nAnalyze more images? (y/n): ").strip().lower()
            if batch in ['y', 'Y', 'yes', 'Yes', 'YES']:
                break  # Continue outer loop
            elif batch in ['n', 'no', 'NO', 'N']:
                print("👋 Goodbye!")
                return  # Exit the function
            else:
                print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()