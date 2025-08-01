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
            self.backbone = efficientnet_b3(weights=None)
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
        uncertainty = self.uncertainty_head(processed_features)
        
        return {
            'age': age_pred.squeeze(),
            'uncertainty': uncertainty.squeeze()
        }

def load_model(model_path='best_bone_age_model.pth'):
    """Load the trained model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = BoneAgeModel()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle models that were trained with gender head
    model_state = checkpoint['model_state_dict']
    filtered_state = {k: v for k, v in model_state.items() if not k.startswith('gender_head')}
    
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    
    # Set to eval mode to fix BatchNorm issue
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
    """
    Predict bone age for an image
    
    Args:
        image_path: Path to the X-ray image
        model: Trained bone age model
        device: Computing device (cuda/cpu)
        monte_carlo_samples: Number of prediction samples for uncertainty
    """
    print(f"🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Preprocess
    image_tensor = preprocess_image(image_path).to(device)
    
    predictions = []
    uncertainties = []
    
    model.eval()
    
    with torch.no_grad():
        for i in range(monte_carlo_samples):
            # Multiple forward passes for uncertainty estimation
            output = model(image_tensor)
            
            predictions.append(output['age'].cpu().item())
            uncertainties.append(output['uncertainty'].cpu().item())
            
            # Add small noise for uncertainty estimation instead of using dropout
            if i < monte_carlo_samples - 1:  # Don't modify on last iteration
                # Add tiny amount of noise to input for uncertainty estimation
                noise = torch.randn_like(image_tensor) * 0.01
                image_tensor = image_tensor + noise
    
    # Calculate statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions) if len(predictions) > 1 else 0
    uncertainty_mean = np.mean(uncertainties)
    
    # Total uncertainty
    total_uncertainty = max(np.sqrt(pred_std**2 + uncertainty_mean**2), 3.0)  # Minimum 3 months uncertainty
    confidence = min(1.0 / (1.0 + total_uncertainty / 12.0), 0.95)  # Cap at 95%
    
    # Age range (95% confidence interval)
    age_range_min = max(0, pred_mean - 1.96 * total_uncertainty)
    age_range_max = pred_mean + 1.96 * total_uncertainty
    
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
    print(f"{'='*50}")
    print(f"🦴 Age: {pred_mean:.1f} months ({pred_mean/12:.1f} years)")
    print(f"📈 Confidence: {confidence:.1%}")
    print(f"📏 Range: {age_range_min:.1f} - {age_range_max:.1f} months")
    print(f"🎯 Uncertainty: ±{total_uncertainty:.1f} months")
    print(f"🏷️  Stage: {stage}")
    print(f"{'='*50}")
    
    return {
        'age_months': pred_mean,
        'age_years': pred_mean / 12,
        'confidence': confidence,
        'uncertainty': total_uncertainty,
        'age_range': (age_range_min, age_range_max),
        'stage': stage
    }

def get_user_inputs():
    """Get user input for image path"""
    while True:
        image_path = input("\nEnter image path (e.g., ./1500.png): ").strip()
        
        if os.path.exists(image_path):
            return image_path
        else:
            print(f"❌ Image not found: {image_path}")
            while True:
                continue_choice = input("Try again? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    break
                elif continue_choice in ['n', 'no']:
                    return None
                else:
                    print("Please enter 'y' or 'n'")
            
            if continue_choice in ['n', 'no']:
                return None

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
    
    print("🔬 Bone Age Predictor")
    print("=" * 50)
    
    # Main loop for processing images
    while True:
        # Get user input
        image_path = get_user_inputs()
        
        if image_path is None:  # User chose to quit
            print("👋 Goodbye!")
            return
        
        # Make prediction
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