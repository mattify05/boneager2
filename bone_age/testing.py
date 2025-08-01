#!/usr/bin/env python3
"""
Batch Bone Age Predictor for FlexibleBoneAgePredictor
Works with your existing predictor.py file
"""

import os
import csv
import glob
from datetime import datetime

# Import your predictor class
try:
    from predictor import FlexibleBoneAgePredictor
    print("✅ Successfully imported FlexibleBoneAgePredictor")
except ImportError as e:
    print(f"❌ Error importing from predictor.py: {e}")
    print("Make sure predictor.py is in the same directory")
    exit(1)

def get_image_files(dataset_dir):
    """Get all image files from the dataset directory"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(dataset_dir, ext)
        image_files.extend(glob.glob(pattern))
        # Also check uppercase
        pattern = os.path.join(dataset_dir, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    # Extract filenames without extension
    image_names = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        image_name = os.path.splitext(filename)[0]
        image_names.append((image_name, img_path))
    
    # Remove duplicates and sort
    unique_images = list(set(image_names))
    return sorted(unique_images)

def get_development_stage(age_years):
    """Determine development stage based on age"""
    if age_years < 1:
        return "Infant"
    elif age_years < 3:
        return "Toddler"
    elif age_years < 13:
        return "Child"
    elif age_years < 20:
        return "Adolescent"
    else:
        return "Adult"

def batch_predict(dataset_dir, model_path, output_file=None, gender_option=None, use_tta=False):
    """
    Process all images using FlexibleBoneAgePredictor
    
    Args:
        dataset_dir: Directory containing images
        model_path: Path to trained model
        output_file: Output CSV file (optional)
        gender_option: "female", "male", "average", "both", or None
        use_tta: Whether to use Test Time Augmentation
    """
    
    print(f"🦴 Batch Bone Age Prediction")
    print("="*60)
    
    # Initialize predictor
    try:
        print(f"🔄 Initializing predictor...")
        predictor = FlexibleBoneAgePredictor(model_path)
        print(f"✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    # Get image files
    image_files = get_image_files(dataset_dir)
    if not image_files:
        print(f"❌ No images found in {dataset_dir}")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF")
        return
    
    # Prepare output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bone_age_results_{timestamp}.csv"
    
    print(f"\n🚀 Processing {len(image_files)} images")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"👤 Gender option: {gender_option or 'Both (average)'}")
    print(f"🔄 Using TTA: {use_tta}")
    print(f"💾 Output: {output_file}")
    print("="*60)
    
    # Process images
    results = []
    successful = 0
    failed = 0
    
    for i, (image_name, image_path) in enumerate(image_files, 1):
        print(f"[{i:4d}/{len(image_files):4d}] {image_name}", end=" ")
        
        try:
            # Make prediction
            result = predictor.predict_single_image(image_path, gender_option, use_tta)
            
            # Handle different result types
            if isinstance(result, list):
                # Multiple results (both genders)
                # Calculate average
                avg_months = sum(r.predicted_age_months for r in result) / len(result)
                avg_confidence = sum(r.confidence_score for r in result) / len(result)
                avg_uncertainty = sum(r.uncertainty for r in result) / len(result)
                gender_used = "Both"
                
                # Create individual results string
                individual_results = []
                for r in result:
                    individual_results.append(f"{r.gender_used}:{r.predicted_age_months:.1f}")
                individual_str = ", ".join(individual_results)
                
                # Use average for main result
                final_months = avg_months
                final_confidence = avg_confidence
                final_uncertainty = avg_uncertainty
                
            else:
                # Single result
                final_months = result.predicted_age_months
                final_confidence = result.confidence_score
                final_uncertainty = result.uncertainty
                gender_used = result.gender_used
                individual_str = f"{gender_used}:{final_months:.1f}"
            
            # Calculate derived values
            final_years = final_months / 12.0
            stage = get_development_stage(final_years)
            
            # Store result
            results.append({
                'image_name': image_name,
                'predicted_age_months': round(final_months, 1),
                'predicted_age_years': round(final_years, 1),
                'confidence': round(final_confidence, 3),
                'uncertainty_months': round(final_uncertainty, 1),
                'development_stage': stage,
                'gender_used': gender_used,
                'individual_predictions': individual_str
            })
            
            successful += 1
            print(f"✅ {final_months:.1f}m ({final_years:.1f}y) [{gender_used}]")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'image_name': image_name,
                'predicted_age_months': 'ERROR',
                'predicted_age_years': 'ERROR',
                'confidence': 'ERROR',
                'uncertainty_months': 'ERROR',
                'development_stage': 'ERROR',
                'gender_used': 'ERROR',
                'individual_predictions': 'ERROR'
            })
            failed += 1
        
        # Progress update every 25 images
        if i % 25 == 0 or i == len(image_files):
            print(f"   Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
    
    # Save results
    try:
        # Save detailed CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_name', 'predicted_age_months', 'predicted_age_years',
                'confidence', 'uncertainty_months', 'development_stage', 
                'gender_used', 'individual_predictions'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # Save simple text file
        simple_output = output_file.replace('.csv', '_simple.txt')
        with open(simple_output, 'w', encoding='utf-8') as f:
            f.write("Image_Name\tAge_Months\tAge_Years\tGender_Used\n")
            for result in results:
                if result['predicted_age_months'] != 'ERROR':
                    f.write(f"{result['image_name']}\t{result['predicted_age_months']}\t{result['predicted_age_years']}\t{result['gender_used']}\n")
        
        # Results summary
        print("="*60)
        print(f"📊 BATCH PREDICTION COMPLETED")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📄 Detailed results: {output_file}")
        print(f"📄 Simple results: {simple_output}")
        
        # Statistics
        valid_results = [r for r in results if r['predicted_age_months'] != 'ERROR']
        if valid_results:
            ages = [float(r['predicted_age_months']) for r in valid_results]
            print(f"\n📊 Summary Statistics:")
            print(f"   Images processed: {len(valid_results)}")
            print(f"   Min age: {min(ages):.1f} months ({min(ages)/12:.1f} years)")
            print(f"   Max age: {max(ages):.1f} months ({max(ages)/12:.1f} years)")
            print(f"   Mean age: {sum(ages)/len(ages):.1f} months ({sum(ages)/len(ages)/12:.1f} years)")
            
            # Age distribution
            stages = {}
            for result in valid_results:
                stage = result['development_stage']
                stages[stage] = stages.get(stage, 0) + 1
            
            print(f"\n📈 Age Distribution:")
            for stage, count in sorted(stages.items()):
                percentage = (count / len(valid_results)) * 100
                print(f"   {stage}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Interactive main function"""
    print("🦴 Batch Bone Age Predictor")
    print("="*50)
    print("✨ Uses your FlexibleBoneAgePredictor")
    print()
    
    # Get model path
    model_path = input("📂 Enter model path [best_bone_age_model.pth]: ").strip()
    if not model_path:
        model_path = "best_bone_age_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Get dataset directory
    dataset_dir = input("📁 Enter dataset directory [boneage-validation-dataset-1]: ").strip()
    if not dataset_dir:
        dataset_dir = "boneage-validation-dataset-1"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Directory not found: {dataset_dir}")
        return
    
    # Gender options
    print("\n👤 Gender Options:")
    print("  1. Both genders (shows average)")
    print("  2. Female only")
    print("  3. Male only")
    print("  4. Average of both genders")
    print("  5. Show both predictions separately")
    
    gender_choice = input("Choose gender option [1]: ").strip()
    if not gender_choice:
        gender_choice = "1"
    
    gender_map = {
        "1": None,         # Both (will average)
        "2": "female",
        "3": "male",
        "4": "average",
        "5": "both"        # Will show both
    }
    
    gender_option = gender_map.get(gender_choice, None)
    
    # TTA option
    use_tta = input("🔄 Use Test Time Augmentation for better accuracy? (y/n) [n]: ").strip().lower()
    use_tta = use_tta in ['y', 'yes']
    
    # Output filename
    output_file = input("💾 Output filename [auto-generate]: ").strip()
    if not output_file:
        output_file = None
    
    # Confirmation
    print(f"\n📋 Batch Settings:")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Gender: {gender_choice} ({gender_option or 'Both (average)'})")
    print(f"   TTA: {use_tta}")
    print(f"   Output: {output_file or 'Auto-generated'}")
    
    proceed = input(f"\nProceed with batch prediction? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("❌ Cancelled")
        return
    
    # Run batch prediction
    batch_predict(dataset_dir, model_path, output_file, gender_option, use_tta)

def quick_batch():
    """Quick batch with defaults"""
    print("🚀 Quick Batch Prediction")
    
    model_path = "best_bone_age_model.pth"
    dataset_dir = "boneage-validation-dataset-1"
    
    # Check if defaults exist
    if not os.path.exists(model_path):
        model_path = input(f"Model not found. Enter path: ").strip()
        if not os.path.exists(model_path):
            print(f"❌ Model not found")
            return
    
    if not os.path.exists(dataset_dir):
        dataset_dir = input(f"Dataset not found. Enter directory: ").strip()
        if not os.path.exists(dataset_dir):
            print(f"❌ Dataset not found")
            return
    
    print(f"📂 Model: {model_path}")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"👤 Gender: Both (average)")
    print(f"🔄 TTA: No")
    
    batch_predict(dataset_dir, model_path, None, None, False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_batch()
    else:
        main()