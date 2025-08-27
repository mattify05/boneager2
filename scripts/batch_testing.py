#!/usr/bin/env python3
"""
Simple Batch Bone Age Prediction Script
Creates a clean output file with just image names and predicted bone ages
"""

import os
import csv
import sys
from datetime import datetime
import glob

# Import functions from simple_predict.py
try:
    from bone_age.utils import load_model, predict_bone_age
except ImportError:
    print("❌ Error: Could not import from simple_predict.py")
    print("Make sure simple_predict.py is in the same directory as this script")
    sys.exit(1)

def get_image_files(dataset_dir):
    """Get all PNG image files from the dataset directory"""
    image_pattern = os.path.join(dataset_dir, "*.png")
    image_files = glob.glob(image_pattern)
    
    # Extract just the filenames without path and extension
    image_names = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        image_name = os.path.splitext(filename)[0]  # Remove .png extension
        image_names.append((image_name, img_path))
    
    return sorted(image_names)  # Sort by image name

def simple_batch_predict(dataset_dir="boneage-validation-dataset-1", output_file=None):
    """
    Process all images and create a simple output file with image names and predictions
    
    Args:
        dataset_dir: Directory containing the images
        output_file: CSV file to save results (optional)
    """
    
    # Load model once
    print("🔄 Loading model...")
    try:
        model, device = load_model('best_bone_age_model.pth')
        if model is None:
            print("❌ Failed to load model")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get all image files
    image_files = get_image_files(dataset_dir)
    if not image_files:
        print(f"❌ No PNG images found in {dataset_dir}")
        return
    
    # Prepare output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bone_age_results_{timestamp}.csv"
    
    print(f"🚀 Found {len(image_files)} images to process")
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"💾 Output file: {output_file}")
    print("=" * 60)
    
    # Process images and collect results
    results = []
    successful = 0
    failed = 0
    
    for i, (image_name, image_path) in enumerate(image_files, 1):
        print(f"[{i:4d}/{len(image_files):4d}] Processing {image_name}.png", end=" ")
        
        try:
            # Make prediction
            result = predict_bone_age(image_path, model, device, monte_carlo_samples=5)  # Reduced samples for speed
            
            # Store simplified result
            results.append({
                'image_name': image_name,
                'predicted_age_months': round(result['age_months'], 1),
                'predicted_age_years': round(result['age_years'], 1),
                'confidence': round(result['confidence'], 2),
                'uncertainty_months': round(result['uncertainty'], 1),
                'development_stage': result['stage']
            })
            
            successful += 1
            print(f"✅ {result['age_months']:.1f} months ({result['age_years']:.1f} years)")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'image_name': image_name,
                'predicted_age_months': 'ERROR',
                'predicted_age_years': 'ERROR',
                'confidence': 'ERROR',
                'uncertainty_months': 'ERROR',
                'development_stage': 'ERROR'
            })
            failed += 1
    
    # Save results to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_name', 
                'predicted_age_months', 
                'predicted_age_years',
                'confidence',
                'uncertainty_months',
                'development_stage'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print("=" * 60)
        print(f"📊 BATCH PREDICTION COMPLETED")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📄 Results saved to: {output_file}")
        
        # Also create a simple text file with just image_name and age
        simple_output = output_file.replace('.csv', '_simple.txt')
        with open(simple_output, 'w', encoding='utf-8') as f:
            f.write("Image_Name\tPredicted_Age_Months\tPredicted_Age_Years\n")
            for result in results:
                if result['predicted_age_months'] != 'ERROR':
                    f.write(f"{result['image_name']}\t{result['predicted_age_months']}\t{result['predicted_age_years']}\n")
        
        print(f"📄 Simple text file saved to: {simple_output}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main function with user-friendly interface"""
    print("🦴 Simple Bone Age Batch Predictor")
    print("=" * 50)
    
    # Get dataset directory
    dataset_dir = input("Enter dataset directory [boneage-validation-dataset-1]: ").strip()
    if not dataset_dir:
        dataset_dir = "boneage-validation-dataset-1"
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Directory not found: {dataset_dir}")
        return
    
    # Get output filename
    output_file = input("Enter output filename [auto-generate]: ").strip()
    if not output_file:
        output_file = None
    
    # Confirm before proceeding
    proceed = input(f"\nProcess all PNG images in '{dataset_dir}'? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("❌ Cancelled")
        return
    
    # Run the batch prediction
    simple_batch_predict(dataset_dir, output_file)

if __name__ == "__main__":
    main()