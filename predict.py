import os
import sys
import glob
from ultralytics import YOLO

def main():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "runs", "detect", "train", "weights", "best.pt")
    test_images_dir = os.path.join(current_dir, "test", "images")
    output_dir = os.path.join(current_dir, "runs", "detect", "predict")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        print("Please run train.py first.")
        sys.exit(1)
        
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        sys.exit(1)
        
    print(f"Loading trained YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # Get all test images
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        
    if not test_images:
        print(f"No images found in {test_images_dir}")
        sys.exit(1)
        
    print(f"Found {len(test_images)} test images. Starting batch inference...")
    
    # Run prediction
    # - source: path to test images
    # - conf: confidence threshold (e.g., 0.25)
    # - save: save predictions as images with bounding boxes plotted
    # - project: base output folder
    # - name: predict (saves to runs/detect/predict)
    # - exist_ok: overwrite existing folder if it exists
    results = model.predict(
        source=test_images_dir,
        conf=0.25,
        save=True,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )
    
    print("\nInference complete!")
    print(f"Predicted images are saved to: {output_dir}")
    
    # Compile simple summary statistics
    total_detections = 0
    images_with_defects = 0
    
    for r in results:
        boxes = r.boxes
        num_boxes = len(boxes)
        if num_boxes > 0:
            total_detections += num_boxes
            images_with_defects += 1
            
    print("=" * 50)
    print("TEST INFERENCE SUMMARY:")
    print(f"  Total images processed:       {len(results)}")
    print(f"  Total 'missing gem' defects:  {total_detections}")
    print(f"  Images containing defects:    {images_with_defects} / {len(results)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
