import os
import sys
import torch
from ultralytics import YOLO

def check_gpu():
    print("=" * 50)
    print("System GPU Check:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")
        # Check VRAM
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Total GPU Memory: {total_memory:.2f} GB")
        if total_memory < 4.5:
            print("  VRAM Classification: Low (4GB range). Optimizing batch size and workers.")
    else:
        print("  WARNING: CUDA is NOT available. Running on CPU will be extremely slow!")
    print("=" * 50)

def main():
    check_gpu()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        sys.exit(1)
        
    print(f"Loading YOLOv8 nano pre-trained model ('yolov8n.pt')...")
    # Load pretrained YOLOv8 Nano model (recommended for 4GB VRAM)
    model = YOLO("yolov8n.pt") 

    print("\nStarting YOLOv8 training with optimal hyperparameters...")
    # Optimal Hyperparameters for GTX 1650 4GB VRAM and 671 images:
    # - data: path to data.yaml
    # - epochs: 100
    # - imgsz: 640
    # - device: 0 (NVIDIA GPU)
    # - batch: -1 (YOLOv8 AutoBatch size optimizer to maximize 4GB VRAM usage)
    # - workers: 2 (low workers prevents Windows multiprocessing OOM/handles issues)
    # - cache: False (disabled to prevent high system memory load)
    # - patience: 20 (Early stopping: stops if no improvement for 20 epochs)
    # - amp: True (Mixed precision training: reduces VRAM footprint and increases speed)
    # - plots: True (saves confusion matrix, PR curves, etc.)
    # - save: True (saves best.pt and last.pt checkpoints)
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        device=0,
        batch=-1,         # Auto-batch size optimizer
        workers=2,        # Safe for Windows
        cache=False,      # Prevents high memory consumption
        patience=20,      # Early stopping patience
        amp=True,         # Mixed precision enabled
        plots=True,       # Save training metrics/plots
        save=True,        # Save checkpoints
        project="runs/detect",
        name="train"
    )
    
    print("\nTraining Finished!")
    print(f"Model saved to: {os.path.join(current_dir, 'runs', 'detect', 'train')}")
    
    # Run validation to output final metrics
    print("\nEvaluating model on validation split...")
    val_results = model.val()
    
    # Print metrics
    print("=" * 50)
    print("FINAL EVALUATION METRICS:")
    # val_results.results_dict contains mAP metrics
    metrics = val_results.results_dict
    precision = metrics.get('metrics/precision(B)', 0.0)
    recall = metrics.get('metrics/recall(B)', 0.0)
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  mAP@50:      {map50:.4f}")
    print(f"  mAP@50-95:   {map95:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
