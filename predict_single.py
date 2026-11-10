import os
import sys
import argparse
import cv2
from PIL import Image
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test single jewellery image with trained YOLOv8 model")
    parser.add_argument("image_path", nargs="?", default="", help="Path to input jewellery image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="Path to weights file")
    parser.add_argument("--save-dir", type=str, default="runs/detect/predict_single", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Open window showing detection result")
    args = parser.parse_args()

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workspace_dir, args.weights)

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)

    # If no image path is passed, pick first sample image from test folder
    image_path = args.image_path
    if not image_path:
        test_dir = os.path.join(workspace_dir, "test", "images")
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                image_path = os.path.join(test_dir, files[0])
                print(f"No image path provided. Using sample test image: {files[0]}")
    
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image file not found: '{image_path}'")
        print("Usage: python predict_single.py <path_to_image> [--conf 0.25] [--show]")
        sys.exit(1)

    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running inference on: {image_path} (confidence: {args.conf * 100:.0f}%)...")
    results = model.predict(
        source=image_path,
        conf=args.conf,
        save=True,
        project=os.path.dirname(os.path.abspath(args.save_dir)),
        name=os.path.basename(args.save_dir),
        exist_ok=True,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes
    num_defects = len(boxes) if boxes is not None else 0

    print("\n" + "=" * 55)
    print("  JEWELLERY INSPECTION RESULT")
    print("=" * 55)
    if num_defects > 0:
        print(f"  [STATUS]: [!] DEFECT DETECTED ({num_defects} Missing Gem(s) Identified)")
        print("-" * 55)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for idx, (box, conf) in enumerate(zip(xyxy, confs)):
            b = [round(x) for x in box]
            print(f"   #Defect {idx + 1}: Confidence: {conf * 100:.1f}% | Bounding Box: {b}")
    else:
        print("  [STATUS]: [✓] QUALITY PASSED (No Missing Gem Anomalies Detected)")

    save_path = os.path.join(workspace_dir, args.save_dir, os.path.basename(image_path))
    print("-" * 55)
    print(f"  Annotated image saved to: {save_path}")
    print("=" * 55 + "\n")

    if args.show:
        annotated_img = result.plot()
        cv2.imshow(f"Inspection Result - {os.path.basename(image_path)}", annotated_img)
        print("Press any key in the window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
