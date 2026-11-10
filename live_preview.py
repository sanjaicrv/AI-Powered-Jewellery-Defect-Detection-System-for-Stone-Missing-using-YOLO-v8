import os
import sys
import cv2
from ultralytics import YOLO

def main():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "runs", "detect", "train", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        print("Please train the model first.")
        sys.exit(1)
        
    print(f"Loading trained YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # Initialize the camera (default is 0 for built-in camera)
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # If the input is a number, convert it to integer
        if source.isdigit():
            source = int(source)
            
    print(f"\nInitializing camera source: {source}... Press 'q' key to quit the live preview.")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source: {source}. Check if it is connected or URL is correct.")
        sys.exit(1)
        
    # Set webcam resolution to 640x480 only if using a local USB webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    retry_count = 0
    max_retries = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame. Stream may have disconnected. Attempting to reconnect...")
            cap.release()
            
            # Try to reconnect
            connected = False
            for i in range(max_retries):
                print(f"Reconnection attempt {i+1}/{max_retries} in 2 seconds...")
                cv2.waitKey(2000)
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    print("Reconnection successful!")
                    connected = True
                    break
            
            if not connected:
                print("Error: Could not reconnect to stream after multiple attempts.")
                break
            continue
            
        # Run YOLOv8 inference on the live frame
        # - stream=True: returns generator for optimized memory usage during video processing
        # - conf=0.25: confidence threshold (only display defects with >25% confidence)
        results = model(frame, conf=0.25)
        
        # Plot the detections on the frame (returns annotated numpy array)
        annotated_frame = results[0].plot()
        
        # Display the live annotated frame
        cv2.imshow("AI Jewellery Defect Detection (Webcam Preview)", annotated_frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean release and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam preview closed successfully.")

if __name__ == "__main__":
    main()
