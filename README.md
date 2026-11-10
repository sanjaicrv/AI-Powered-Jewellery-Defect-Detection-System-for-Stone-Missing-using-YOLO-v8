# AI-Powered Jewellery Defect Detection using YOLOv8

This repository contains the complete implementation for a Computer Vision project focused on detecting jewellery defects (specifically **missing gems** on rings, necklaces, etc.) using the Ultralytics YOLOv8 architecture.

The project is optimized to run locally on a Windows machine with GPU acceleration using an **NVIDIA GeForce GTX 1650 (4GB VRAM)**.

---

## Table of Contents
1. [Dataset Information](#dataset-information)
2. [System Specifications](#system-specifications)
3. [Installation & Setup](#installation--setup)
4. [Dataset Validation & Processing](#dataset-validation--processing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Inference (Prediction)](#inference-prediction)
8. [File Structure](#file-structure)

---

## Dataset Information
The dataset was exported via Roboflow in YOLOv8 format:
* **Total Images**: 671 images
* **Classes**: 1 class (`missing gem`)
* **Split Structure**:
  * **Train**: 537 images
  * **Valid**: 34 images
  * **Test**: 100 images

---

## System Specifications
Training and testing were conducted under the following setup:
* **OS**: Windows 11
* **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
* **NVIDIA Driver**: 555.97
* **CUDA Driver**: 12.5 (PyTorch compiled with CUDA 12.4 support)
* **Python Version**: 3.12 (Python 3.14 was avoided as PyTorch stable wheels are not yet available for it).

---

## Installation & Setup

1. **Create Python 3.12 Virtual Environment**:
   ```powershell
   # Delete the old venv if it exists
   Remove-Item -Recurse -Force .\venv
   
   # Create new venv using Python 3.12
   & "C:\Users\SANJAI CRV\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
   ```

2. **Activate the environment**:
   * **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Command Prompt**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```

3. **Install Dependencies**:
   Install PyTorch with CUDA 12.4 support:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   Install the remaining packages (including `ultralytics` and data libraries):
   ```powershell
   pip install -r requirements.txt
   ```

---

## Dataset Validation & Processing

### 1. Polygon-to-Bounding-Box Conversion
The downloaded Roboflow dataset was annotated using polygon segments (YOLOv8 Segmentation format). To train a standard Object Detection model (`yolov8n.pt`), the annotations were converted into bounding boxes using `restore_and_convert.py`:
```powershell
python restore_and_convert.py
```
This script reads the original segments from the backup zip file, calculates the bounding box coordinates, and overwrites the text label files in the project.

### 2. Dataset Integrity Check
To validate the final dataset structure (missing labels, corrupt files, empty annotations, duplicate images, out-of-bounds bounding boxes, and class mismatches), run:
```powershell
python validate_dataset.py
```
It generates a full validation report at `dataset_validation_report.md`.

---

## Model Training
The training script [train.py](train.py) uses the YOLOv8 Nano model. It is configured to run on the GPU while maintaining compatibility with 4GB VRAM.

Run training:
```powershell
python train.py
```

### Key Training Hyperparameters:
* **Model**: YOLOv8 Nano (`yolov8n.pt`)
* **Image Size**: 640x640 (`imgsz=640`)
* **Device**: GPU (`device=0`)
* **Batch Size**: Auto-optimized (`batch=-1`) to maximize the usage of 4GB VRAM
* **AMP (Mixed Precision)**: `True` (enabled to speed up training and save VRAM)
* **Dataloader Workers**: `workers=2` (prevents Windows multiprocessing memory leaks)
* **Early Stopping**: `patience=20` (stops training if validation loss does not improve for 20 epochs)
* **Cache**: `False`

---

## Model Evaluation
Training was completed in **69 epochs** (early stopping triggered as no improvement was observed after epoch 49).

### Final Validation Metrics:
* **Precision**: `0.8964` (89.64%)
* **Recall**: `0.5676` (56.76%)
* **mAP@50**: `0.6553` (65.53%)
* **mAP@50-95**: `0.4572` (45.72%)
* **Inference Speed**: `9.5ms preprocess + preprocess/postprocess per image`

All metrics plots, confusion matrices, and training logs are saved automatically in the run outputs directory:
`runs/detect/train/`
* `results.png` (displays losses and accuracy metrics across epochs)
* `confusion_matrix.png` (confusion matrix of classes)
* `BoxPR_curve.png` / `BoxF1_curve.png` (precision-recall and F1 confidence curves)
* `val_batch0_pred.jpg` (visual predictions on the validation set)

---

---

## Multi-Modal Interactive Web Application

An interactive web application is included to run real-time AI jewellery anomaly inspection across **three modes**:
1. **📸 Image Inspection**: Drag-and-drop / paste macro photographs or pick from preloaded test dataset samples with side-by-side comparison, defect bounding boxes, and CSV breakdown.
2. **🎬 Video Inspection**: Upload video files (`.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`) or test the 1-click preloaded demo clip (`sample_jewellery_demo.mp4`). Features synchronized dual video playback (Original vs YOLOv8 Annotated), real-time progress bar, interactive defect timestamp jump-markers, and downloadable annotated MP4 videos.
3. **🔴 Live Camera Feed**: Real-time continuous defect detection via Browser Webcam (or Server USB Camera). Features a live dynamically adjustable confidence score slider that updates detection thresholds in real time, audible anomaly alert chime, live FPS and inference latency monitors, and instant snapshot capture.

### Running the Web Application:
* **Double-click batch runner**:
  ```cmd
  run_web_app.bat
  ```
* **Or run from PowerShell/terminal**:
  ```powershell
  .\venv\Scripts\python.exe web_app.py 8080
  ```
* Open your browser at: **`http://localhost:8080`**

---

## File Structure
```text
Anomaly Detection in Jewellery.v1i.yolov8/
│
├── train/                  # Training set (images and labels)
├── valid/                  # Validation set (images and labels)
├── test/                   # Test set (images and labels)
│
├── venv/                   # Python 3.12 Virtual Environment
│
├── data.yaml               # YOLOv8 dataset configuration file
├── requirements.txt        # PIP dependencies
│
├── restore_and_convert.py  # Script converting polygon labels to bounding boxes
├── validate_dataset.py     # Script checking dataset integrity and generating report
├── train.py                # Script training YOLOv8 Nano on GPU
├── predict.py              # Script performing inference on the test split
│
├── dataset_validation_report.md # Markdown report detailing dataset status
│
└── runs/
    └── detect/
        ├── train/          # Training outputs (weights, plots, csv logs)
        │   └── weights/
        │       ├── best.pt # Best weights (to be used for deployment)
        │       └── last.pt # Last epoch weights
        │
        └── predict/        # Visualization of predictions on the test split
```
