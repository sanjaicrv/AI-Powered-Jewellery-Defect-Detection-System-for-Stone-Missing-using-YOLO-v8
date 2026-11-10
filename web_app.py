import os
import sys
import io
import json
import base64
import time
import uuid
import socket
import threading
import urllib.parse
import re
import webbrowser
import ssl
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import qrcode

# Path configuration
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE_DIR, "runs", "detect", "train", "weights", "best.pt")
TEST_IMAGES_DIR = os.path.join(WORKSPACE_DIR, "test", "images")
DEMO_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "sample_jewellery_demo.mp4")
VIDEOS_DIR = os.path.join(WORKSPACE_DIR, "runs", "web_videos")
INPUT_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "inputs")
OUTPUT_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "outputs")

os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

# Verify model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at {MODEL_PATH}")
    print("Please check that runs/detect/train/weights/best.pt exists.")
    sys.exit(1)

print(f"Loading trained YOLOv8 model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# Ensure sample demo video exists
if not os.path.exists(DEMO_VIDEO_PATH):
    live_avi = os.path.join(WORKSPACE_DIR, "runs", "detect", "predict-2", "live.avi")
    if os.path.exists(live_avi):
        try:
            cap_sample = cv2.VideoCapture(live_avi)
            cap_sample.set(cv2.CAP_PROP_POS_FRAMES, 1460)
            w, h = 480, 854
            out_sample = cv2.VideoWriter(DEMO_VIDEO_PATH, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), 25.0, (w, h))
            if not out_sample.isOpened():
                out_sample = cv2.VideoWriter(DEMO_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))
            count = 0
            while count < 90:
                ret, frame = cap_sample.read()
                if not ret:
                    break
                out_sample.write(cv2.resize(frame, (w, h)))
                count += 1
            cap_sample.release()
            out_sample.release()
        except Exception as e:
            print(f"Warning: Failed to generate demo video clip: {e}")

# Helper: Network IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

# Helper: Ensure SSL certificate for HTTPS live camera streaming
def ensure_ssl_cert(cert_path=None, key_path=None, host_ip="127.0.0.1"):
    if cert_path is None:
        cert_path = os.path.join(WORKSPACE_DIR, "cert.pem")
    if key_path is None:
        key_path = os.path.join(WORKSPACE_DIR, "key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, host_ip),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aurum Vision QC"),
        ])
        alt_names = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            alt_names.append(x509.IPAddress(ipaddress.IPv4Address(host_ip)))
        except Exception:
            pass

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
    except Exception as e:
        print(f"Warning: Failed to generate self-signed SSL cert: {e}")

    return cert_path, key_path

# Helper: Generate QR Code data URL
def generate_qr_data_url(url):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#f59e0b", back_color="#0e121e")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# Video processing tasks store
video_tasks_lock = threading.Lock()
video_tasks = {}

def process_video_background(task_id, input_path, output_path, conf_threshold=0.25, frame_skip=1):
    with video_tasks_lock:
        task = video_tasks.get(task_id)
    if not task:
        return

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            with video_tasks_lock:
                task["status"] = "error"
                task["error"] = f"Failed to open video file: {os.path.basename(input_path)}"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = 1.0
        if width > 1280 or height > 1280:
            scale = 1280.0 / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        target_fps = max(1.0, orig_fps / frame_skip)

        writer = None
        try:
            writer = cv2.VideoWriter(output_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), target_fps, (width, height))
            if not writer.isOpened():
                writer = None
        except Exception:
            writer = None

        if writer is None:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

        with video_tasks_lock:
            task["total_frames"] = total_frames

        start_time = time.time()
        frame_idx = 0
        defect_frames_count = 0
        total_defect_instances = 0
        defect_timeline = []
        total_inference_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if (frame_idx - 1) % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))

            t_inf_0 = time.time()
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            t_inf = time.time() - t_inf_0
            total_inference_time += t_inf

            res = results[0]
            boxes = res.boxes
            num_boxes = len(boxes) if boxes is not None else 0

            if num_boxes > 0:
                defect_frames_count += 1
                total_defect_instances += num_boxes
                confs = boxes.conf.cpu().numpy().tolist()
                max_conf = max(confs) if confs else 0.0
                clss = boxes.cls.cpu().numpy().tolist()
                classes = [model.names.get(int(c), str(int(c))) for c in clss]

                timestamp = frame_idx / orig_fps
                defect_timeline.append({
                    "time_sec": round(timestamp, 2),
                    "frame": frame_idx,
                    "count": num_boxes,
                    "max_conf": round(max_conf, 3),
                    "classes": classes
                })

            annotated_frame = res.plot()
            writer.write(annotated_frame)

            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0

            with video_tasks_lock:
                task["current_frame"] = frame_idx
                task["progress"] = min(99.0, round((frame_idx / max(1, total_frames)) * 100, 1))
                task["defects_count"] = total_defect_instances
                task["defect_frames"] = defect_frames_count
                task["fps"] = round(proc_fps, 1)

        cap.release()
        if writer is not None:
            writer.release()

        total_elapsed = time.time() - start_time
        processed_count = (frame_idx + frame_skip - 1) // frame_skip
        avg_inf_ms = (total_inference_time / max(1, processed_count)) * 1000.0

        with video_tasks_lock:
            task["status"] = "done"
            task["progress"] = 100.0
            task["results"] = {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "defect_frames": defect_frames_count,
                "total_defects": total_defect_instances,
                "defect_rate_percent": round((defect_frames_count / max(1, processed_count)) * 100.0, 1),
                "duration_sec": round(total_frames / orig_fps, 2),
                "processing_time_sec": round(total_elapsed, 2),
                "avg_inference_time_ms": round(avg_inf_ms, 1),
                "avg_fps": round(processed_count / max(0.01, total_elapsed), 1),
                "defect_timeline": defect_timeline,
                "is_defective": total_defect_instances > 0,
                "conf_used": conf_threshold
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        with video_tasks_lock:
            task["status"] = "error"
            task["error"] = str(e)


# Server Hardware Camera Streamer
class ServerCameraStreamer:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()

    def get_frame(self, conf=0.25):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    return None
            ret, frame = self.cap.read()
            if not ret:
                return None
            res = model.predict(source=frame, conf=conf, verbose=False)[0]
            annotated = res.plot()
            ret, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ret:
                return jpeg.tobytes()
            return None

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

server_camera = ServerCameraStreamer()

# External IP / RTSP Camera Streamer (e.g. iOS Live-Reporter, IP Camera Lite, DroidCam)
class IPCameraStreamer:
    def __init__(self):
        self.cap = None
        self.url = ""
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.fps = 0.0

    def start(self, url):
        self.stop()
        clean_url = (url or "").strip()
        if not clean_url:
            return False

        if clean_url.lower().startswith("rtsp://"):
            # Enable TCP transport, zero-buffering, and 3s connection timeout for low-latency RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;3000000"

        try:
            cap = cv2.VideoCapture(clean_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(clean_url)
            if not cap.isOpened():
                return False
        except Exception as e:
            print(f"Error opening IP camera stream: {e}")
            return False

        with self.lock:
            self.cap = cap
            self.url = clean_url
            self.running = True
            self.latest_frame = None
            self.fps = 0.0

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

        # Wait up to 3.0 seconds for first frame
        for _ in range(30):
            with self.lock:
                if self.latest_frame is not None:
                    return True
            time.sleep(0.1)

        with self.lock:
            return self.cap is not None and self.cap.isOpened()

    def _reader_loop(self):
        fps_count = 0
        t_fps = time.time()
        while self.running:
            with self.lock:
                cap = self.cap
                if cap is None or not cap.isOpened():
                    break
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                fps_count += 1
                now = time.time()
                if now - t_fps >= 1.0:
                    self.fps = round(fps_count / (now - t_fps), 1)
                    fps_count = 0
                    t_fps = now

    def get_frame(self, conf=0.25):
        frame = None
        with self.lock:
            if not self.running or self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
            fps_val = self.fps

        # Downscale ultra-high resolution streams (e.g. 4K/1080p phone cameras) to max 1280px for real-time 30 FPS inference
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        t0 = time.time()
        res = model.predict(source=frame, conf=conf, verbose=False)[0]
        lat = (time.time() - t0) * 1000.0
        ann = res.plot()
        ret, buf = cv2.imencode('.jpg', ann, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
            boxes = res.boxes
            num_boxes = len(boxes) if boxes is not None else 0
            detections = []
            if boxes is not None and num_boxes > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                names = model.names
                for i in range(num_boxes):
                    cls_id = int(clss[i])
                    detections.append({
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": float(confs[i]),
                        "bbox": xyxy[i].tolist()
                    })
            return {
                "success": True,
                "annotated_image": b64,
                "detections_count": num_boxes,
                "is_defective": num_boxes > 0,
                "inference_time_ms": lat,
                "detections": detections,
                "fps": fps_val
            }
        return None

    def stop(self):
        self.running = False
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_frame = None
            self.fps = 0.0

ip_camera = IPCameraStreamer()

# =====================================================================
# Real-Time Decoupled GPU Inference & MJPEG Broadcast Pipeline
# =====================================================================
class PhoneStreamPipeline:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.raw_frame = None
        self.raw_timestamp = 0
        self.device_name = "Mobile Scanner"
        self.conf = 0.25

        self.annotated_jpeg = None
        self.annotated_b64 = None
        self.annotated_timestamp = 0
        self.detections_count = 0
        self.is_defective = False
        self.inference_time_ms = 0.0
        self.fps = 0.0
        self.running = True

        self.worker = threading.Thread(target=self._gpu_worker, daemon=True)
        self.worker.start()

    def push_raw_frame(self, frame_bgr, device="Mobile Scanner", conf=0.25):
        with self.cond:
            self.raw_frame = frame_bgr
            self.raw_timestamp = time.time()
            self.device_name = device
            self.conf = conf
            self.cond.notify()

    def _gpu_worker(self):
        fps_count = 0
        t_fps_start = time.time()
        while self.running:
            with self.cond:
                while self.raw_frame is None and self.running:
                    self.cond.wait(timeout=0.1)
                if not self.running:
                    break
                frame = self.raw_frame
                conf = self.conf
                device = self.device_name
                self.raw_frame = None  # Consume latest frame to avoid backlog lag

            try:
                t0 = time.time()
                results = model.predict(source=frame, conf=conf, verbose=False)
                inf_time = (time.time() - t0) * 1000.0

                res = results[0]
                annotated_bgr = res.plot()
                ret, jpeg_buf = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                boxes = res.boxes
                num_boxes = len(boxes) if boxes is not None else 0

                fps_count += 1
                now = time.time()
                if now - t_fps_start >= 1.0:
                    self.fps = round(fps_count / (now - t_fps_start), 1)
                    fps_count = 0
                    t_fps_start = now

                if ret:
                    jpeg_bytes = jpeg_buf.tobytes()
                    with self.lock:
                        self.annotated_jpeg = jpeg_bytes
                        self.annotated_b64 = None
                        self.annotated_timestamp = now
                        self.detections_count = num_boxes
                        self.is_defective = num_boxes > 0
                        self.inference_time_ms = inf_time
                        self.device_name = device
            except Exception:
                pass

    def get_annotated_jpeg(self):
        with self.lock:
            return self.annotated_jpeg, self.annotated_timestamp

    def get_telemetry(self):
        with self.lock:
            return {
                "has_frame": self.annotated_jpeg is not None,
                "timestamp": self.annotated_timestamp,
                "detections_count": self.detections_count,
                "is_defective": self.is_defective,
                "inference_time_ms": round(self.inference_time_ms, 1),
                "device_name": self.device_name,
                "fps": self.fps
            }

phone_pipeline = PhoneStreamPipeline()
phone_stream_lock = phone_pipeline.lock


# =====================================================================
# MOBILE WEB APP TEMPLATE (Dedicated Touch Interface for Smartphones)
# =====================================================================
MOBILE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>AURUM VISION | Wireless Mobile Scanner</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #07090e;
      --bg-surface: #0e121e;
      --bg-card: #131929;
      --gold-primary: #f59e0b;
      --gold-light: #fbbf24;
      --gold-glow: rgba(245, 158, 11, 0.35);
      --emerald-pass: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.3);
      --ruby-defect: #f43f5e;
      --ruby-glow: rgba(244, 63, 94, 0.35);
      --border: rgba(255, 255, 255, 0.08);
      --border-gold: rgba(245, 158, 11, 0.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-base);
      color: #fff;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 0.9rem;
      user-select: none;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.85rem;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .gem-icon {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, #fbbf24, #d97706);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 14px var(--gold-glow);
    }
    .gem-icon svg { width: 22px; height: 22px; stroke: #fff; fill: none; }
    h1 { font-family: 'Outfit', sans-serif; font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; }
    .subhead { font-size: 0.7rem; color: #94a3b8; }
    
    .status-badge {
      font-size: 0.68rem;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      padding: 4px 9px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .status-secure {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .status-snap {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* iOS Guidance Banner */
    .ios-notice-box {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(19, 25, 41, 0.95) 100%);
      border: 1px solid var(--border-gold);
      border-radius: 12px;
      padding: 0.9rem;
      margin-bottom: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .ios-notice-title {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--gold-light);
      font-family: 'Outfit', sans-serif;
    }
    .ios-notice-desc {
      font-size: 0.75rem;
      color: #cbd5e1;
      line-height: 1.35;
    }
    .btn-https-link {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #07090e;
      text-decoration: none;
      font-family: 'Outfit', sans-serif;
      font-size: 0.82rem;
      font-weight: 700;
      padding: 8px 12px;
      border-radius: 8px;
      margin-top: 4px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      box-shadow: 0 2px 10px var(--gold-glow);
    }
    .ios-notice-sub {
      font-size: 0.68rem;
      color: #94a3b8;
      font-style: italic;
    }

    /* Viewfinder */
    .viewfinder-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
      margin-bottom: 0.85rem;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    }
    .viewfinder-body {
      width: 100%;
      aspect-ratio: 4 / 3;
      background: repeating-conic-gradient(#090c14 0% 25%, #0d121f 0% 50%) 50% / 20px 20px;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    video, img, canvas {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .reticle {
      position: absolute;
      width: 18px;
      height: 18px;
      border-color: var(--gold-primary);
      pointer-events: none;
      z-index: 5;
    }
    .ret-tl { top: 12px; left: 12px; border-top: 2px solid; border-left: 2px solid; }
    .ret-tr { top: 12px; right: 12px; border-top: 2px solid; border-right: 2px solid; }
    .ret-bl { bottom: 12px; left: 12px; border-bottom: 2px solid; border-left: 2px solid; }
    .ret-br { bottom: 12px; right: 12px; border-bottom: 2px solid; border-right: 2px solid; }
    
    .hud-badge {
      position: absolute;
      top: 10px;
      left: 10px;
      right: 10px;
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      font-weight: 600;
      z-index: 10;
      pointer-events: none;
    }
    .hud-pill {
      background: rgba(7, 9, 14, 0.85);
      backdrop-filter: blur(6px);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      font-family: 'JetBrains Mono', monospace;
    }

    /* Controls */
    .ctrl-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 11px;
    }
    .slider-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      color: #94a3b8;
    }
    .presets-row {
      display: flex;
      gap: 6px;
    }
    .preset-chip {
      flex: 1;
      padding: 5px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.7rem;
      font-family: 'JetBrains Mono', monospace;
      color: #94a3b8;
      text-align: center;
      cursor: pointer;
    }
    .preset-chip.active {
      background: rgba(245, 158, 11, 0.2);
      border-color: var(--gold-primary);
      color: var(--gold-light);
      font-weight: 700;
    }
    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #1e263d;
      outline: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--gold-primary);
      border: 2px solid #fff;
      box-shadow: 0 0 10px var(--gold-glow);
    }
    
    .btn-main {
      width: 100%;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #07090e;
      font-family: 'Outfit', sans-serif;
      font-size: 0.98rem;
      font-weight: 700;
      padding: 13px;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 18px var(--gold-glow);
      transition: all 0.2s ease;
    }
    .btn-main:active { transform: scale(0.98); }
    .btn-stop {
      background: linear-gradient(135deg, #f43f5e, #be123c);
      color: #fff;
      box-shadow: 0 4px 18px var(--ruby-glow);
    }
    .btn-sub {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
      border: 1px solid var(--border);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 10px;
      border-radius: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      text-decoration: none;
    }
    .btn-sub:active { background: rgba(255, 255, 255, 0.15); }

    /* Results Telemetry */
    .result-telemetry-box {
      margin-top: 0.85rem;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.9rem;
      display: none;
    }
    .result-verdict {
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .verdict-pass {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }
    .verdict-defect {
      background: rgba(244, 63, 94, 0.15);
      border: 1px solid rgba(244, 63, 94, 0.4);
      color: #fecdd3;
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <!-- Header -->
  <header>
    <div class="brand">
      <div class="gem-icon">
        <svg viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
      </div>
      <div>
        <h1>AURUM MOBILE</h1>
        <p class="subhead">Precision Wireless QC Scanner</p>
      </div>
    </div>
    <div class="status-badge status-secure" id="netBadge">SYNC READY</div>
  </header>

  <!-- iOS Safari HTTP Guidance Banner -->
  <div class="ios-notice-box" id="iosNoticeBox" style="display: none;">
    <div class="ios-notice-title">
      <svg width="17" height="17" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
      <span>iPhone Live Stream Notice</span>
    </div>
    <p class="ios-notice-desc">
      iOS Safari requires <strong>HTTPS</strong> for 30 FPS continuous video streaming.
    </p>
    <a id="btnUpgradeHttps" href="#" class="btn-https-link">
      <span>⚡ Switch to Secure HTTPS Stream (Port 8443)</span>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
    </a>
    <div class="ios-notice-sub">
      (Tap <em>"Show Details" &rarr; "Visit this website"</em> once on Safari to unlock)
    </div>
  </div>

  <!-- Viewfinder -->
  <div class="viewfinder-card">
    <div class="viewfinder-body">
      <div class="reticle ret-tl"></div>
      <div class="reticle ret-tr"></div>
      <div class="reticle ret-bl"></div>
      <div class="reticle ret-br"></div>
      
      <div class="hud-badge">
        <div class="hud-pill" id="verdictPill" style="color: #34d399;">STANDBY</div>
        <div class="hud-pill" id="fpsPill" style="color: var(--gold-light);">0 FPS</div>
      </div>

      <video id="mobVideo" playsinline autoplay muted style="display: none;"></video>
      <canvas id="mobCanvas" style="display: none;"></canvas>
      <img id="mobImg" style="display: none;" alt="Neural Output">
      
      <div id="mobEmpty" style="text-align: center; color: #64748b; padding: 2rem;">
        <svg width="42" height="42" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="opacity: 0.35; margin-bottom: 8px;">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
        </svg>
        <p id="mobEmptyText" style="font-size: 0.85rem;">Ready to inspect jewellery specimen</p>
        <p id="mobEmptySub" style="font-size: 0.72rem; color: #475569; margin-top: 4px;">Tap action button below</p>
      </div>
    </div>
  </div>

  <!-- Controls -->
  <div class="ctrl-card">
    <div class="slider-row">
      <span>Neural Sensitivity:</span>
      <strong id="mobConfVal" style="color: var(--gold-light); font-family: 'JetBrains Mono', monospace;">25%</strong>
    </div>
    <input type="range" id="mobConfSlider" min="10" max="90" value="25" step="5">
    <div class="presets-row">
      <button class="preset-chip" onclick="setPreset(15)">15% Micro</button>
      <button class="preset-chip active" onclick="setPreset(25)">25% Standard</button>
      <button class="preset-chip" onclick="setPreset(50)">50% High</button>
      <button class="preset-chip" onclick="setPreset(75)">75% Strict</button>
    </div>

    <!-- Primary Action Button -->
    <button id="btnAction" class="btn-main" onclick="handlePrimaryAction()">
      <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
      <span id="btnActionText">Start Wireless Stream</span>
    </button>

    <!-- Secondary Row -->
    <div style="display: flex; gap: 8px;">
      <button id="btnFlipCam" class="btn-sub" style="flex: 1;" onclick="flipCamera()">
        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
        Flip Camera
      </button>
      <button class="btn-sub" style="flex: 1;" onclick="triggerNativeSnap()">
        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><path d="M19 4h-3.5L14 2H10L8.5 4H5a2 2 0 00-2 2v12a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z"/></svg>
        <span>Snap Photo QC</span>
      </button>
    </div>

    <!-- Hidden Native File/Camera Input -->
    <input type="file" id="mobFileInput" accept="image/*" capture="environment" style="display: none;" onchange="handleSnapUpload(this)">
  </div>

  <!-- Telemetry Card (Shown after inspection) -->
  <div class="result-telemetry-box" id="telemetryCard">
    <div class="result-verdict verdict-pass" id="resVerdictBanner">
      <span id="resVerdictIcon">✓</span>
      <span id="resVerdictText">SPECIMEN FLAWLESS</span>
    </div>
    <div class="meta-row">
      <span>MISSING GEMS: <strong id="resDefectCount" style="color: #fff;">0</strong></span>
      <span>INFERENCE: <strong id="resLatency" style="color: #38bdf8;">14.2 ms</strong></span>
      <span style="color: #34d399;">✓ DESKTOP SYNCED</span>
    </div>
  </div>

  <script>
    const isHttps = window.location.protocol === 'https:';
    const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

    let isStreaming = false;
    let streamMedia = null;
    let facingMode = 'environment';
    let loopId = null;
    let isProcessing = false;
    let fpsCounter = 0;
    let fpsTimer = null;

    const mobVideo = document.getElementById('mobVideo');
    const mobCanvas = document.getElementById('mobCanvas');
    const mobImg = document.getElementById('mobImg');
    const mobEmpty = document.getElementById('mobEmpty');
    const mobEmptyText = document.getElementById('mobEmptyText');
    const mobEmptySub = document.getElementById('mobEmptySub');
    const btnAction = document.getElementById('btnAction');
    const btnActionText = document.getElementById('btnActionText');
    const verdictPill = document.getElementById('verdictPill');
    const fpsPill = document.getElementById('fpsPill');
    const mobConfSlider = document.getElementById('mobConfSlider');
    const mobConfVal = document.getElementById('mobConfVal');
    const mobFileInput = document.getElementById('mobFileInput');
    const iosNoticeBox = document.getElementById('iosNoticeBox');
    const btnUpgradeHttps = document.getElementById('btnUpgradeHttps');
    const btnFlipCam = document.getElementById('btnFlipCam');
    const netBadge = document.getElementById('netBadge');
    const telemetryCard = document.getElementById('telemetryCard');
    const resVerdictBanner = document.getElementById('resVerdictBanner');
    const resVerdictText = document.getElementById('resVerdictText');
    const resVerdictIcon = document.getElementById('resVerdictIcon');
    const resDefectCount = document.getElementById('resDefectCount');
    const resLatency = document.getElementById('resLatency');

    // Initialize UI based on security context
    function initSecurityContext() {
      const host = window.location.hostname;
      btnUpgradeHttps.href = `https://${host}:8443/mobile`;

      if (hasMediaDevices) {
        // Full WebRTC / Live Stream Supported (HTTPS or localhost)
        iosNoticeBox.style.display = 'none';
        netBadge.textContent = '🔒 HTTPS READY';
        netBadge.className = 'status-badge status-secure';
        btnActionText.textContent = '⚡ Start Wireless Stream';
        mobEmptyText.textContent = 'Tap Start Wireless Stream below';
        mobEmptySub.textContent = 'Continuous 30 FPS jewellery inspection';
      } else {
        // Plain HTTP on iOS (MediaDevices blocked by Safari WebKit)
        iosNoticeBox.style.display = 'flex';
        netBadge.textContent = '📸 HTTP SNAP MODE';
        netBadge.className = 'status-badge status-snap';
        btnActionText.textContent = '📸 Snap & Inspect Specimen';
        btnFlipCam.style.display = 'none';
        mobEmptyText.textContent = 'Tap Snap & Inspect Specimen below';
        mobEmptySub.textContent = 'Uses native camera (100% reliable over HTTP)';
      }
    }
    initSecurityContext();

    function setPreset(val) {
      mobConfSlider.value = val;
      mobConfVal.textContent = val + '%';
      document.querySelectorAll('.preset-chip').forEach(c => {
        c.classList.toggle('active', c.textContent.startsWith(val + '%'));
      });
    }

    mobConfSlider.addEventListener('input', (e) => {
      mobConfVal.textContent = e.target.value + '%';
    });

    function handlePrimaryAction() {
      if (hasMediaDevices) {
        toggleStream();
      } else {
        triggerNativeSnap();
      }
    }

    function triggerNativeSnap() {
      mobFileInput.value = '';
      mobFileInput.click();
    }

    async function toggleStream() {
      if (isStreaming) {
        stopStream();
      } else {
        await startStream();
      }
    }

    let telemetryTimer = null;
    let isUploadingBlob = false;
    let lastTransmitTs = 0;

    async function startStream() {
      try {
        const constraints = {
          video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        };
        streamMedia = await navigator.mediaDevices.getUserMedia(constraints);
        mobVideo.srcObject = streamMedia;
        await mobVideo.play();

        isStreaming = true;
        mobEmpty.style.display = 'none';
        mobVideo.style.display = 'block';
        mobImg.style.display = 'none';
        btnAction.classList.add('btn-stop');
        btnActionText.textContent = '🛑 Terminate Stream';
        netBadge.textContent = 'BROADCASTING 30 FPS';
        netBadge.style.color = '#38bdf8';

        fpsCounter = 0;
        fpsTimer = setInterval(() => {
          fpsPill.textContent = `${fpsCounter} FPS`;
          fpsCounter = 0;
        }, 1000);

        telemetryTimer = setInterval(pollMobileTelemetry, 300);

        scheduleFrame();
      } catch (err) {
        console.warn('Live stream error, activating Snap mode:', err);
        iosNoticeBox.style.display = 'flex';
        btnActionText.textContent = '📸 Snap & Inspect Specimen';
        triggerNativeSnap();
      }
    }

    function stopStream() {
      isStreaming = false;
      isUploadingBlob = false;
      if (streamMedia) {
        streamMedia.getTracks().forEach(t => t.stop());
        streamMedia = null;
      }
      if (loopId) cancelAnimationFrame(loopId);
      if (fpsTimer) clearInterval(fpsTimer);
      if (telemetryTimer) clearInterval(telemetryTimer);

      mobVideo.style.display = 'none';
      mobImg.style.display = 'none';
      mobEmpty.style.display = 'block';
      btnAction.classList.remove('btn-stop');
      btnActionText.textContent = hasMediaDevices ? '⚡ Start Wireless Stream' : '📸 Snap & Inspect Specimen';
      netBadge.textContent = hasMediaDevices ? '🔒 HTTPS READY' : '📸 HTTP SNAP MODE';
      netBadge.style.color = hasMediaDevices ? '#34d399' : '#fbbf24';
      verdictPill.textContent = 'STANDBY';
      verdictPill.style.color = '#34d399';
    }

    async function flipCamera() {
      facingMode = facingMode === 'environment' ? 'user' : 'environment';
      if (isStreaming) {
        stopStream();
        await startStream();
      }
    }

    function scheduleFrame() {
      if (!isStreaming) return;
      loopId = requestAnimationFrame(captureAndTransmit);
    }

    async function captureAndTransmit(now) {
      if (!isStreaming) return;

      // Throttle upload to ~30 FPS (every 33ms)
      if (now - lastTransmitTs >= 33 && !isUploadingBlob) {
        if (mobVideo.videoWidth > 0 && mobVideo.videoHeight > 0) {
          lastTransmitTs = now;
          // Scale to 480px width for minimal Wi-Fi overhead and instant transfer
          const targetW = 480;
          const targetH = Math.round((mobVideo.videoHeight * targetW) / mobVideo.videoWidth);
          mobCanvas.width = targetW;
          mobCanvas.height = targetH;
          const ctx = mobCanvas.getContext('2d');
          ctx.drawImage(mobVideo, 0, 0, targetW, targetH);

          isUploadingBlob = true;
          mobCanvas.toBlob(blob => {
            if (!blob || !isStreaming) {
              isUploadingBlob = false;
              return;
            }
            const conf = parseFloat(mobConfSlider.value) / 100.0;
            const device = navigator.userAgent.includes('iPhone') ? 'iPhone' : 'Android Mobile';

            // Fire-and-forget binary push directly to GPU pipeline
            fetch(`/api/phone_stream/push_frame?conf=${conf}&device=${device}`, {
              method: 'POST',
              headers: { 'Content-Type': 'image/jpeg' },
              body: blob
            }).then(() => {
              fpsCounter++;
            }).catch(() => {}).finally(() => {
              isUploadingBlob = false;
            });
          }, 'image/jpeg', 0.65);
        }
      }
      scheduleFrame();
    }

    async function pollMobileTelemetry() {
      if (!isStreaming) return;
      try {
        const res = await fetch('/api/phone_stream/telemetry');
        const data = await res.json();
        if (data.has_frame) {
          if (data.is_defective) {
            verdictPill.textContent = `ALERT: ${data.detections_count} DEFECT`;
            verdictPill.style.color = 'var(--ruby-defect)';
            document.querySelector('.viewfinder-card').style.boxShadow = '0 0 25px rgba(244, 63, 94, 0.45)';
          } else {
            verdictPill.textContent = 'FLAWLESS PASS';
            verdictPill.style.color = 'var(--emerald-pass)';
            document.querySelector('.viewfinder-card').style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.25)';
          }
        }
      } catch (e) {}
    }

    // High-Resolution Native Photo Snap Upload
    function handleSnapUpload(input) {
      if (!input.files || !input.files[0]) return;
      const file = input.files[0];
      const reader = new FileReader();

      mobEmptyText.textContent = 'Analyzing jewellery specimen...';
      mobEmptySub.textContent = 'Running YOLOv8 inference on CUDA GPU';

      reader.onload = async (e) => {
        try {
          // Pre-scale image to max 1280px for instant upload over Wi-Fi
          const img = new Image();
          img.onload = async () => {
            const maxDim = 1280;
            let w = img.width;
            let h = img.height;
            if (w > maxDim || h > maxDim) {
              if (w > h) { h = Math.round((h * maxDim) / w); w = maxDim; }
              else { w = Math.round((w * maxDim) / h); h = maxDim; }
            }
            mobCanvas.width = w;
            mobCanvas.height = h;
            const ctx = mobCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            const scaledDataUrl = mobCanvas.toDataURL('image/jpeg', 0.88);

            const conf = parseFloat(mobConfSlider.value) / 100.0;
            const resp = await fetch('/api/phone_stream/frame', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                image: scaledDataUrl,
                conf: conf,
                device: 'Mobile Camera Snap'
              })
            });
            const data = await resp.json();

            if (data.success) {
              mobEmpty.style.display = 'none';
              mobImg.src = data.annotated_image;
              mobImg.style.display = 'block';

              telemetryCard.style.display = 'block';
              resDefectCount.textContent = data.detections_count;
              resLatency.textContent = `${data.inference_time_ms.toFixed(1)} ms`;

              if (data.is_defective) {
                verdictPill.textContent = `ALERT: ${data.detections_count} DEFECT`;
                verdictPill.style.color = 'var(--ruby-defect)';
                resVerdictBanner.className = 'result-verdict verdict-defect';
                resVerdictIcon.textContent = '⚠️';
                resVerdictText.textContent = `ALERT: ${data.detections_count} MISSING GEM DEFECT(S) FOUND`;
              } else {
                verdictPill.textContent = 'FLAWLESS PASS';
                verdictPill.style.color = 'var(--emerald-pass)';
                resVerdictBanner.className = 'result-verdict verdict-pass';
                resVerdictIcon.textContent = '✓';
                resVerdictText.textContent = 'SPECIMEN FLAWLESS — ZERO DEFECTS';
              }

              btnActionText.textContent = '📸 Snap Next Specimen';
            }
          };
          img.src = e.target.result;
        } catch (err) {
          alert('Upload error: ' + err.message);
        }
      };
      reader.readAsDataURL(file);
    }
  </script>
</body>
</html>
"""


# =====================================================================
# DESKTOP MAIN APPLICATION TEMPLATE (AURUM VISION)
# =====================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AURUM VISION | AI Jewellery Defect Detection Suite</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #07090e;
      --bg-surface: #0e121e;
      --bg-card: #131929;
      --border-subtle: rgba(255, 255, 255, 0.07);
      --border-gold: rgba(245, 158, 11, 0.35);
      --text-main: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --gold-primary: #f59e0b;
      --gold-light: #fbbf24;
      --gold-glow: rgba(245, 158, 11, 0.28);
      --emerald-pass: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --emerald-bg: rgba(16, 185, 129, 0.12);
      --ruby-defect: #f43f5e;
      --ruby-glow: rgba(244, 63, 94, 0.3);
      --ruby-bg: rgba(244, 63, 94, 0.14);
      --ice-telemetry: #38bdf8;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      line-height: 1.55;
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(245, 158, 11, 0.04) 0%, transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(56, 189, 248, 0.03) 0%, transparent 40%);
      background-attachment: fixed;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(14, 18, 30, 0.88);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      padding: 0.85rem 2rem;
    }
    .header-inner {
      max-width: 1520px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      flex-wrap: wrap;
    }
    .brand-wrap { display: flex; align-items: center; gap: 14px; }
    .brand-gem-icon {
      width: 46px;
      height: 46px;
      border-radius: 13px;
      background: linear-gradient(135deg, #fbbf24 0%, #d97706 60%, #92400e 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 25px var(--gold-glow);
      border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .brand-gem-icon svg { width: 26px; height: 26px; stroke: #fff; fill: none; stroke-width: 2.2; }
    .brand-title {
      font-family: var(--font-display);
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      background: linear-gradient(90deg, #ffffff 40%, var(--gold-light) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .mode-nav {
      display: flex;
      background: rgba(7, 9, 14, 0.7);
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--border-subtle);
      gap: 4px;
    }
    .mode-tab-btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 9px;
      font-family: var(--font-display);
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-secondary);
      background: transparent;
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.25s;
    }
    .mode-tab-btn:hover { color: #fff; background: rgba(255, 255, 255, 0.04); }
    .mode-tab-btn.active {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #07090e;
      box-shadow: 0 4px 16px var(--gold-glow);
    }
    .telemetry-pills { display: flex; align-items: center; gap: 10px; }
    .pill-badge {
      font-family: var(--font-mono);
      font-size: 0.73rem;
      padding: 5px 12px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
    }
    .pill-cuda {
      background: rgba(56, 189, 248, 0.1);
      color: var(--ice-telemetry);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .pill-status {
      background: var(--emerald-bg);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: currentColor;
      box-shadow: 0 0 10px currentColor;
    }
    main {
      flex: 1;
      max-width: 1520px;
      width: 100%;
      margin: 0 auto;
      padding: 1.75rem 2rem 2.5rem;
    }
    .mode-view {
      display: none;
      grid-template-columns: 460px 1fr;
      gap: 1.75rem;
    }
    .mode-view.active { display: grid; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 1180px) { .mode-view { grid-template-columns: 1fr; } }
    .luxury-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.6rem;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
      position: relative;
      overflow: hidden;
    }
    .luxury-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(245, 158, 11, 0.4), transparent);
    }
    .card-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.3rem;
    }
    .card-heading h2 {
      font-family: var(--font-display);
      font-size: 1.12rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-heading .tag {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.05);
      padding: 3px 8px;
      border-radius: 6px;
    }
    .luxury-dropzone {
      border: 2px dashed rgba(245, 158, 11, 0.35);
      background: linear-gradient(180deg, rgba(19, 25, 41, 0.6) 0%, rgba(14, 18, 30, 0.8) 100%);
      border-radius: var(--radius-md);
      padding: 2rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
    }
    .luxury-dropzone:hover, .luxury-dropzone.dragover {
      border-color: var(--gold-light);
      background: rgba(245, 158, 11, 0.07);
      transform: translateY(-2px);
      box-shadow: 0 10px 25px var(--gold-glow);
    }
    .dropzone-icon-wrap {
      width: 52px;
      height: 52px;
      margin: 0 auto 12px;
      border-radius: 14px;
      background: rgba(245, 158, 11, 0.12);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--gold-primary);
    }
    .dropzone-icon-wrap svg { width: 28px; height: 28px; }
    .control-block { margin-top: 1.3rem; }
    .control-label-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 0.86rem;
      font-weight: 500;
    }
    .conf-badge {
      font-family: var(--font-mono);
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--gold-light);
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.3);
      padding: 2px 8px;
      border-radius: 6px;
    }
    .luxury-slider {
      -webkit-appearance: none;
      width: 100%;
      height: 7px;
      border-radius: 4px;
      background: #1e263d;
      outline: none;
    }
    .luxury-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: linear-gradient(135deg, #fbbf24, #d97706);
      cursor: pointer;
      box-shadow: 0 0 12px var(--gold-glow);
      border: 2px solid #ffffff;
    }
    .conf-presets-row { display: flex; gap: 6px; margin-top: 10px; }
    .conf-chip {
      flex: 1;
      padding: 5px 6px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-secondary);
      text-align: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .conf-chip:hover { background: rgba(245, 158, 11, 0.12); color: var(--gold-light); }
    .conf-chip.active { background: rgba(245, 158, 11, 0.2); border-color: var(--gold-primary); color: var(--gold-light); font-weight: 700; }
    .luxury-select {
      width: 100%;
      background: #111728;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      color: var(--text-main);
      padding: 9px 12px;
      font-size: 0.85rem;
      outline: none;
    }
    .btn-gold-action {
      width: 100%;
      margin-top: 1.3rem;
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
      color: #07090e;
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 0.98rem;
      padding: 0.9rem;
      border: none;
      border-radius: var(--radius-md);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 4px 20px var(--gold-glow);
    }
    .btn-gold-action:hover:not(:disabled) { filter: brightness(1.12); transform: translateY(-2px); }
    .btn-gold-action:disabled { opacity: 0.55; cursor: not-allowed; }
    .btn-secondary { background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-subtle); color: var(--text-main); box-shadow: none; }
    .btn-secondary:hover:not(:disabled) { background: rgba(255, 255, 255, 0.1); color: #fff; }
    .btn-danger { background: linear-gradient(135deg, #f43f5e 0%, #be123c 100%); color: #fff; }
    
    .samples-header {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-top: 1.5rem;
      margin-bottom: 0.75rem;
    }
    .samples-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .sample-thumb-card {
      aspect-ratio: 1;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-subtle);
      cursor: pointer;
      background: #121829;
    }
    .sample-thumb-card img { width: 100%; height: 100%; object-fit: cover; }
    .results-col { display: flex; flex-direction: column; gap: 1.4rem; }
    .luxury-status-banner {
      padding: 1.15rem 1.5rem;
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 0.95rem;
      font-weight: 500;
    }
    .luxury-status-banner.pass { background: var(--emerald-bg); border: 1px solid rgba(16, 185, 129, 0.4); color: #a7f3d0; }
    .luxury-status-banner.defect { background: var(--ruby-bg); border: 1px solid rgba(244, 63, 94, 0.4); color: #fecdd3; }
    .luxury-status-banner.idle { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle); color: var(--text-secondary); }
    .luxury-status-banner svg { width: 26px; height: 26px; flex-shrink: 0; }
    .stage-toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem; }
    .sub-tabs { display: flex; background: rgba(0, 0, 0, 0.3); padding: 3px; border-radius: 8px; gap: 4px; }
    .sub-tab-btn {
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-secondary);
      background: transparent;
      border: none;
      cursor: pointer;
    }
    .sub-tab-btn.active { background: var(--bg-card); color: #fff; }
    .btn-stage-action {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .stage-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; min-height: 400px; }
    .stage-grid.single-mode { grid-template-columns: 1fr; }
    @media (max-width: 860px) { .stage-grid { grid-template-columns: 1fr; } }
    .inspection-viewport {
      background: #090c14;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }
    .viewport-header {
      padding: 10px 14px;
      background: rgba(14, 18, 30, 0.9);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-secondary);
    }
    .viewport-body {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0.75rem;
      min-height: 320px;
      background: repeating-conic-gradient(#0b0f19 0% 25%, #0e1422 0% 50%) 50% / 24px 24px;
      position: relative;
      overflow: hidden;
    }
    .viewport-body img, .viewport-body video {
      max-width: 100%;
      max-height: 480px;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
      z-index: 2;
    }
    .viewfinder-reticle {
      position: absolute;
      width: 18px;
      height: 18px;
      border-color: rgba(245, 158, 11, 0.6);
      pointer-events: none;
      z-index: 5;
    }
    .reticle-tl { top: 12px; left: 12px; border-top: 2px solid; border-left: 2px solid; }
    .reticle-tr { top: 12px; right: 12px; border-top: 2px solid; border-right: 2px solid; }
    .reticle-bl { bottom: 12px; left: 12px; border-bottom: 2px solid; border-left: 2px solid; }
    .reticle-br { bottom: 12px; right: 12px; border-bottom: 2px solid; border-right: 2px solid; }
    
    .metrics-dashboard { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
    @media (max-width: 860px) { .metrics-dashboard { grid-template-columns: repeat(2, 1fr); } }
    .metric-box {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.1rem 1.25rem;
    }
    .metric-tag { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }
    .metric-number { font-family: var(--font-mono); font-size: 1.45rem; font-weight: 700; }
    .verdict-seal-badge { font-family: var(--font-display); font-size: 1.05rem; font-weight: 700; }
    
    .detections-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .detections-table th {
      text-align: left;
      padding: 9px 12px;
      background: rgba(0, 0, 0, 0.35);
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }
    .detections-table td { padding: 10px 12px; border-top: 1px solid var(--border-subtle); font-family: var(--font-mono); font-size: 0.82rem; }
    .confidence-pill {
      padding: 2px 8px;
      border-radius: 6px;
      background: rgba(245, 158, 11, 0.15);
      color: var(--gold-light);
      font-weight: 600;
      font-size: 0.75rem;
    }
    
    /* QR Sync Banner & Card */
    .qr-sync-container {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(14, 18, 30, 0.95) 100%);
      border: 1px solid var(--border-gold);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      display: flex;
      align-items: center;
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .qr-box {
      width: 130px;
      height: 130px;
      background: #0e121e;
      border-radius: 10px;
      border: 2px solid var(--border-gold);
      padding: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 15px var(--gold-glow);
      flex-shrink: 0;
    }
    .qr-box img { width: 100%; height: 100%; border-radius: 6px; }
    .qr-instructions h3 { font-family: var(--font-display); font-size: 0.98rem; color: #fff; margin-bottom: 4px; }
    .qr-instructions p { font-size: 0.78rem; color: var(--text-secondary); line-height: 1.4; }
    .qr-url-chip {
      display: inline-block;
      margin-top: 6px;
      font-size: 0.74rem;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.08);
      padding: 3px 8px;
      border-radius: 6px;
      color: var(--gold-light);
    }
    
    .live-hud-layer {
      position: absolute;
      top: 14px; left: 14px; right: 14px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
      z-index: 10;
    }
    .hud-chip {
      background: rgba(7, 9, 14, 0.8);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.78rem;
      font-weight: 600;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .pulse-alarm-border { border: 2px solid transparent !important; }
    .pulse-alarm-border.alarm-active {
      border-color: var(--ruby-defect) !important;
      box-shadow: 0 0 40px var(--ruby-glow) inset, 0 0 20px var(--ruby-glow) !important;
    }
    
    /* QA Certificate Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(10px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 1rem;
    }
    .certificate-card {
      background: #0f1422;
      border: 2px solid var(--border-gold);
      border-radius: var(--radius-lg);
      max-width: 680px;
      width: 100%;
      padding: 2.2rem;
      position: relative;
    }
    .cert-seal-watermark {
      position: absolute;
      bottom: 20px; right: 20px;
      width: 130px; height: 130px;
      border: 3px dashed rgba(245, 158, 11, 0.25);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      transform: rotate(-15deg);
      pointer-events: none;
      font-family: var(--font-display);
      font-size: 0.75rem;
      font-weight: 700;
      color: rgba(245, 158, 11, 0.4);
      text-align: center;
    }
    footer {
      border-top: 1px solid var(--border-subtle);
      background: rgba(10, 14, 24, 0.9);
      padding: 0.75rem 2rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .footer-inner {
      max-width: 1520px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .spinner {
      width: 18px; height: 18px;
      border: 2px solid rgba(0, 0, 0, 0.2);
      border-top-color: currentColor;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <!-- Header -->
  <header>
    <div class="header-inner">
      <div class="brand-wrap">
        <div class="brand-gem-icon">
          <svg viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        </div>
        <div>
          <div class="brand-title">
            AURUM VISION
            <span style="font-size: 0.68rem; padding: 2px 7px; border-radius: 4px; background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245,158,11,0.4); color: var(--gold-light);">v2.5 QC</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-secondary);">Automated Jewellery Gemstone Metrology & Anomaly Detection</p>
        </div>
      </div>

      <nav class="mode-nav">
        <button class="mode-tab-btn active" id="tabBtnImage" onclick="switchSuiteMode('image')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
          Image Inspection
        </button>
        <button class="mode-tab-btn" id="tabBtnVideo" onclick="switchSuiteMode('video')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
          Video Inspection
        </button>
        <button class="mode-tab-btn" id="tabBtnLive" onclick="switchSuiteMode('live')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"></path></svg>
          Live Optical Feed
        </button>
      </nav>

      <div class="telemetry-pills">
        <div class="pill-badge pill-cuda">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
          GTX 1650 (CUDA)
        </div>
        <div class="pill-badge pill-status">
          <span class="status-pulse-dot"></span>
          Engine Active
        </div>
      </div>
    </div>
  </header>

  <main>
    <!-- 1. IMAGE INSPECTION -->
    <div id="viewImage" class="mode-view active">
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="luxury-card">
          <div class="card-heading">
            <h2>Input Jewellery Specimen</h2>
            <span class="tag">MACRO PHOTOMETRY</span>
          </div>

          <div class="luxury-dropzone" id="imgDropzone">
            <div class="dropzone-icon-wrap">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
            </div>
            <h3>Drop Jewellery Photograph Here</h3>
            <p>Upload rings, gem settings, necklaces (JPG, PNG, WEBP)</p>
            <input type="file" id="imgFileInput" accept="image/*">
          </div>

          <div class="control-block">
            <div class="control-label-row">
              <span style="color: var(--text-secondary);">Confidence Sensitivity Threshold</span>
              <span id="imgConfValue" class="conf-badge">25%</span>
            </div>
            <input type="range" id="imgConfSlider" class="luxury-slider" min="10" max="90" value="25" step="5">
            <div class="conf-presets-row">
              <button class="conf-chip" onclick="setConfidencePreset('img', 15, this)">🔬 Micro (15%)</button>
              <button class="conf-chip active" onclick="setConfidencePreset('img', 25, this)">⚖️ Standard (25%)</button>
              <button class="conf-chip" onclick="setConfidencePreset('img', 50, this)">💎 High (50%)</button>
              <button class="conf-chip" onclick="setConfidencePreset('img', 75, this)">🛡️ Strict (75%)</button>
            </div>
          </div>

          <button id="imgRunBtn" class="btn-gold-action" disabled>
            <span id="imgBtnText">Select Specimen to Inspect</span>
          </button>

          <div class="samples-header"><span>⚡ Verified Test Samples</span></div>
          <div class="samples-grid" id="imgSamplesGrid"></div>
        </div>
      </div>

      <div class="results-col">
        <div id="imgStatusBanner" class="luxury-status-banner idle">
          <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          <span id="imgStatusText">Awaiting specimen upload. Select an image or click a sample to run detection.</span>
        </div>

        <div class="luxury-card" style="padding: 1.25rem;">
          <div class="stage-toolbar">
            <div class="sub-tabs">
              <button class="sub-tab-btn active" id="imgTabSplit" onclick="setImgView('split')">Side-by-Side</button>
              <button class="sub-tab-btn" id="imgTabAnnotated" onclick="setImgView('annotated')">Detections Only</button>
              <button class="sub-tab-btn" id="imgTabOriginal" onclick="setImgView('original')">Original Only</button>
            </div>
            <div>
              <button id="btnCertModal" class="btn-stage-action" style="display: none;" onclick="openCertModal()">QA Certificate</button>
              <button id="imgDownloadBtn" class="btn-stage-action" style="display: none;" onclick="downloadAnnotatedImage()">Export Image</button>
            </div>
          </div>

          <div class="stage-grid" id="imgStageGrid">
            <div class="inspection-viewport" id="imgOrigViewport">
              <div class="viewport-header"><span>RAW SPECIMEN</span><span id="imgOrigDim">-</span></div>
              <div class="viewport-body">
                <div class="viewfinder-reticle reticle-tl"></div><div class="viewfinder-reticle reticle-tr"></div>
                <div class="viewfinder-reticle reticle-bl"></div><div class="viewfinder-reticle reticle-br"></div>
                <div class="empty-prompt" id="imgOrigEmpty"><p>No specimen loaded</p></div>
                <img id="imgOrigPreview" style="display: none;" alt="Raw Specimen">
              </div>
            </div>

            <div class="inspection-viewport" id="imgAnnViewport">
              <div class="viewport-header"><span style="color: var(--gold-light);">YOLOV8 ANNOTATION</span><span id="imgDetCountBadge">-</span></div>
              <div class="viewport-body">
                <div class="viewfinder-reticle reticle-tl"></div><div class="viewfinder-reticle reticle-tr"></div>
                <div class="viewfinder-reticle reticle-bl"></div><div class="viewfinder-reticle reticle-br"></div>
                <div class="empty-prompt" id="imgAnnEmpty"><p>Defect overlay will appear here</p></div>
                <img id="imgAnnPreview" style="display: none;" alt="YOLOv8 Detection">
              </div>
            </div>
          </div>
        </div>

        <div class="metrics-dashboard">
          <div class="metric-box"><div class="metric-tag">Anomalies Detected</div><div class="metric-number" id="imgMetricDefects" style="color: var(--text-muted);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Peak Confidence</div><div class="metric-number" id="imgMetricConfidence" style="color: var(--gold-light);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Inference Latency</div><div class="metric-number" id="imgMetricSpeed" style="color: var(--ice-telemetry);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Metrology Verdict</div><div class="verdict-seal-badge" id="imgMetricVerdict" style="color: var(--text-muted);">-</div></div>
        </div>

        <div class="luxury-card" id="imgDetailsCard" style="display: none;">
          <div class="card-heading"><h2>Spatial Coordinate Telemetry</h2><span class="tag">BOUNDING BOXES</span></div>
          <table class="detections-table">
            <thead><tr><th>#</th><th>Classification</th><th>Confidence</th><th>Bounding Box</th></tr></thead>
            <tbody id="imgDetectionsTbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 2. VIDEO INSPECTION -->
    <div id="viewVideo" class="mode-view">
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="luxury-card">
          <div class="card-heading"><h2>Sequential Video Stream</h2><span class="tag">CONVEYOR QC</span></div>
          <div class="luxury-dropzone" id="vidDropzone">
            <div class="dropzone-icon-wrap">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
            </div>
            <h3 id="vidDropTitle">Upload Inspection Video</h3>
            <p id="vidDropSub">Drop video file or click to browse (MP4, AVI, MOV, WEBM)</p>
            <input type="file" id="vidFileInput" accept="video/*" style="display: none;">
          </div>
          <button class="btn-gold-action btn-secondary" style="margin-top: 0.9rem;" onclick="loadDemoVideo()">
            ⚡ Load Pre-configured Jewellery QC Demo Clip
          </button>
          <div class="control-block">
            <div class="control-label-row"><span>Video Sensitivity</span><span id="vidConfValue" class="conf-badge">25%</span></div>
            <input type="range" id="vidConfSlider" class="luxury-slider" min="10" max="90" value="25" step="5">
          </div>
          <div class="control-block">
            <div class="control-label-row"><span>Frame Stride</span></div>
            <select id="vidFrameSkip" class="luxury-select">
              <option value="1">Comprehensive (100% Frames)</option>
              <option value="2">Optimized 2x (Alternate Frames)</option>
              <option value="3">High-Speed 3x (Every 3rd Frame)</option>
            </select>
          </div>
          <button id="vidRunBtn" class="btn-gold-action" disabled onclick="startVideoProcessing()">
            <span id="vidBtnText">Select Video to Inspect</span>
          </button>
        </div>
      </div>

      <div class="results-col">
        <div id="vidStatusBanner" class="luxury-status-banner idle">
          <span>Select a video or click 'Load Demo Clip' to analyze continuous jewellery rotation for missing gems.</span>
        </div>
        <div class="luxury-card" style="padding: 1.25rem;">
          <div class="stage-grid" id="vidStageGrid">
            <div class="inspection-viewport">
              <div class="viewport-header"><span>ORIGINAL RECORDING</span></div>
              <div class="viewport-body"><video id="vidOrigPlayer" controls style="display: none;" playsinline></video></div>
            </div>
            <div class="inspection-viewport">
              <div class="viewport-header"><span style="color: var(--gold-light);">YOLOV8 ANNOTATED OUTPUT</span></div>
              <div class="viewport-body"><video id="vidAnnPlayer" controls style="display: none;" playsinline></video></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 3. LIVE OPTICAL FEED (FEATURING PHONE QR CODE SYNC!) -->
    <div id="viewLive" class="mode-view">
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="luxury-card">
          <div class="card-heading">
            <h2>Optical Sensor Source</h2>
            <span class="pill-badge pill-status" id="liveCameraStatus">
              <span class="status-pulse-dot"></span> Ready
            </span>
          </div>

          <!-- Camera Source Selector -->
          <div class="control-block" style="margin-top: 0;">
            <div class="control-label-row"><span>Select Camera Input Source</span></div>
            <select id="liveCameraSource" class="luxury-select" onchange="onCameraSourceChange()">
              <option value="phone_qr">📱 Wireless Phone Camera (QR Code 30 FPS Stream)</option>
              <option value="usb_phone">🔌 Phone via USB Cable (Iriun / DroidCam 1080p 60 FPS)</option>
              <option value="browser">🌐 Client Laptop / Browser Webcam</option>
              <option value="ip_camera">📡 External IP Camera App (DroidCam / RTSP)</option>
              <option value="server">🖥️ Host USB Hardware Camera (Index 0)</option>
            </select>
          </div>

          <!-- USB Phone Cable Mode Panel -->
          <div id="usbPhonePanel" class="qr-sync-container" style="display: none;">
            <div style="flex: 1;">
              <h3 style="color: var(--gold-light); font-size: 0.95rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                <span>🔌 Zero-Lag USB Cable Mode (Guaranteed 60 FPS)</span>
              </h3>
              <p style="font-size: 0.77rem; color: var(--text-secondary); line-height: 1.45;">
                For the smoothest offline experience without Wi-Fi jitter:<br>
                1. Install free <strong>Iriun Webcam</strong> or <strong>DroidCam</strong> on your PC & phone.<br>
                2. Connect your iPhone to your laptop using a USB Lightning / Type-C cable.<br>
                3. Click <strong>"Activate Live Optical Feed"</strong> below for instant 1080p 60 FPS metrology!
              </p>
            </div>
          </div>

          <!-- Phone QR Sync Panel -->
          <div id="phoneQrPanel" class="qr-sync-container">
            <div class="qr-box">
              <img id="qrCodeImg" src="/api/qr_code?mode=https" alt="Scan QR Code">
            </div>
            <div class="qr-instructions" style="flex: 1;">
              <div style="display: flex; gap: 6px; margin-bottom: 8px;">
                <button id="qrModeHttpsBtn" class="conf-chip active" style="flex: 1;" onclick="setQrStreamMode('https')">⚡ HTTPS Live (iOS & Android)</button>
                <button id="qrModeHttpBtn" class="conf-chip" style="flex: 1;" onclick="setQrStreamMode('http')">📸 HTTP Snap Mode</button>
              </div>
              <p id="qrHelpText" style="font-size: 0.77rem; color: var(--text-secondary); line-height: 1.45;">
                1. Scan with iPhone or Android camera.<br>
                2. On iPhone Safari: tap <strong>"Show Details" &rarr; "Visit this website"</strong> once to authorize camera.<br>
                3. Continuous 30 FPS stream syncs to this screen!
              </p>
              <div style="display: flex; align-items: center; gap: 8px; margin-top: 6px;">
                <div class="qr-url-chip" id="qrUrlText" style="margin-top: 0;">https://192.168.1.6:8443/mobile</div>
                <a id="qrDirectLink" href="/mobile" target="_blank" style="font-size: 0.72rem; color: var(--gold-light); text-decoration: none;">↗ Open</a>
              </div>
            </div>
          </div>

          <!-- External IP Camera URL Input Panel -->
          <div id="ipCamPanel" class="control-block" style="display: none;">
            <div class="control-label-row">
              <span>External RTSP / IP Camera URL</span>
              <span style="font-size: 0.72rem; color: var(--gold-light);">iOS & Android Supported</span>
            </div>
            <input type="text" id="ipCamUrl" class="luxury-select" placeholder="rtsp://192.168.1.50:8554/live" value="rtsp://192.168.1.6:8554/live">
            <div style="display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap;">
              <button type="button" class="conf-chip" onclick="setIpCamUrlPreset('rtsp')">📱 iOS RTSP (IP Camera Lite)</button>
              <button type="button" class="conf-chip" onclick="setIpCamUrlPreset('droidcam')">📹 DroidCam (Port 4747)</button>
              <button type="button" class="conf-chip" onclick="setIpCamUrlPreset('android')">🤖 Android IP Webcam</button>
            </div>
            <p style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4; margin-top: 8px;">
              <strong>iOS RTSP Apps:</strong> Install <em>"IP Camera Lite"</em> or <em>"Live-Reporter"</em> (free on iOS App Store). Start the broadcast in the app and paste the displayed RTSP URL above!
            </p>
            <button id="btnConnectIp" class="btn-gold-action btn-secondary" style="margin-top: 10px;" onclick="connectIpCamera()">Connect IP Stream</button>
          </div>

          <!-- Real-Time Confidence Slider -->
          <div class="control-block">
            <div class="control-label-row">
              <span>Real-Time Sensitivity</span>
              <span id="liveConfValue" class="conf-badge">25%</span>
            </div>
            <input type="range" id="liveConfSlider" class="luxury-slider" min="10" max="90" value="25" step="5">
            <div class="conf-presets-row">
              <button class="conf-chip" onclick="setConfidencePreset('live', 15, this)">🔬 Micro (15%)</button>
              <button class="conf-chip active" onclick="setConfidencePreset('live', 25, this)">⚖️ Standard (25%)</button>
              <button class="conf-chip" onclick="setConfidencePreset('live', 50, this)">💎 High (50%)</button>
              <button class="conf-chip" onclick="setConfidencePreset('live', 75, this)">🛡️ Strict (75%)</button>
            </div>
          </div>

          <!-- Action Toggle Button -->
          <button id="liveToggleBtn" class="btn-gold-action" onclick="toggleLiveCamera()">
            <span id="liveToggleText">Activate Live Optical Feed</span>
          </button>
        </div>
      </div>

      <!-- Right Column: Live Viewport -->
      <div class="results-col">
        <div id="liveStatusBanner" class="luxury-status-banner idle">
          <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19 4h-3.5L14 2H10L8.5 4H5a2 2 0 00-2 2v12a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z"/></svg>
          <span id="liveStatusText">Scan the QR code with your phone or click 'Activate Live Optical Feed' to begin continuous anomaly inspection.</span>
        </div>

        <div class="luxury-card" style="padding: 1.25rem;">
          <div class="inspection-viewport pulse-alarm-border" id="liveDisplayPanel" style="min-height: 440px;">
            <div class="viewport-header">
              <span style="color: var(--gold-light);">REAL-TIME METROLOGY FEED</span>
              <span id="liveFeedInfo">FEED STANDBY</span>
            </div>

            <div class="viewport-body" style="min-height: 420px;">
              <div class="viewfinder-reticle reticle-tl"></div><div class="viewfinder-reticle reticle-tr"></div>
              <div class="viewfinder-reticle reticle-bl"></div><div class="viewfinder-reticle reticle-br"></div>

              <div class="live-hud-layer" id="liveHud" style="display: none;">
                <div class="hud-chip" id="liveVerdictBadge" style="color: #34d399;">
                  <span class="status-pulse-dot"></span>
                  <span id="liveVerdictBadgeText">FLAWLESS SPECIMEN</span>
                </div>
                <div class="hud-chip" style="color: var(--gold-light); font-family: var(--font-mono);">
                  <span id="liveFpsText">0.0 FPS</span> | <span id="liveLatencyText">0 ms</span>
                </div>
              </div>

              <div class="empty-prompt" id="liveEmpty">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"></path></svg>
                <p id="liveEmptySub">Scan the QR code with your phone to stream live video</p>
              </div>

              <video id="liveVideoElement" playsinline autoplay muted style="display: none;"></video>
              <canvas id="liveOffscreenCanvas" style="display: none;"></canvas>
              <img id="liveDisplayImg" style="display: none;" alt="Live Neural Feed">
            </div>
          </div>
        </div>

        <div class="metrics-dashboard">
          <div class="metric-box"><div class="metric-tag">Active In-Frame Defects</div><div class="metric-number" id="liveMetricDefects" style="color: var(--text-muted);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Peak Detection Confidence</div><div class="metric-number" id="liveMetricConfidence" style="color: var(--gold-light);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Sensor Latency</div><div class="metric-number" id="liveMetricLatency" style="color: var(--ice-telemetry);">-</div></div>
          <div class="metric-box"><div class="metric-tag">Stream Frame Rate</div><div class="metric-number" id="liveMetricFps" style="color: #34d399;">-</div></div>
        </div>
      </div>
    </div>
  </main>

  <!-- Certificate Modal -->
  <div class="modal-overlay" id="certificateModal" onclick="closeCertModal(event)">
    <div class="certificate-card" onclick="event.stopPropagation()">
      <div class="cert-seal-watermark">AURUM QC<br>OFFICIAL AUDIT<br>PASSED</div>
      <div style="border-bottom: 2px solid var(--border-gold); padding-bottom: 1rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between;">
        <div>
          <h2 style="font-family: var(--font-display); font-size: 1.5rem; color: #fff;">QUALITY INSPECTION CERTIFICATE</h2>
          <p style="font-size: 0.8rem; color: var(--gold-light); font-family: var(--font-mono);">AI Metrology Quality Audit Report</p>
        </div>
        <button onclick="closeCertModal()" style="background: none; border: none; color: #fff; font-size: 1.5rem; cursor: pointer;">&times;</button>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.85rem; margin-bottom: 1.5rem;">
        <div><span style="color: var(--text-muted);">Inspection ID:</span> <strong id="certId" style="color: #fff;">AURUM-8921</strong></div>
        <div><span style="color: var(--text-muted);">Timestamp:</span> <strong id="certTime" style="color: #fff;">-</strong></div>
        <div><span style="color: var(--text-muted);">Model Spec:</span> <strong style="color: var(--ice-telemetry);">YOLOv8 Nano (GTX 1650)</strong></div>
        <div><span style="color: var(--text-muted);">Sensitivity:</span> <strong id="certConf" style="color: var(--gold-light);">25%</strong></div>
      </div>
      <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1.2rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <div><div style="font-size: 0.72rem; color: var(--text-muted);">OFFICIAL VERDICT</div><div id="certVerdict" style="font-size: 1.4rem; font-weight: 700;">-</div></div>
        <div style="text-align: right;"><div style="font-size: 0.72rem; color: var(--text-muted);">DEFECTS DETECTED</div><div id="certDefectCount" style="font-size: 1.4rem; font-weight: 700;">-</div></div>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 10px;">
        <button class="btn-stage-action" onclick="window.print()">Print Certificate</button>
        <button class="btn-stage-action" onclick="closeCertModal()">Close</button>
      </div>
    </div>
  </div>

  <footer>
    <div class="footer-inner">
      <div><span>ARCHITECTURE:</span> <strong>Ultralytics YOLOv8</strong> | <span>RUNTIME:</span> <strong>PyTorch 2.6.0 + CUDA 12.4</strong></div>
      <div><span>DATASET:</span> <strong>671 Images</strong> | <span>WEIGHTS:</span> <strong>best.pt</strong></div>
    </div>
  </footer>

  <script>
    let activeSuiteMode = 'image';
    let currentImageBase64 = null;
    let currentAnnotatedImageBase64 = null;
    let lastImageResults = null;

    // Mobile Phone Wireless Sync Polling State
    let phonePollTimer = null;
    let isPhoneStreaming = false;

    // Elements
    const imgDropzone = document.getElementById('imgDropzone');
    const imgFileInput = document.getElementById('imgFileInput');
    const imgConfSlider = document.getElementById('imgConfSlider');
    const imgConfValue = document.getElementById('imgConfValue');
    const imgRunBtn = document.getElementById('imgRunBtn');
    const imgBtnText = document.getElementById('imgBtnText');
    const imgSamplesGrid = document.getElementById('imgSamplesGrid');
    const imgOrigPreview = document.getElementById('imgOrigPreview');
    const imgOrigEmpty = document.getElementById('imgOrigEmpty');
    const imgOrigDim = document.getElementById('imgOrigDim');
    const imgAnnPreview = document.getElementById('imgAnnPreview');
    const imgAnnEmpty = document.getElementById('imgAnnEmpty');
    const imgDetCountBadge = document.getElementById('imgDetCountBadge');
    const imgDownloadBtn = document.getElementById('imgDownloadBtn');
    const btnCertModal = document.getElementById('btnCertModal');

    const imgStatusBanner = document.getElementById('imgStatusBanner');
    const imgStatusText = document.getElementById('imgStatusText');
    const imgMetricDefects = document.getElementById('imgMetricDefects');
    const imgMetricConfidence = document.getElementById('imgMetricConfidence');
    const imgMetricSpeed = document.getElementById('imgMetricSpeed');
    const imgMetricVerdict = document.getElementById('imgMetricVerdict');
    const imgDetailsCard = document.getElementById('imgDetailsCard');
    const imgDetectionsTbody = document.getElementById('imgDetectionsTbody');

    // Navigation
    function switchSuiteMode(mode) {
      if (activeSuiteMode === 'live' && mode !== 'live') stopLiveCamera();
      activeSuiteMode = mode;
      document.querySelectorAll('.mode-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.mode-view').forEach(v => v.classList.remove('active'));

      if (mode === 'image') {
        document.getElementById('tabBtnImage').classList.add('active');
        document.getElementById('viewImage').classList.add('active');
      } else if (mode === 'video') {
        document.getElementById('tabBtnVideo').classList.add('active');
        document.getElementById('viewVideo').classList.add('active');
      } else if (mode === 'live') {
        document.getElementById('tabBtnLive').classList.add('active');
        document.getElementById('viewLive').classList.add('active');
        fetchNetworkInfo();
      }
    }

    let currentNetworkInfo = null;
    let currentQrStreamMode = 'https';

    async function fetchNetworkInfo() {
      try {
        const res = await fetch('/api/network_info');
        currentNetworkInfo = await res.json();
        updateQrDisplay();
      } catch (e) {
        console.warn('Network info error:', e);
      }
    }

    function setQrStreamMode(mode) {
      currentQrStreamMode = mode;
      const btnHttps = document.getElementById('qrModeHttpsBtn');
      const btnHttp = document.getElementById('qrModeHttpBtn');
      if (btnHttps && btnHttp) {
        btnHttps.classList.toggle('active', mode === 'https');
        btnHttp.classList.toggle('active', mode === 'http');
      }
      updateQrDisplay();
    }

    function updateQrDisplay() {
      if (!currentNetworkInfo) return;
      const mode = currentQrStreamMode;
      const targetUrl = mode === 'https' ? currentNetworkInfo.https_mobile_url : currentNetworkInfo.http_mobile_url;
      document.getElementById('qrUrlText').textContent = targetUrl;
      const directLink = document.getElementById('qrDirectLink');
      if (directLink) directLink.href = targetUrl;
      document.getElementById('qrCodeImg').src = `/api/qr_code?mode=${mode}&t=` + Date.now();

      const help = document.getElementById('qrHelpText');
      if (help) {
        if (mode === 'https') {
          help.innerHTML = '1. Scan with iPhone or Android camera.<br>2. On iPhone Safari: tap <strong>"Show Details" &rarr; "Visit this website"</strong> once to authorize camera.<br>3. Continuous 30 FPS stream syncs to this screen!';
        } else {
          help.innerHTML = '1. Scan with phone to open HTTP Snap QC.<br>2. Tap <strong>"Snap & Inspect Specimen"</strong> to take high-res camera photo.<br>3. Zero warnings, 100% reliable instant metrology!';
        }
      }
    }

    function setConfidencePreset(mode, val, el) {
      const p = el.parentElement;
      p.querySelectorAll('.conf-chip').forEach(c => c.classList.remove('active'));
      el.classList.add('active');
      if (mode === 'img') {
        imgConfSlider.value = val;
        imgConfValue.textContent = val + '%';
        if (currentImageBase64) runImageInference();
      } else if (mode === 'live') {
        liveConfSlider.value = val;
        liveConfValue.textContent = val + '%';
      }
    }

    // Image logic
    imgConfSlider.addEventListener('input', (e) => {
      imgConfValue.textContent = e.target.value + '%';
      if (currentImageBase64) runImageInference();
    });

    imgDropzone.addEventListener('click', () => imgFileInput.click());
    imgDropzone.addEventListener('dragover', (e) => { e.preventDefault(); imgDropzone.classList.add('dragover'); });
    imgDropzone.addEventListener('dragleave', () => imgDropzone.classList.remove('dragover'));
    imgDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      imgDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) handleImg(e.dataTransfer.files[0]);
    });
    imgFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) handleImg(e.target.files[0]);
    });

    function handleImg(file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        imgOrigPreview.src = e.target.result;
        imgOrigPreview.style.display = 'block';
        imgOrigEmpty.style.display = 'none';
        imgRunBtn.disabled = false;
        imgBtnText.textContent = 'Run Anomaly Detection';
        runImageInference();
      };
      reader.readAsDataURL(file);
    }

    async function runImageInference() {
      if (!currentImageBase64) return;
      imgBtnText.innerHTML = '<span class="spinner"></span> Scanning...';
      imgRunBtn.disabled = true;
      const conf = parseFloat(imgConfSlider.value) / 100.0;
      try {
        const resp = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: currentImageBase64, conf: conf })
        });
        const data = await resp.json();
        lastImageResults = data;
        currentAnnotatedImageBase64 = data.annotated_image;
        imgAnnPreview.src = data.annotated_image;
        imgAnnPreview.style.display = 'block';
        imgAnnEmpty.style.display = 'none';
        imgDownloadBtn.style.display = 'inline-flex';
        btnCertModal.style.display = 'inline-flex';

        const count = data.detections_count;
        imgDetCountBadge.textContent = `${count} DETECTIONS`;
        imgMetricDefects.textContent = count;
        imgMetricSpeed.textContent = `${data.inference_time_ms.toFixed(1)} ms`;

        if (count > 0) {
          imgStatusBanner.className = 'luxury-status-banner defect';
          imgStatusBanner.innerHTML = `<div><strong>ANOMALY IDENTIFIED:</strong> Found <strong>${count} Missing Gem defect(s)</strong> in setting.</div>`;
          imgMetricDefects.style.color = 'var(--ruby-defect)';
          imgMetricVerdict.textContent = 'DEFECTIVE';
          imgMetricVerdict.style.color = 'var(--ruby-defect)';
          imgDetectionsTbody.innerHTML = '';
          data.detections.forEach((d, idx) => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>#${idx + 1}</td><td style="color:#fda4af;">${d.class_name.toUpperCase()}</td><td><span class="confidence-pill">${(d.confidence * 100).toFixed(1)}%</span></td><td>[${d.bbox.map(Math.round).join(', ')}]</td>`;
            imgDetectionsTbody.appendChild(row);
          });
          imgDetailsCard.style.display = 'block';
        } else {
          imgStatusBanner.className = 'luxury-status-banner pass';
          imgStatusBanner.innerHTML = `<div><strong>QUALITY VERIFIED:</strong> Specimen passed inspection. Zero missing gem defects found!</div>`;
          imgMetricDefects.style.color = 'var(--emerald-pass)';
          imgMetricVerdict.textContent = 'FLAWLESS';
          imgMetricVerdict.style.color = 'var(--emerald-pass)';
          imgDetailsCard.style.display = 'none';
        }
      } catch (err) {
        imgStatusBanner.className = 'luxury-status-banner defect';
        imgStatusBanner.textContent = 'Error: ' + err.message;
      } finally {
        imgRunBtn.disabled = false;
        imgBtnText.textContent = 'Re-Evaluate Specimen';
      }
    }

    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        imgSamplesGrid.innerHTML = '';
        samples.forEach(s => {
          const div = document.createElement('div');
          div.className = 'sample-thumb-card';
          div.innerHTML = `<img src="/sample_image/${encodeURIComponent(s)}" alt="Sample Specimen">`;
          div.addEventListener('click', () => {
            fetch(`/sample_image/${encodeURIComponent(s)}`)
              .then(r => r.blob())
              .then(blob => {
                const reader = new FileReader();
                reader.onload = (e) => {
                  currentImageBase64 = e.target.result;
                  imgOrigPreview.src = e.target.result;
                  imgOrigPreview.style.display = 'block';
                  imgOrigEmpty.style.display = 'none';
                  imgRunBtn.disabled = false;
                  runImageInference();
                };
                reader.readAsDataURL(blob);
              });
          });
          imgSamplesGrid.appendChild(div);
        });
      } catch (e) {}
    }
    loadSamples();

    function setImgView(mode) {
      document.querySelectorAll('#viewImage .sub-tab-btn').forEach(t => t.classList.remove('active'));
      const stage = document.getElementById('imgStageGrid');
      const orig = document.getElementById('imgOrigViewport');
      const ann = document.getElementById('imgAnnViewport');
      if (mode === 'split') {
        document.getElementById('imgTabSplit').classList.add('active');
        stage.classList.remove('single-mode');
        orig.style.display = 'flex';
        ann.style.display = 'flex';
      } else if (mode === 'annotated') {
        document.getElementById('imgTabAnnotated').classList.add('active');
        stage.classList.add('single-mode');
        orig.style.display = 'none';
        ann.style.display = 'flex';
      } else {
        document.getElementById('imgTabOriginal').classList.add('active');
        stage.classList.add('single-mode');
        orig.style.display = 'flex';
        ann.style.display = 'none';
      }
    }

    function openCertModal() {
      if (!lastImageResults) return;
      document.getElementById('certId').textContent = 'AURUM-' + Math.random().toString(36).substring(2, 8).toUpperCase();
      document.getElementById('certTime').textContent = new Date().toLocaleString();
      document.getElementById('certConf').textContent = imgConfSlider.value + '%';
      const count = lastImageResults.detections_count;
      document.getElementById('certDefectCount').textContent = count;
      const v = document.getElementById('certVerdict');
      if (count > 0) {
        v.textContent = 'REJECTED - DEFECTS FOUND';
        v.style.color = 'var(--ruby-defect)';
      } else {
        v.textContent = 'APPROVED - FLAWLESS PASS';
        v.style.color = 'var(--emerald-pass)';
      }
      document.getElementById('certificateModal').style.display = 'flex';
    }
    function closeCertModal(e) {
      if (e && e.target !== document.getElementById('certificateModal')) return;
      document.getElementById('certificateModal').style.display = 'none';
    }

    // ==========================================
    // 2. VIDEO INSPECTION LOGIC
    // ==========================================
    let selectedVideoFile = null;
    let isVideoDemoMode = false;
    let videoPollTimer = null;

    const vidDropzone = document.getElementById('vidDropzone');
    const vidFileInput = document.getElementById('vidFileInput');
    const vidDropTitle = document.getElementById('vidDropTitle');
    const vidDropSub = document.getElementById('vidDropSub');
    const vidConfSlider = document.getElementById('vidConfSlider');
    const vidConfValue = document.getElementById('vidConfValue');
    const vidFrameSkip = document.getElementById('vidFrameSkip');
    const vidRunBtn = document.getElementById('vidRunBtn');
    const vidBtnText = document.getElementById('vidBtnText');
    const vidStatusBanner = document.getElementById('vidStatusBanner');
    const vidOrigPlayer = document.getElementById('vidOrigPlayer');
    const vidAnnPlayer = document.getElementById('vidAnnPlayer');

    if (vidConfSlider && vidConfValue) {
      vidConfSlider.addEventListener('input', (e) => {
        vidConfValue.textContent = e.target.value + '%';
      });
    }

    if (vidDropzone && vidFileInput) {
      vidDropzone.addEventListener('click', () => vidFileInput.click());
      vidDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        vidDropzone.classList.add('dragover');
      });
      vidDropzone.addEventListener('dragleave', () => {
        vidDropzone.classList.remove('dragover');
      });
      vidDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        vidDropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          handleVideoSelect(e.dataTransfer.files[0]);
        }
      });
      vidFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleVideoSelect(e.target.files[0]);
        }
      });
    }

    function handleVideoSelect(file) {
      selectedVideoFile = file;
      isVideoDemoMode = false;
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      vidDropTitle.textContent = file.name;
      vidDropSub.textContent = `${sizeMb} MB — Click or drop to select another video`;

      // Preview raw video
      try {
        const fileUrl = URL.createObjectURL(file);
        vidOrigPlayer.src = fileUrl;
        vidOrigPlayer.style.display = 'block';
        vidOrigPlayer.load();
      } catch (e) {
        console.warn('Could not create video preview URL:', e);
      }

      vidAnnPlayer.style.display = 'none';
      vidAnnPlayer.src = '';
      vidRunBtn.disabled = false;
      vidBtnText.textContent = 'Run Video Anomaly Inspection';

      vidStatusBanner.className = 'luxury-status-banner idle';
      vidStatusBanner.innerHTML = `<span>Video <strong>${file.name}</strong> loaded (${sizeMb} MB). Click 'Run Video Anomaly Inspection' to begin YOLOv8 metrology.</span>`;
    }

    function loadDemoVideo() {
      selectedVideoFile = null;
      isVideoDemoMode = true;
      vidOrigPlayer.src = '/api/demo_video';
      vidOrigPlayer.style.display = 'block';
      vidOrigPlayer.load();

      vidDropTitle.textContent = 'sample_jewellery_demo.mp4';
      vidDropSub.textContent = 'Pre-configured Rotating Jewellery Specimen (4.3 MB)';

      vidAnnPlayer.style.display = 'none';
      vidAnnPlayer.src = '';
      vidRunBtn.disabled = false;
      vidBtnText.textContent = 'Start Video QC Inference';

      vidStatusBanner.className = 'luxury-status-banner idle';
      vidStatusBanner.innerHTML = '<span>Pre-configured Jewellery QC Demo Clip loaded. Click below to begin frame-by-frame neural audit.</span>';
    }

    async function startVideoProcessing() {
      const conf = parseFloat(vidConfSlider.value) / 100.0;
      const skip = vidFrameSkip.value;
      vidRunBtn.disabled = true;
      vidBtnText.innerHTML = '<span class="spinner"></span> Processing Video...';

      if (videoPollTimer) clearInterval(videoPollTimer);

      try {
        let taskId = null;

        if (selectedVideoFile) {
          vidStatusBanner.className = 'luxury-status-banner idle';
          vidStatusBanner.innerHTML = `<div><span class="spinner"></span> <strong>Uploading video file (${(selectedVideoFile.size / (1024*1024)).toFixed(1)} MB)...</strong></div>`;

          const res = await fetch(`/api/video/upload?conf=${conf}&skip=${skip}`, {
            method: 'POST',
            body: selectedVideoFile
          });
          const data = await res.json();
          if (!data.success) {
            throw new Error(data.error || 'Failed to upload video');
          }
          taskId = data.task_id;
        } else if (isVideoDemoMode) {
          vidStatusBanner.className = 'luxury-status-banner idle';
          vidStatusBanner.innerHTML = '<div><span class="spinner"></span> <strong>Initializing GPU inference pipeline on demo video...</strong></div>';

          const res = await fetch(`/api/video/process_sample?conf=${conf}&skip=${skip}`, { method: 'POST' });
          const data = await res.json();
          if (!data.success) {
            throw new Error(data.error || 'Failed to initialize sample video processing');
          }
          taskId = data.task_id;
        } else {
          alert('Please select a video file or load the demo clip first.');
          vidRunBtn.disabled = false;
          vidBtnText.textContent = 'Select Video to Inspect';
          return;
        }

        // Poll video processing status
        videoPollTimer = setInterval(async () => {
          try {
            const resp = await fetch(`/api/video/status/${taskId}`);
            const task = await resp.json();

            if (task.status === 'processing') {
              const pct = (task.progress || 0).toFixed(0);
              const fps = task.fps || 0;
              const cur = task.current_frame || 0;
              const total = task.total_frames || '?';
              const defects = task.defects_count || 0;

              vidStatusBanner.className = 'luxury-status-banner idle';
              vidStatusBanner.innerHTML = `<div><strong>ANALYZING VIDEO:</strong> ${pct}% complete &bull; Frame ${cur}/${total} &bull; ${fps} FPS &bull; ${defects} anomaly instance(s) found</div>`;
            } else if (task.status === 'done') {
              clearInterval(videoPollTimer);
              const r = task.results;

              vidAnnPlayer.src = `/api/video/stream/${taskId}`;
              vidAnnPlayer.style.display = 'block';
              vidAnnPlayer.load();
              vidAnnPlayer.play().catch(() => {});

              vidRunBtn.disabled = false;
              vidBtnText.textContent = 'Re-Analyze Video';

              if (r.total_defects > 0) {
                vidStatusBanner.className = 'luxury-status-banner defect';
                vidStatusBanner.innerHTML = `<div><strong>VIDEO AUDIT COMPLETED:</strong> Detected <strong>${r.total_defects} defect(s)</strong> across ${r.defect_frames} defective frame(s) (${r.defect_rate_percent}% defect rate). Processed at ${r.avg_fps} FPS (${r.avg_inference_time_ms} ms/frame).</div>`;
              } else {
                vidStatusBanner.className = 'luxury-status-banner pass';
                vidStatusBanner.innerHTML = `<div><strong>SPECIMEN APPROVED:</strong> 100% of video frames passed metrology criteria with zero defects detected (${r.avg_fps} FPS).</div>`;
              }
            } else if (task.status === 'error') {
              clearInterval(videoPollTimer);
              vidStatusBanner.className = 'luxury-status-banner defect';
              vidStatusBanner.textContent = 'Processing error: ' + (task.error || 'Unknown error');
              vidRunBtn.disabled = false;
              vidBtnText.textContent = 'Retry Video Analysis';
            }
          } catch (err) {
            console.warn('Status poll error:', err);
          }
        }, 500);

      } catch (err) {
        vidStatusBanner.className = 'luxury-status-banner defect';
        vidStatusBanner.textContent = 'Error: ' + err.message;
        vidRunBtn.disabled = false;
        vidBtnText.textContent = 'Retry Video Analysis';
      }
    }

    // ==========================================
    // 3. LIVE OPTICAL FEED WITH PHONE QR SYNC
    // ==========================================
    let isLiveActive = false;
    let liveMediaStream = null;
    let liveFrameLoopId = null;

    const liveCameraSource = document.getElementById('liveCameraSource');
    const phoneQrPanel = document.getElementById('phoneQrPanel');
    const ipCamPanel = document.getElementById('ipCamPanel');
    const liveConfSlider = document.getElementById('liveConfSlider');
    const liveConfValue = document.getElementById('liveConfValue');
    const liveToggleBtn = document.getElementById('liveToggleBtn');
    const liveToggleText = document.getElementById('liveToggleText');
    const liveDisplayPanel = document.getElementById('liveDisplayPanel');
    const liveHud = document.getElementById('liveHud');
    const liveVerdictBadge = document.getElementById('liveVerdictBadge');
    const liveVerdictBadgeText = document.getElementById('liveVerdictBadgeText');
    const liveFpsText = document.getElementById('liveFpsText');
    const liveLatencyText = document.getElementById('liveLatencyText');
    const liveFeedInfo = document.getElementById('liveFeedInfo');
    const liveEmpty = document.getElementById('liveEmpty');
    const liveVideoElement = document.getElementById('liveVideoElement');
    const liveOffscreenCanvas = document.getElementById('liveOffscreenCanvas');
    const liveDisplayImg = document.getElementById('liveDisplayImg');
    const liveStatusBanner = document.getElementById('liveStatusBanner');
    const liveStatusText = document.getElementById('liveStatusText');
    const liveMetricDefects = document.getElementById('liveMetricDefects');
    const liveMetricConfidence = document.getElementById('liveMetricConfidence');
    const liveMetricLatency = document.getElementById('liveMetricLatency');
    const liveMetricFps = document.getElementById('liveMetricFps');

    liveConfSlider.addEventListener('input', (e) => {
      liveConfValue.textContent = e.target.value + '%';
    });

    const usbPhonePanel = document.getElementById('usbPhonePanel');

    function onCameraSourceChange() {
      const src = liveCameraSource.value;
      if (src === 'phone_qr') {
        phoneQrPanel.style.display = 'flex';
        if (usbPhonePanel) usbPhonePanel.style.display = 'none';
        ipCamPanel.style.display = 'none';
        document.getElementById('liveEmptySub').textContent = 'Scan the QR code with your phone to stream live video';
      } else if (src === 'usb_phone') {
        phoneQrPanel.style.display = 'none';
        if (usbPhonePanel) usbPhonePanel.style.display = 'flex';
        ipCamPanel.style.display = 'none';
        document.getElementById('liveEmptySub').textContent = 'Connect phone via USB cable and click Activate below';
      } else if (src === 'ip_camera') {
        phoneQrPanel.style.display = 'none';
        if (usbPhonePanel) usbPhonePanel.style.display = 'none';
        ipCamPanel.style.display = 'block';
        document.getElementById('liveEmptySub').textContent = 'Enter external IP Camera RTSP / HTTP URL';
      } else {
        phoneQrPanel.style.display = 'none';
        if (usbPhonePanel) usbPhonePanel.style.display = 'none';
        ipCamPanel.style.display = 'none';
        document.getElementById('liveEmptySub').textContent = 'Click Activate Live Optical Feed';
      }
      if (isLiveActive) stopLiveCamera();
    }

    function toggleLiveCamera() {
      if (isLiveActive) {
        stopLiveCamera();
      } else {
        startLiveCamera();
      }
    }

    async function startLiveCamera() {
      const src = liveCameraSource.value;

      if (src === 'phone_qr') {
        // Start Wireless Phone Stream Listener (30 FPS MJPEG)
        isLiveActive = true;
        isPhoneStreaming = true;
        liveEmpty.style.display = 'none';
        liveDisplayImg.style.display = 'block';
        liveHud.style.display = 'flex';
        liveFeedInfo.textContent = 'WIRELESS PHONE CAMERA FEED (30 FPS MJPEG)';
        liveToggleBtn.className = 'btn-gold-action btn-danger';
        liveToggleText.textContent = 'Terminate Wireless Feed';

        liveStatusBanner.className = 'luxury-status-banner idle';
        liveStatusText.textContent = 'Awaiting incoming frame packets from mobile phone scanner...';

        startPhoneStreamListener();
      } else if (src === 'usb_phone' || src === 'browser') {
        // Laptop Webcam or USB Phone Webcam (e.g. Iriun / DroidCam via USB)
        try {
          liveMediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
          });
          liveVideoElement.srcObject = liveMediaStream;
          await liveVideoElement.play();

          isLiveActive = true;
          liveEmpty.style.display = 'none';
          liveDisplayImg.style.display = 'block';
          liveHud.style.display = 'flex';
          liveFeedInfo.textContent = src === 'usb_phone' ? 'USB PHONE WEBCAM (1080p 60 FPS)' : 'LOCAL WEBCAM (720p)';
          liveToggleBtn.className = 'btn-gold-action btn-danger';
          liveToggleText.textContent = 'Terminate Optical Feed';

          scheduleBrowserFrame();
        } catch (e) {
          alert('Camera error: ' + e.message);
          stopLiveCamera();
        }
      } else if (src === 'server') {
        // Server USB Camera
        isLiveActive = true;
        liveEmpty.style.display = 'none';
        liveDisplayImg.style.display = 'block';
        liveHud.style.display = 'flex';
        liveFeedInfo.textContent = 'SERVER USB CAMERA (INDEX 0)';
        liveToggleBtn.className = 'btn-gold-action btn-danger';
        liveToggleText.textContent = 'Terminate Optical Feed';
        pollServerStream();
      } else if (src === 'ip_camera') {
        // External RTSP / IP Camera
        await connectIpCamera();
      }
    }

    function stopLiveCamera() {
      isLiveActive = false;
      isPhoneStreaming = false;
      if (phonePollTimer) clearInterval(phonePollTimer);
      if (liveMediaStream) {
        liveMediaStream.getTracks().forEach(t => t.stop());
        liveMediaStream = null;
      }
      if (liveFrameLoopId) clearTimeout(liveFrameLoopId);
      fetch('/api/ip_camera/stop', { method: 'POST' }).catch(() => {});

      liveDisplayImg.src = '';
      liveDisplayImg.style.display = 'none';
      liveEmpty.style.display = 'block';
      liveHud.style.display = 'none';
      liveDisplayPanel.classList.remove('alarm-active');
      liveFeedInfo.textContent = 'FEED STANDBY';
      liveToggleBtn.className = 'btn-gold-action';
      liveToggleText.textContent = 'Activate Live Optical Feed';
      liveStatusBanner.className = 'luxury-status-banner idle';
      liveStatusText.textContent = 'Optical sensor standby.';
    }

    // Hardware-Accelerated MJPEG Video Socket with 4Hz Telemetry
    function startPhoneStreamListener() {
      // Connect to hardware-accelerated MJPEG socket stream
      liveDisplayImg.src = '/api/phone_stream/live.mjpeg?t=' + Date.now();

      phonePollTimer = setInterval(async () => {
        if (!isLiveActive || !isPhoneStreaming) return;
        try {
          const res = await fetch('/api/phone_stream/telemetry');
          const data = await res.json();
          if (data.has_frame) {
            liveLatencyText.textContent = `${data.inference_time_ms.toFixed(0)} ms`;
            liveMetricLatency.textContent = `${data.inference_time_ms.toFixed(1)} ms`;
            liveFpsText.textContent = `${data.fps.toFixed(1)} FPS`;
            liveMetricFps.textContent = `${data.fps.toFixed(1)} FPS`;
            liveFeedInfo.textContent = `PHONE LINKED: ${data.device_name.toUpperCase()} (30 FPS MJPEG)`;

            const count = data.detections_count;
            liveMetricDefects.textContent = count;

            if (count > 0) {
              liveDisplayPanel.classList.add('alarm-active');
              liveVerdictBadge.style.color = 'var(--ruby-defect)';
              liveVerdictBadgeText.textContent = `ALERT: ${count} DEFECT`;
              liveStatusBanner.className = 'luxury-status-banner defect';
              liveStatusBanner.innerHTML = `<div><strong>WIRELESS DETECTIONS ACTIVE:</strong> Found ${count} missing gem defect(s) via mobile camera!</div>`;
            } else {
              liveDisplayPanel.classList.remove('alarm-active');
              liveVerdictBadge.style.color = 'var(--emerald-pass)';
              liveVerdictBadgeText.textContent = 'SPECIMEN FLAWLESS';
              liveStatusBanner.className = 'luxury-status-banner pass';
              liveStatusBanner.innerHTML = `<div><strong>MOBILE SCAN PASS:</strong> Zero anomalies visible from phone camera angle.</div>`;
            }
          }
        } catch (e) {}
      }, 250);
    }

    // Laptop Webcam loop
    function scheduleBrowserFrame() {
      if (!isLiveActive) return;
      liveFrameLoopId = setTimeout(captureBrowserFrame, 35);
    }
    async function captureBrowserFrame() {
      if (!isLiveActive) return;
      if (liveVideoElement.videoWidth === 0) { scheduleBrowserFrame(); return; }
      try {
        const vw = liveVideoElement.videoWidth;
        const vh = liveVideoElement.videoHeight;
        liveOffscreenCanvas.width = vw;
        liveOffscreenCanvas.height = vh;
        const ctx = liveOffscreenCanvas.getContext('2d');
        ctx.drawImage(liveVideoElement, 0, 0, vw, vh);

        const dataUrl = liveOffscreenCanvas.toDataURL('image/jpeg', 0.82);
        const conf = parseFloat(liveConfSlider.value) / 100.0;
        const resp = await fetch('/api/predict_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl, conf: conf })
        });
        const data = await resp.json();
        if (data.success && isLiveActive) {
          liveDisplayImg.src = data.annotated_image;
          liveLatencyText.textContent = `${data.inference_time_ms.toFixed(0)} ms`;
          const count = data.detections_count;
          liveMetricDefects.textContent = count;
          if (count > 0) {
            liveDisplayPanel.classList.add('alarm-active');
            liveVerdictBadge.style.color = 'var(--ruby-defect)';
            liveVerdictBadgeText.textContent = `ALERT: ${count} DEFECT`;
          } else {
            liveDisplayPanel.classList.remove('alarm-active');
            liveVerdictBadge.style.color = 'var(--emerald-pass)';
            liveVerdictBadgeText.textContent = 'SPECIMEN FLAWLESS';
          }
        }
      } catch (e) {}
      scheduleBrowserFrame();
    }

    async function pollServerStream() {
      if (!isLiveActive) return;
      const conf = parseFloat(liveConfSlider.value) / 100.0;
      liveDisplayImg.src = `/api/server_camera_stream?conf=${conf}&t=${Date.now()}`;
      setTimeout(pollServerStream, 2000);
    }

    function setIpCamUrlPreset(type) {
      const input = document.getElementById('ipCamUrl');
      const ip = (currentNetworkInfo && currentNetworkInfo.local_ip) || '192.168.1.6';
      if (type === 'rtsp') {
        input.value = `rtsp://${ip}:8554/live`;
      } else if (type === 'droidcam') {
        input.value = `http://${ip}:4747/video`;
      } else if (type === 'android') {
        input.value = `http://${ip}:8080/video`;
      }
    }

    async function connectIpCamera() {
      const url = document.getElementById('ipCamUrl').value.trim();
      if (!url) {
        alert('Please enter a valid RTSP or HTTP camera URL.');
        return;
      }
      const connectBtn = document.getElementById('btnConnectIp');
      if (connectBtn) connectBtn.innerHTML = '<span class="spinner"></span> Connecting...';
      try {
        const res = await fetch('/api/ip_camera/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
        const data = await res.json();
        if (data.success) {
          isLiveActive = true;
          liveEmpty.style.display = 'none';
          liveDisplayImg.style.display = 'block';
          liveHud.style.display = 'flex';
          liveFeedInfo.textContent = url.toLowerCase().startsWith('rtsp://') ? 'RTSP CAMERA STREAM (iOS / IP Cam)' : 'IP CAMERA STREAM';
          liveToggleBtn.className = 'btn-gold-action btn-danger';
          liveToggleText.textContent = 'Terminate Optical Feed';
          liveStatusBanner.className = 'luxury-status-banner idle';
          liveStatusText.textContent = 'Connected to external RTSP/IP camera. Running real-time YOLOv8 QC...';
          pollIpCameraLoop();
        } else {
          alert('Failed to connect to IP Camera stream: ' + (data.error || 'Connection failed. Please verify that your iPhone RTSP app is broadcasting on this IP/port.'));
        }
      } catch (e) {
        alert('Connection error: ' + e.message);
      } finally {
        if (connectBtn) connectBtn.innerHTML = 'Connect IP Stream';
      }
    }

    async function pollIpCameraLoop() {
      if (!isLiveActive || liveCameraSource.value !== 'ip_camera') return;
      const conf = parseFloat(liveConfSlider.value) / 100.0;
      try {
        const res = await fetch(`/api/ip_camera/frame?conf=${conf}`);
        const data = await res.json();
        if (data.success && isLiveActive) {
          liveDisplayImg.src = data.annotated_image;
          liveLatencyText.textContent = `${data.inference_time_ms.toFixed(0)} ms`;
          liveMetricLatency.textContent = `${data.inference_time_ms.toFixed(1)} ms`;
          const count = data.detections_count;
          liveMetricDefects.textContent = count;
          if (data.detections && data.detections.length > 0) {
            const maxConf = Math.max(...data.detections.map(d => d.confidence));
            liveMetricConfidence.textContent = `${(maxConf * 100).toFixed(1)}%`;
          } else {
            liveMetricConfidence.textContent = '0.0%';
          }
          if (data.fps) {
            liveFpsText.textContent = `${data.fps.toFixed(1)} FPS`;
            liveMetricFps.textContent = `${data.fps.toFixed(1)} FPS`;
          }

          if (count > 0) {
            liveDisplayPanel.classList.add('alarm-active');
            liveVerdictBadge.style.color = 'var(--ruby-defect)';
            liveVerdictBadgeText.textContent = `ALERT: ${count} DEFECT`;
            liveStatusBanner.className = 'luxury-status-banner defect';
            liveStatusBanner.innerHTML = `<div><strong>RTSP STREAM DETECTIONS:</strong> Found ${count} defect(s) on specimen!</div>`;
          } else {
            liveDisplayPanel.classList.remove('alarm-active');
            liveVerdictBadge.style.color = 'var(--emerald-pass)';
            liveVerdictBadgeText.textContent = 'SPECIMEN FLAWLESS';
            liveStatusBanner.className = 'luxury-status-banner pass';
            liveStatusBanner.innerHTML = `<div><strong>RTSP STREAM PASS:</strong> Specimen meets metrology criteria.</div>`;
          }
        }
      } catch (e) {}
      if (isLiveActive && liveCameraSource.value === 'ip_camera') {
        setTimeout(pollIpCameraLoop, 40);
      }
    }
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json_response(self, obj, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def serve_file_with_range(self, filepath, content_type="video/mp4", is_head=False):
        if not os.path.exists(filepath):
            self.send_response(404)
            self.end_headers()
            return

        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                if not is_head:
                    with open(filepath, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not is_head:
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

    def do_HEAD(self):
        self.do_GET(is_head=True)

    def do_GET(self, is_head=False):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path.startswith("/?"):
            body_bytes = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            if not is_head:
                self.wfile.write(body_bytes)
            return

        # Dedicated Mobile App Route
        if path == "/mobile" or path == "/phone":
            body_bytes = MOBILE_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            if not is_head:
                self.wfile.write(body_bytes)
            return

        # Network Info (LAN IP & Mobile URLs)
        if path == "/api/network_info":
            local_ip = get_local_ip()
            https_mobile_url = f"https://{local_ip}:8443/mobile"
            http_mobile_url = f"http://{local_ip}:8080/mobile"
            self.send_json_response({
                "local_ip": local_ip,
                "https_mobile_url": https_mobile_url,
                "http_mobile_url": http_mobile_url,
                "mobile_url": https_mobile_url
            })
            return

        # Dynamic QR Code Image (PNG)
        if path == "/api/qr_code":
            local_ip = get_local_ip()
            params = urllib.parse.parse_qs(parsed.query)
            mode = params.get("mode", ["https"])[0].lower()
            if mode == "http":
                mobile_url = f"http://{local_ip}:8080/mobile"
            else:
                mobile_url = f"https://{local_ip}:8443/mobile"

            qr = qrcode.QRCode(box_size=6, border=2)
            qr.add_data(mobile_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="#f59e0b", back_color="#0e121e")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_bytes)))
            self.end_headers()
            if not is_head:
                self.wfile.write(png_bytes)
            return

        # Hardware-Accelerated 30 FPS MJPEG Stream (Direct push to browser socket)
        if path == "/api/phone_stream/live.mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last_ts = 0
            try:
                while True:
                    jpeg, ts = phone_pipeline.get_annotated_jpeg()
                    if jpeg is not None and ts > last_ts:
                        last_ts = ts
                        header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode("utf-8")
                        self.wfile.write(header + jpeg + b"\r\n")
                    time.sleep(0.015)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
            return

        # Low-overhead Telemetry Metrics (4-5 Hz polling for stats)
        if path == "/api/phone_stream/telemetry":
            self.send_json_response(phone_pipeline.get_telemetry())
            return

        # Latest Phone Stream Frame (for backward compatibility)
        if path == "/api/phone_stream/latest":
            jpeg, ts = phone_pipeline.get_annotated_jpeg()
            telemetry = phone_pipeline.get_telemetry()
            b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("utf-8") if jpeg else ""
            self.send_json_response({
                "has_frame": jpeg is not None,
                "timestamp": ts,
                "annotated_image": b64,
                "detections_count": telemetry["detections_count"],
                "is_defective": telemetry["is_defective"],
                "inference_time_ms": telemetry["inference_time_ms"],
                "device_name": telemetry["device_name"],
                "fps": telemetry["fps"]
            })
            return

        if path == "/api/samples":
            samples = []
            if os.path.exists(TEST_IMAGES_DIR):
                files = os.listdir(TEST_IMAGES_DIR)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                samples = image_files[:8]
            self.send_json_response(samples)
            return

        if path.startswith("/sample_image/"):
            filename = os.path.basename(path[len("/sample_image/"):])
            filepath = os.path.join(TEST_IMAGES_DIR, filename)
            if os.path.exists(filepath):
                self.serve_file_with_range(filepath, "image/jpeg", is_head=is_head)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        if path == "/api/demo_video":
            if os.path.exists(DEMO_VIDEO_PATH):
                self.serve_file_with_range(DEMO_VIDEO_PATH, "video/mp4", is_head=is_head)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        if path.startswith("/api/video/status/"):
            task_id = path[len("/api/video/status/"):]
            with video_tasks_lock:
                task = video_tasks.get(task_id)
            if task:
                self.send_json_response(task)
            else:
                self.send_json_response({"status": "error", "error": "Task not found"}, 404)
            return

        if path.startswith("/api/video/stream/"):
            task_id = path[len("/api/video/stream/"):]
            out_file = os.path.join(OUTPUT_VIDEOS_DIR, f"{task_id}.mp4")
            if os.path.exists(out_file):
                self.serve_file_with_range(out_file, "video/mp4", is_head=is_head)
                return
            self.send_response(404)
            self.end_headers()
            return

        if path.startswith("/api/video/input/"):
            task_id = path[len("/api/video/input/"):]
            in_file = os.path.join(INPUT_VIDEOS_DIR, f"{task_id}.mp4")
            if os.path.exists(in_file):
                self.serve_file_with_range(in_file, "video/mp4", is_head=is_head)
                return
            self.send_response(404)
            self.end_headers()
            return

        if path.startswith("/api/server_camera_stream"):
            params = urllib.parse.parse_qs(parsed.query)
            conf = float(params.get("conf", [0.25])[0])
            frame_bytes = server_camera.get_frame(conf=conf)
            if frame_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(503)
                self.end_headers()
            return

        if path.startswith("/api/ip_camera/frame"):
            params = urllib.parse.parse_qs(parsed.query)
            conf = float(params.get("conf", [0.25])[0])
            frame_data = ip_camera.get_frame(conf=conf)
            if frame_data:
                self.send_json_response(frame_data)
            else:
                self.send_json_response({"success": False, "error": "No frame"}, 503)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Single Image Prediction
        if path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))

                image_data = data.get("image", "")
                conf_threshold = float(data.get("conf", 0.25))

                if not image_data:
                    self.send_json_response({"success": False, "error": "No image data provided"}, 400)
                    return

                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                np_img = np.array(pil_img)
                cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

                start_time = time.time()
                results = model.predict(source=cv2_img, conf=conf_threshold, verbose=False)
                inference_time_ms = (time.time() - start_time) * 1000

                result = results[0]
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                out_pil = Image.fromarray(annotated_rgb)

                buffer = io.BytesIO()
                out_pil.save(buffer, format="JPEG", quality=92)
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

                detections = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    names = model.names
                    for i in range(len(boxes)):
                        cls_id = int(clss[i])
                        cls_name = names.get(cls_id, str(cls_id))
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": float(confs[i]),
                            "bbox": xyxy[i].tolist()
                        })

                self.send_json_response({
                    "success": True,
                    "detections_count": len(detections),
                    "is_defective": len(detections) > 0,
                    "inference_time_ms": inference_time_ms,
                    "annotated_image": annotated_b64,
                    "detections": detections
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        # 2. Live Camera Frame Prediction
        if path == "/api/predict_frame":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))

                image_data = data.get("image", "")
                conf_threshold = float(data.get("conf", 0.25))

                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                start_time = time.time()
                results = model.predict(source=cv2_img, conf=conf_threshold, verbose=False)
                inference_time_ms = (time.time() - start_time) * 1000

                result = results[0]
                annotated_bgr = result.plot()
                ret, jpeg_buf = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg_buf).decode("utf-8")

                detections = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    names = model.names
                    for i in range(len(boxes)):
                        cls_id = int(clss[i])
                        cls_name = names.get(cls_id, str(cls_id))
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": float(confs[i]),
                            "bbox": xyxy[i].tolist()
                        })

                self.send_json_response({
                    "success": True,
                    "detections_count": len(detections),
                    "is_defective": len(detections) > 0,
                    "inference_time_ms": inference_time_ms,
                    "annotated_image": annotated_b64,
                    "detections": detections
                })
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        # High-Speed Binary Frame Ingestion (Zero-latency fire-and-forget for 30 FPS Live Stream)
        if path.startswith("/api/phone_stream/push_frame"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_bytes = self.rfile.read(content_length)
                params = urllib.parse.parse_qs(parsed.query)
                conf = float(params.get("conf", [0.25])[0])
                device = params.get("device", ["iPhone"])[0]

                np_arr = np.frombuffer(raw_bytes, np.uint8)
                cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv2_img is not None:
                    phone_pipeline.push_raw_frame(cv2_img, device=device, conf=conf)

                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
            except Exception:
                self.send_response(500)
                self.end_headers()
            return

        # 3. Wireless Phone Stream Frame Receiver (From Mobile Phone Snap)
        if path == "/api/phone_stream/frame":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))

                image_data = data.get("image", "")
                conf_threshold = float(data.get("conf", 0.25))
                device = data.get("device", "Mobile Camera")

                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                start_time = time.time()
                results = model.predict(source=cv2_img, conf=conf_threshold, verbose=False)
                inference_time_ms = (time.time() - start_time) * 1000

                result = results[0]
                annotated_bgr = result.plot()
                ret, jpeg_buf = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg_buf).decode("utf-8")

                detections = []
                boxes = result.boxes
                num_boxes = len(boxes) if boxes is not None else 0
                if boxes is not None and num_boxes > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    names = model.names
                    for i in range(num_boxes):
                        cls_id = int(clss[i])
                        cls_name = names.get(cls_id, str(cls_id))
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": float(confs[i]),
                            "bbox": xyxy[i].tolist()
                        })

                # Broadcast to phone_pipeline for desktop display
                with phone_pipeline.lock:
                    phone_pipeline.annotated_jpeg = jpeg_buf.tobytes()
                    phone_pipeline.annotated_b64 = annotated_b64
                    phone_pipeline.annotated_timestamp = time.time()
                    phone_pipeline.detections_count = num_boxes
                    phone_pipeline.is_defective = num_boxes > 0
                    phone_pipeline.inference_time_ms = inference_time_ms
                    phone_pipeline.device_name = device

                self.send_json_response({
                    "success": True,
                    "detections_count": num_boxes,
                    "is_defective": num_boxes > 0,
                    "inference_time_ms": inference_time_ms,
                    "annotated_image": annotated_b64,
                    "detections": detections
                })
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        # 4. External IP Camera Start/Stop
        if path == "/api/ip_camera/start":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                url = data.get("url", "")
                success = ip_camera.start(url)
                self.send_json_response({"success": success})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        if path == "/api/ip_camera/stop":
            try:
                ip_camera.stop()
                self.send_json_response({"success": True})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        # 5. Video Processing Endpoints
        if path == "/api/video/upload":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                conf = float(params.get("conf", [0.25])[0])
                skip = int(params.get("skip", [1])[0])
                task_id = str(uuid.uuid4())[:8]

                input_path = os.path.join(INPUT_VIDEOS_DIR, f"{task_id}.mp4")
                output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{task_id}.mp4")

                content_length = int(self.headers.get("Content-Length", 0))
                bytes_read = 0
                with open(input_path, "wb") as f:
                    while bytes_read < content_length:
                        chunk_size = min(65536, content_length - bytes_read)
                        chunk = self.rfile.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_read += len(chunk)

                with video_tasks_lock:
                    video_tasks[task_id] = {
                        "task_id": task_id,
                        "status": "processing",
                        "progress": 0.0,
                        "current_frame": 0,
                        "total_frames": 0,
                        "defects_count": 0,
                        "defect_frames": 0,
                        "fps": 0.0,
                        "error": None,
                        "results": None
                    }

                t = threading.Thread(
                    target=process_video_background,
                    args=(task_id, input_path, output_path, conf, skip),
                    daemon=True
                )
                t.start()

                self.send_json_response({"success": True, "task_id": task_id})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        if path == "/api/video/process_sample":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                conf = float(params.get("conf", [0.25])[0])
                skip = int(params.get("skip", [1])[0])
                task_id = str(uuid.uuid4())[:8]

                if not os.path.exists(DEMO_VIDEO_PATH):
                    self.send_json_response({"success": False, "error": "Demo video not found on server"}, 404)
                    return

                output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{task_id}.mp4")

                with video_tasks_lock:
                    video_tasks[task_id] = {
                        "task_id": task_id,
                        "status": "processing",
                        "progress": 0.0,
                        "current_frame": 0,
                        "total_frames": 0,
                        "defects_count": 0,
                        "defect_frames": 0,
                        "fps": 0.0,
                        "error": None,
                        "results": None
                    }

                t = threading.Thread(
                    target=process_video_background,
                    args=(task_id, DEMO_VIDEO_PATH, output_path, conf, skip),
                    daemon=True
                )
                t.start()

                self.send_json_response({"success": True, "task_id": task_id})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)
            return

        self.send_response(404)
        self.end_headers()


def run_server(http_port=8080, https_port=8443):
    socketserver.TCPServer.allow_reuse_address = True
    local_ip = get_local_ip()

    # 1. Ensure SSL certificate exists for HTTPS
    cert_path, key_path = ensure_ssl_cert(
        os.path.join(WORKSPACE_DIR, "cert.pem"),
        os.path.join(WORKSPACE_DIR, "key.pem"),
        local_ip
    )

    # 2. Setup HTTP Server (Desktop & Quick Snap)
    httpd = socketserver.ThreadingTCPServer(("", http_port), RequestHandler)

    # 3. Setup HTTPS Server (iPhone & Android Live Stream)
    httpsd = None
    try:
        httpsd = socketserver.ThreadingTCPServer(("", https_port), RequestHandler)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        httpsd.socket = ssl_ctx.wrap_socket(httpsd.socket, server_side=True)
    except Exception as e:
        print(f"Warning: Could not start HTTPS server on port {https_port}: {e}")
        httpsd = None

    url_http = f"http://localhost:{http_port}"
    url_desktop_ip = f"http://{local_ip}:{http_port}"
    url_mobile_https = f"https://{local_ip}:{https_port}/mobile" if httpsd else None
    url_mobile_http = f"http://{local_ip}:{http_port}/mobile"

    print("=" * 72)
    print("  AURUM VISION | AI JEWELLERY DEFECT METROLOGY SUITE")
    print(f"  Desktop Web App:            {url_http}")
    print(f"  Desktop Web App (LAN IP):   {url_desktop_ip}")
    if url_mobile_https:
        print(f"  Mobile Live Camera (HTTPS): {url_mobile_https}  (iOS / Android 30 FPS)")
    print(f"  Mobile Snap Camera (HTTP):  {url_mobile_http}   (Zero-Config HTTP Snap)")
    print(f"  Host Machine IP:            {local_ip}")
    print("=" * 72)

    # Start HTTP in daemon background thread
    t_http = threading.Thread(target=httpd.serve_forever, daemon=True)
    t_http.start()

    try:
        webbrowser.open(url_http)
    except Exception:
        pass

    try:
        if httpsd:
            httpsd.serve_forever()
        else:
            t_http.join()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully...")
    finally:
        server_camera.release()
        ip_camera.stop()
        httpd.shutdown()
        if httpsd:
            httpsd.shutdown()

if __name__ == "__main__":
    http_port = 8080
    https_port = 8443
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        http_port = int(sys.argv[1])
    run_server(http_port, https_port)

