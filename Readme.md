# 🚁 Skynetra

**Skynetra** is a modular, real-time **face detection, tracking, and identification pipeline** designed for **video streams and aerial / edge scenarios**.

It combines modern deep-learning components with **stateful, event-driven system design**, ensuring high stability, low latency, and extensibility for future research and deployment.

---

## Live Demo(8-second clip)

* **Left**: Raw input video
* **Right**: Processed Output video

![skynetra_demo](https://github.com/user-attachments/assets/58ce3822-ab77-498d-81e7-2b18abda928a)

---

## Performance Benchmarks

Tested on a Full HD (1080p) 60 FPS video clip .  
Pipeline includes sparse YOLOv8-Face detection, dense ByteTrack tracking, FaceNet embeddings, Hopfield temporal pooling, and visualization.

| Hardware                  | Resolution | Average FPS | Avg Latency (ms/frame) | Min/Max Latency (ms) |
|---------------------------|------------|-------------|-------------------------|-----------------------|
| RTX 2050 Laptop GPU       | 1080p      | 19.2        | 22.7                    | 2.0 / 81.4            | 
| Intel Core i5-12450H CPU  | 1080p      | 12.4        | 43.3                    | 2.4 / 243.7           |

- FPS calculated as total frames / total processing time (excluding video I/O overhead).
- Latency = per-frame pipeline time (detection + tracking + ID decision + drawing).
- Future tests planned: Jetson Orin/edge devices, 4K input, optimizations (TensorRT, ONNX).

---

## ✨ Key Features

* **Face Detection** using YOLOv8-Face
* **Multi-Object Tracking** using ByteTrack
* **Face Embedding** via FaceNet (VGGFace2)
* **Temporal Identity Stabilization** using embedding pooling
* **Event-based Identification** (not per-frame, avoids flicker)
* **Frame-sampling aware architecture** (detector ≠ tracker cadence)
* **Fully modular design** (swap models without touching pipeline logic)

Skynetra is built as a **system**, not a demo.

---

## 🧠 Core Design Philosophy

Skynetra follows three fundamental principles:

### 1. Detection creates state

YOLO detects faces only when needed.

### 2. Tracking propagates state

ByteTrack runs every frame to maintain continuity.

### 3. Identity is decided sparsely

Face identification happens **only after sufficient evidence**, not every frame.

This separation avoids:

* Identity flickering
* Excessive GPU load
* Latency spikes
* Fragile pipelines

---

## 🏗️ Architecture Overview

```
Video Frame
   ↓
YOLOv8-Face (sparse, sampled)
   ↓
ByteTrack (every frame)
   ↓
Face Cropping (stable tracks only)
   ↓
FaceNet Embeddings
   ↓
Identity Memory (per track)
   ↓
Embedding Pooling
   ↓
Identity Matching
```

Each stage is **decoupled** and **replaceable**.

---

## 📁 Repository Structure

```
skynetra/
│
├── models/          # Detector, tracker, embedder wrappers
├── detectors/        # Detector logic
├── utils/           # Frame sampler, vision utilities
├── trackers/       # ByteTrack 
│
├── main.py          # Entry point
├── requirements.txt
└── README.md
```

This structure allows:

* Easy experimentation
* Component-level optimization
* Clean scaling to real-time systems

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

Update the video path and model weights inside `main.py` as needed.

---

## 🧩 Swappable Components

Skynetra is intentionally modular.

You can easily:

* Replace YOLOv8-Face with another detector
* Swap ByteTrack with a different tracker
* Replace mean pooling with Hopfield pooling
* Add new identity matching strategies
* Move detection or embedding to async threads

No changes to the core pipeline are required.

---

## 🧪 Current Capabilities

* Stable ID assignment across frames
* Robust to skipped detections
* Identity confidence increases over time
* Handles high-resolution (4K) video
* Works in offline and near-real-time modes

---

## 🔮 Planned Extensions

* Hopfield pooling for identity refinement
* Identity confidence decay & re-identification
* Async / multi-threaded execution
* Persistent identity databases
* Edge deployment (Jetson / drone hardware)
* Performance profiling & benchmarks

---

## ⚠️ Important Notes

* Identification is **event-based**, not frame-based
* Tracker initialization includes a warm-up phase
* Designed for **systems research**, not plug-and-play apps

---

## 🧠 Why “Skynetra”?

> **Sky** — aerial & vision systems
> **Netra** — Sanskrit for *eye / vision*

Skynetra literally means:
**“An eye in the sky, with memory.”**

---

## 📜 License

MIT License 

