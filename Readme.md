<h1 align="center">Skynetra</h1>

<p align="center">
Real-time ISTAR target identification with temporal identity memory
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
  <img src="https://img.shields.io/badge/framework-PyTorch-red">
  <img src="https://img.shields.io/badge/domain-Computer%20Vision-purple">
  <img src="https://img.shields.io/badge/deployment-Edge%20AI-orange">
  <img src="https://img.shields.io/badge/performance-Real--Time-success">
  <img src="https://img.shields.io/badge/license-MIT-green">
  <img src="https://img.shields.io/badge/status-Active%20Development-yellow">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#live-demo">Demo</a> •
  <a href="#performance-benchmarks">Benchmarks</a>
</p>

**Skynetra** is a real-time **target identification and tracking pipeline** designed for high-motion aerial video streams from **ISTAR drones and surveillance systems**.

The system combines **sparse detection, multi-object tracking, deep feature embeddings, and temporal identity memory** to maintain **stable target identities in dynamic environments**.

## Problem

Modern ISTAR drone operations increasingly rely on AI-assisted target identification.
However, many current systems operate purely on **frame-by-frame predictions**, which introduces critical risks.

Key challenges include:

* **Identity flickering** under rapid motion
* **Lost tracking** during occlusions
* **Overconfidence in incorrect predictions**
* **Automation bias**, where human operators over-trust AI outputs

In high-stakes environments, these failures can lead to **incorrect situational awareness and dangerous decisions**.

## Solution

Skynetra addresses these issues through a **temporal, tracking-aware identification pipeline**.

Instead of relying on single-frame predictions, the system integrates:

- **YOLO-based target detection**
- **ByteTrack multi-object tracking**
- **Deep feature embeddings (MobileFaceNet)**
- **Hopfield-based temporal memory**

This architecture aggregates identity evidence across time, resulting in **stable target recognition and reduced decision uncertainty**.

## Key Features

- Real-time ISTAR target identification  
- Sparse detection + dense tracking pipeline  
- Temporal identity memory via Hopfield networks  
- Stable IDs with minimal flicker  
- Modular architecture for rapid experimentation  
- Edge-ready inference (ONNX / TensorRT)

## Live Demo

**Left**: Raw input video  
**Right**: Skynetra output with YOLOv8 detection, ByteTrack tracking, MobileFaceNet embeddings, Hopfield temporal pooling, and **stable, persistent IDs** (no jumping around!)

![Skynetra Demo - Raw vs Processed](assets/534364962-58ce3822-ab77-498d-81e7-2b18abda928a.gif)

(Looping 8-second Full HD clip processed in real-time. Watch how IDs stay consistent across frames.)

## System Architecture

Drone Video Feed → Target Detection (YOLO) → Multi-Object Tracking (ByteTrack)  
→ Target Embedding Extraction (MobileFaceNet) → Temporal Identity Memory (Hopfield Network)  
→ Identity Recognition + Confidence Score → Operator Decision Support

Skynetra processes drone video streams through a modular pipeline that separates
detection, tracking, embedding, and temporal identity reasoning.

The Hopfield-based temporal memory aggregates identity evidence across
multiple frames, allowing the system to maintain stable target identities
even under rapid motion, occlusion, or noisy observations.


## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add known identities to persistent storage
python add_info.py

# 3. Run the pipeline on a video
python main.py
```

That's literally it. No complicated setup. Just run and watch.

## Project Structure

The repository is organized into modular components so that detectors,
trackers, and embedding models can be easily swapped or extended.

skynetra/
│
├── assets/ # Demo videos, GIFs, and visual resources
├── detectors/ # Target / face detection modules (YOLO models)
├── trackers/ # Multi-object tracking implementations (ByteTrack)
├── models/ # Embedding models (MobileFaceNet, TensorRT versions)
├── utils/ # Helper utilities (preprocessing, visualization, etc.)
│
├── add_info.py # Script to add known identities to persistent storage
├── main.py # Entry point for running the full pipeline
├── requirements.txt
│
├── LICENSE
└── README.md

## System Features

- Real-time processing — **~120 FPS raw** on 720p (no rendering/visualization overhead)
- **Persistent identity storage** — JSON metadata + tensor files via `add_info.py` — auto-loads known targets on startup
- **Modern Hopfield layer** — temporal embedding pooling inspired by [Modern Hopfield Networks (Ramsauer et al., 2021)](https://arxiv.org/abs/2008.02217), acting like associative human memory
- **Smart frame sampling** with tracker feedback
- **Asynchronous GPU detection** — sparse YOLOv8-Face + dense ByteTrack
- **Fully modular** — swap detector, tracker, embedder, pooling, etc. without touching core logic
- **TensorRT compilation** support
- **MobileFaceNet ONNX** embedding (huge speedup over original FaceNet)
- **Open Set Rejection** explicit Unknown Handling

## Performance Benchmarks

**Pipeline**  
YOLOv8-Face + ByteTrack + MobileFaceNet (TensorRT FP16) + Hopfield temporal pooling

**Mode**  
Raw pipeline (no visualization unless stated)

**Hardware**  
RTX 2050 Laptop GPU + Intel i5-12450H

**Content**  
High-motion, crowded real-world videos

---

### Raw Throughput (No Rendering)

| Resolution | Avg FPS | Detector (ms) | Tracker (ms) | Embedding (ms) | Notes |
|-----------:|--------:|--------------:|-------------:|---------------:|-------|
| 720p       | ~130    | ~0.2          | ~1.5         | ~3.0           | Fully real-time |
| 1080p      | ~50     | ~0.5          | ~1.8         | ~6.0           | Stable IDs |
| 4K         | ~14     | ~2.0          | ~1.3         | ~7.7           | Pixel-bound |

- **FPS** = total frames processed / total runtime  
- Latencies are **per-frame averages** across full runs  
- Tracker cost remains nearly constant across resolutions  
- Embedding cost scales mainly with **number of tracked targets**, not pixels  

---

### With Visualization / Rendering Enabled

| Resolution | Avg FPS | Notes |
|-----------:|--------:|-------|
| 720p       | ~90     | OpenCV overlay + ID drawing |
| 1080p      | ~43     | Smooth playback |
| 4K         | ~13     | Rendering becomes dominant |

Rendering overhead is **outside the core pipeline** and can be disabled for deployment.

---

### Key Observations

- Graceful performance degradation with increasing resolution  
- No ID flickering under chaotic motion  
- Tracker is **not** a bottleneck (flat cost across resolutions)  
- Pipeline becomes **pixel-bound at 4K**, not algorithm-bound  
- Real-time 4K achievable via detector downscaling or stronger GPUs


## Planned Extensions

- Multi-sensor fusion (e.g., IR + RGB gating mechanisms)
- One-click modularity (easy config-based component swapping)

## License

MIT License — see [LICENSE](LICENSE) file.

Thanks for reading!  
If you're building drones, surveillance, edge AI, or just love modular CV pipelines — fork it, break it, improve it. Let's make something useful together. 🚀

Harshit
