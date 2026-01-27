# Skynetra 

**Skynetra** is a **modular, real-time face detection, tracking, and identification pipeline** built for video streams — especially aerial, drone, and edge scenarios.

It combines modern deep-learning tools with **stateful, event-driven design** to achieve:
- Extremely stable IDs (almost no flickering)
- Very low latency
- Full modularity for easy experimentation and deployment

Hello nerds!!  
Yeah — I know what you're thinking:  
"What the heck is this Skynetra thing??"  
But before I explain anything, let's just get it working — because the proof is in the pudding.

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

## Live Demo (8-second clip – Looping GIF)

**Left**: Raw input video  
**Right**: Skynetra output with YOLOv8-Face detection, ByteTrack tracking, MobileFaceNet embeddings, Hopfield temporal pooling, and **stable, persistent IDs** (no jumping around!)

![Skynetra Demo - Raw vs Processed](assets/534364962-58ce3822-ab77-498d-81e7-2b18abda928a.gif)

(Looping 8-second Full HD clip processed in real-time. Watch how IDs stay consistent across frames.)

## Current Capabilities

- Real-time processing — **~120 FPS raw** on 720p (no rendering/visualization overhead)
- **Persistent identity storage** — JSON metadata + tensor files via `add_info.py` — auto-loads known faces on startup
- **Modern Hopfield layer** — temporal embedding pooling inspired by [Modern Hopfield Networks (Ramsauer et al., 2021)](https://arxiv.org/abs/2008.02217), acting like associative human memory
- **Smart frame sampling** with tracker feedback
- **Asynchronous GPU detection** — sparse YOLOv8-Face + dense ByteTrack
- **Fully modular** — swap detector, tracker, embedder, pooling, etc. without touching core logic
- **TensorRT compilation** support
- **MobileFaceNet ONNX** embedding (huge speedup over original FaceNet)
- **Open Set Rejection** explicit Unknown Handling

## Performance Benchmarks (Jan 2026)

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
- Embedding cost scales mainly with **number of faces**, not pixels  

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


## Why Skynetra?

This isn't just another face-recognition script you see in phones or basic demos.  
It has **actual temporal memory** via the Hopfield layer — reducing **[automation bias](https://en.wikipedia.org/wiki/Automation_bias)** through robust, event-based identification that mimics how humans remember and associate faces over time.

Built for **real-world deployment** — drones, border surveillance, robotics, edge cameras — where flickering IDs, high latency, or fragile persistence destroy usability.

## Planned Extensions

- Multi-sensor fusion (e.g., IR + RGB gating mechanisms)
- One-click modularity (easy config-based component swapping)

## License

MIT License — see [LICENSE](LICENSE) file.

Thanks for reading!  
If you're building drones, surveillance, edge AI, or just love modular CV pipelines — fork it, break it, improve it. Let's make something useful together. 🚀

Harshit
