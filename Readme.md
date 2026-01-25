```markdown
# Skynetra 🛩️

**Skynetra** is a **modular, real-time face detection, tracking, and identification pipeline** built for video streams — especially aerial, drone, and edge scenarios.

It combines modern deep-learning tools with **stateful, event-driven design** to achieve:
- Extremely stable IDs (almost no flickering)
- Very low latency
- Full modularity for easy experimentation and deployment

Hello nerds & founders!!  
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
python main.py --video path/to/your/video.mp4
```

That's literally it. No complicated setup. Just run and watch.

## Live Demo (8-second clip – Looping GIF)

**Left**: Raw input video  
**Right**: Skynetra output with YOLOv8-Face detection, ByteTrack tracking, MobileFaceNet embeddings, Hopfield temporal pooling, and **stable, persistent IDs** (no jumping around!)

![Skynetra Demo - Raw vs Processed]((https://github.com/user-attachments/assets/58ce3822-ab77-498d-81e7-2b18abda928a))

(Looping 8-second Full HD clip processed in real-time. Watch how IDs stay consistent across frames.)

## Current Capabilities

- Real-time processing — **~120 FPS raw** on 720p (no rendering/visualization overhead)
- **Persistent identity storage** — JSON metadata + tensor files via `add_info.py` — auto-loads known faces on startup
- **Modern Hopfield layer** — temporal embedding pooling that acts like actual human associative memory
- **Smart frame sampling** with tracker feedback
- **Asynchronous GPU detection** — sparse YOLOv8-Face + dense ByteTrack
- **Fully modular** — swap detector, tracker, embedder, pooling, etc. without touching core logic
- **TensorRT compilation** support (planned full integration)
- **MobileFaceNet ONNX** embedding (huge speedup over original FaceNet)

## Performance Benchmarks (Latest – Jan 2026)

**Major upgrade**: Switched from FaceNet → **MobileFaceNet ONNX** + eliminated redundant CPU↔GPU memory transfers.

Tested on ~20-minute 720p video (raw pipeline, no rendering).

| Hardware                  | Resolution | Avg FPS (raw) | Avg Latency (ms/frame) | Notes                                      |
|---------------------------|------------|---------------|-------------------------|--------------------------------------------|
| RTX 2050 Laptop GPU       | 720p       | ~120          | ~8.3                    | MobileFaceNet ONNX + optimizations         |
| RTX 2050 Laptop GPU (old) | 1080p      | 19–20         | 22.7–22.9               | Previous FaceNet baseline (~6× speedup)    |
| Intel Core i5-12450H CPU  | 720p       | TBD           | TBD                     | CPU fallback (pending tests)               |

- FPS = total frames processed / total time (excluding video I/O)
- Latency = average per-frame pipeline time (detection + tracking + embedding + pooling + decision)
- Future tests: Jetson Orin / edge hardware, 4K input, int8/FP16 quantization

## Why Skynetra?

This isn't just another face-recognition script you see in phones or basic demos.  
It has **actual temporal memory** via the Hopfield layer — reducing **automation bias** through robust, event-based identification that mimics how humans remember and associate faces over time.

Built for **real-world deployment** — drones, border surveillance, robotics, edge cameras — where flickering IDs, high latency, or fragile persistence destroy usability.

## Planned Extensions

- Explicit open-set rejection / unknown handling (in progress)
- Multi-sensor fusion (e.g., IR + RGB gating mechanisms)
- One-click modularity (easy config-based component swapping)

## License

MIT License — see [LICENSE](LICENSE) file.

Thanks for reading!  
If you're building drones, surveillance, edge AI, or just love modular CV pipelines — fork it, break it, improve it. Let's make something useful together. 🚀

Harshit
```
