# ComfyUI-SAM3-Chunked

**Memory-efficient SAM3 video segmentation for ComfyUI.**

Drop-in replacement for [PozzettiAndrea/ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) that processes long videos (1000+ frames) without GPU out-of-memory errors.

## Why This Exists

The original ComfyUI-SAM3 loads **all video frames to GPU at once**:

```
1500 frames × 3 × 1008 × 1008 × 2 bytes  ≈  9.2 GB  →  OOM on 24 GB cards
```

This package processes frames in configurable **chunks** (default 100 frames ≈ 0.6 GB), clearing GPU memory between chunks.

| | Original | Chunked |
|---|---|---|
| 100 frames | ✅ 0.6 GB | ✅ 0.6 GB |
| 500 frames | ⚠️ 3.1 GB | ✅ 0.6 GB |
| 1500 frames | ❌ 9.2 GB (OOM) | ✅ 0.6 GB |
| 3000 frames | ❌ 18.4 GB (OOM) | ✅ 0.6 GB |

## Installation

1. Clone into your ComfyUI `custom_nodes/` folder:
   ```bash
   cd ComfyUI/custom_nodes/
   # Place the ComfyUI-SAM3-Chunked folder here
   ```

2. Download the SAM3 checkpoint to `ComfyUI/models/sam3/`:
   ```bash
   mkdir -p models/sam3
   # Download sam3.pt from HuggingFace (or let the node auto-download)
   ```

3. **Remove or disable** the original `ComfyUI-SAM3` package to avoid node-name conflicts (both packages register the same node names).

## Nodes

All nodes use **identical names and types** as the original package, so existing workflows load without changes.

| Node | Type | Purpose |
|------|------|---------|
| **LoadSAM3Model** | `SAM3_MODEL` | Load checkpoint (auto-downloads from HuggingFace) |
| **SAM3BBoxCollector** | `SAM3_BOXES_PROMPT` | Collect bounding-box prompts |
| **SAM3VideoSegmentation** | `SAM3_VIDEO_STATE` | Initialize video state (frames stay on CPU) |
| **SAM3Propagate** | `SAM3_VIDEO_MASKS` | Propagate masks through video **in chunks** |
| **SAM3VideoOutput** | `MASK`, `IMAGE` | Convert masks to standard ComfyUI types |

### New Parameters (on SAM3VideoSegmentation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 100 | Frames per GPU chunk (lower = less VRAM) |
| `overlap_frames` | 10 | Overlap between chunks for mask continuity |

### New Parameters (on SAM3Propagate)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clear_cache` | True | Free GPU memory between chunks |

## Bundled SAM3 Library

The package includes SAM3's core model code (`lib/sam3/`) so it works **standalone** — no separate `pip install sam3` required. If you do have SAM3 installed system-wide, that installation takes priority.

## Model Download

The `LoadSAM3Model` node auto-downloads models from HuggingFace on first use. If the repo is gated, provide your HuggingFace token in the `hf_token` input.

## Compatibility

- **ComfyUI**: Latest
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Works with**: SAM3DBody2abc, any workflow that uses SAM3 nodes

## License

- **This package**: Apache 2.0
- **SAM3 model code** (bundled in `lib/sam3/`): Meta Platforms Inc., Apache 2.0
