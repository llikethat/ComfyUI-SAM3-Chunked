"""
ComfyUI-SAM3-Chunked — Memory-Efficient SAM3 Video Segmentation
================================================================

Drop-in replacement for PozzettiAndrea/ComfyUI-SAM3 with chunked processing
to handle long videos (1000+ frames) without GPU out-of-memory errors.

Provides the SAME output types as the original ComfyUI-SAM3 nodes:
  SAM3_MODEL, SAM3_VIDEO_STATE, SAM3_VIDEO_MASKS, SAM3_VIDEO_SCORES,
  SAM3_BOXES_PROMPT, SAM3_POINTS_PROMPT, SAM3_MULTI_PROMPT, MASK, IMAGE

so it slots directly into existing workflows.

Nodes:
  LoadSAM3Model              →  replaces original LoadSAM3Model
  SAM3BBoxCollector          →  replaces original SAM3BBoxCollector (interactive canvas)
  SAM3PointCollector         →  interactive point prompt collector
  SAM3MultiRegionCollector   →  multi-region prompt collector (points + boxes)
  SAM3VideoSegmentation      →  replaces original SAM3VideoSegmentation
  SAM3Propagate              →  replaces original SAM3Propagate (chunked)
  SAM3VideoOutput            →  replaces original SAM3VideoOutput

Version: 1.1.0
Author:  Claude (Anthropic) for SAM3DBody2abc project
License: Apache 2.0  (SAM3 model code is Meta Platforms Inc., Apache 2.0)
"""

__version__ = "1.1.2"

import os
import importlib.util
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ── Web directory for frontend JS widgets ─────────────────────
WEB_DIRECTORY = "./web"

_dir = os.path.dirname(os.path.abspath(__file__))
_nodes = os.path.join(_dir, "nodes")


def _load_module(name, path):
    """Safely import a module from an absolute path."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[ComfyUI-SAM3-Chunked] Failed to load {name}: {e}")
        traceback.print_exc()
        return None


# ── Model Loader ─────────────────────────────────────────────
_loader = _load_module("sam3c_loader", os.path.join(_nodes, "sam3_model_loader.py"))
if _loader:
    NODE_CLASS_MAPPINGS["LoadSAM3Model"] = _loader.LoadSAM3Model
    NODE_DISPLAY_NAME_MAPPINGS["LoadSAM3Model"] = "🎬 Load SAM3 Model"

# ── BBox Collector (interactive canvas) ──────────────────────
_bbox = _load_module("sam3c_bbox", os.path.join(_nodes, "sam3_bbox_collector.py"))
if _bbox:
    NODE_CLASS_MAPPINGS["SAM3BBoxCollector"] = _bbox.SAM3BBoxCollector
    NODE_DISPLAY_NAME_MAPPINGS["SAM3BBoxCollector"] = "🎬 SAM3 BBox Collector"

# ── Point Collector (interactive canvas) ─────────────────────
_pts = _load_module("sam3c_points", os.path.join(_nodes, "sam3_point_collector.py"))
if _pts:
    NODE_CLASS_MAPPINGS["SAM3PointCollector"] = _pts.SAM3PointCollector
    NODE_DISPLAY_NAME_MAPPINGS["SAM3PointCollector"] = "🎬 SAM3 Point Collector"

# ── Multi-Region Collector (interactive canvas) ──────────────
_multi = _load_module("sam3c_multi", os.path.join(_nodes, "sam3_multiregion_collector.py"))
if _multi:
    NODE_CLASS_MAPPINGS["SAM3MultiRegionCollector"] = _multi.SAM3MultiRegionCollector
    NODE_DISPLAY_NAME_MAPPINGS["SAM3MultiRegionCollector"] = "🎬 SAM3 Multi-Region Collector"

# ── Video Segmentation (init state + prompts) ───────────────
_vidseg = _load_module("sam3c_vidseg", os.path.join(_nodes, "sam3_video_segmentation.py"))
if _vidseg:
    NODE_CLASS_MAPPINGS["SAM3VideoSegmentation"] = _vidseg.SAM3VideoSegmentation
    NODE_DISPLAY_NAME_MAPPINGS["SAM3VideoSegmentation"] = "🎬 SAM3 Video Segmentation"

# ── Propagate (chunked) ─────────────────────────────────────
_prop = _load_module("sam3c_propagate", os.path.join(_nodes, "sam3_propagate.py"))
if _prop:
    NODE_CLASS_MAPPINGS["SAM3Propagate"] = _prop.SAM3Propagate
    NODE_DISPLAY_NAME_MAPPINGS["SAM3Propagate"] = "🎬 SAM3 Propagate (Chunked)"

# ── Video Output ─────────────────────────────────────────────
_output = _load_module("sam3c_output", os.path.join(_nodes, "sam3_video_output.py"))
if _output:
    NODE_CLASS_MAPPINGS["SAM3VideoOutput"] = _output.SAM3VideoOutput
    NODE_DISPLAY_NAME_MAPPINGS["SAM3VideoOutput"] = "🎬 SAM3 Video Output"


# ── Summary ──────────────────────────────────────────────────
print(f"[ComfyUI-SAM3-Chunked] v{__version__} — loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for k in NODE_CLASS_MAPPINGS:
    print(f"  • {NODE_DISPLAY_NAME_MAPPINGS.get(k, k)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
