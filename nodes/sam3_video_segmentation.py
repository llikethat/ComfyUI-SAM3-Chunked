"""
SAM3 Video Segmentation (Chunked)
==================================

Initialize SAM3 video state *without* loading every frame to GPU.
Output type: SAM3_VIDEO_STATE — same as the original node.

Key Difference
--------------
Original ``SAM3VideoSegmentation`` calls ``model.init_state(video_path)``
which pre-allocates ``N×3×H×W`` on GPU (≈9 GB for 1500 frames at 1008²).
This version stores images as a NumPy array on *CPU* and only sends a
chunk of frames to the GPU during ``SAM3Propagate``.
"""

import gc
import os
import sys
import torch
import numpy as np
import cv2
from typing import Dict, Any, Optional


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_sam3():
    try:
        import sam3
        return True
    except ImportError:
        lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            import sam3
            return True
        except ImportError:
            return False


class SAM3VideoSegmentation:
    """Create a SAM3 video state from input frames and optional prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Batch of video frames (B, H, W, C) float32 0-1"
                }),
            },
            "optional": {
                "positive_boxes": ("SAM3_BOXES_PROMPT", {"tooltip": "Positive box prompts"}),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {"tooltip": "Negative box prompts"}),
                "prompt_type": (["box", "text", "auto"], {"default": "box"}),
                "text_prompt": ("STRING", {"default": "person", "tooltip": "Text prompt for segmentation"}),
                "prompt_frame": ("INT", {"default": 0, "min": 0}),
                # ── chunked-specific ──
                "chunk_size": ("INT", {
                    "default": 100, "min": 10, "max": 1000, "step": 10,
                    "tooltip": "Frames per GPU chunk (lower = less VRAM)"
                }),
                "overlap_frames": ("INT", {
                    "default": 10, "min": 0, "max": 100,
                    "tooltip": "Overlap between chunks for mask continuity"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "init_state"
    CATEGORY = "SAM3"

    def init_state(
        self,
        video_frames,
        positive_boxes=None,
        negative_boxes=None,
        prompt_type="box",
        text_prompt="person",
        prompt_frame=0,
        chunk_size=100,
        overlap_frames=10,
    ):
        # ── Convert to uint8 NumPy on CPU ────────────────────
        if torch.is_tensor(video_frames):
            arr = video_frames.cpu().numpy()
        else:
            arr = np.asarray(video_frames)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        num_frames = arr.shape[0]
        orig_h, orig_w = arr.shape[1], arr.shape[2]

        frame_mb = (3 * 1008 * 1008 * 2) / (1024 * 1024)  # float16 per frame
        total_mb = num_frames * frame_mb
        chunk_mb = chunk_size * frame_mb

        print(f"[SAM3] VideoSegmentation: {num_frames} frames  {orig_w}×{orig_h}")
        print(f"[SAM3]   chunk={chunk_size}  overlap={overlap_frames}")
        print(f"[SAM3]   VRAM per chunk ≈ {chunk_mb:.0f} MB  (full video would be ≈ {total_mb:.0f} MB)")

        # ── Build prompt data ────────────────────────────────
        prompt_data = {
            "type": prompt_type,
            "text": text_prompt,
            "frame": prompt_frame,
            "positive": positive_boxes,
            "negative": negative_boxes,
        }

        # ── Build video state ────────────────────────────────
        state = {
            "images_np": arr,                # uint8 (B, H, W, 3) — CPU
            "num_frames": num_frames,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "image_size": 1008,              # SAM3 default resize target
            "chunk_size": chunk_size,
            "overlap_frames": overlap_frames,
            "prompt_data": prompt_data,
            "_chunked": True,                # flag so Propagate knows
        }

        return (state,)
