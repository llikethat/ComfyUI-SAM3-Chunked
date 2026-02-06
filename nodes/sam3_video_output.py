"""
SAM3 Video Output
=================

Convert SAM3_VIDEO_MASKS to standard ComfyUI MASK and IMAGE types.

Compatible with the original ``SAM3VideoOutput`` node interface.
"""

import torch
import numpy as np
from typing import Dict


# ── colour palette for overlays ──────────────────────────────
_PALETTE = {
    0: (0.12, 0.47, 0.71),   # blue
    1: (0.17, 0.63, 0.17),   # green
    2: (0.84, 0.15, 0.16),   # red
    3: (0.58, 0.40, 0.74),   # purple
    4: (1.00, 0.50, 0.05),   # orange
}


class SAM3VideoOutput:
    """Materialise masks from ``SAM3Propagate`` into MASK + IMAGE tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS",),
            },
            "optional": {
                "video_state": ("SAM3_VIDEO_STATE",),
                "scores": ("SAM3_VIDEO_SCORES",),
                "max_objects": ("INT", {"default": -1, "min": -1,
                                        "tooltip": "-1 = all objects"}),
                "binary_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "visualize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "output"
    CATEGORY = "SAM3"

    # ──────────────────────────────────────────────────────────
    def output(
        self,
        masks,
        video_state=None,
        scores=None,
        max_objects=-1,
        binary_threshold=0.5,
        visualize=True,
    ):
        # masks is Dict[int, Tensor] from SAM3Propagate
        if isinstance(masks, dict):
            masks_dict = masks
        else:
            masks_dict = masks

        # Determine dimensions from video_state or first mask
        if video_state is not None:
            orig_h = video_state["orig_height"]
            orig_w = video_state["orig_width"]
            num_frames = video_state["num_frames"]
        else:
            keys = sorted(masks_dict.keys())
            first = masks_dict[keys[0]] if keys else torch.zeros(1, 1)
            orig_h = first.shape[-2] if first.dim() >= 2 else 480
            orig_w = first.shape[-1] if first.dim() >= 2 else 640
            num_frames = max(keys) + 1 if keys else 0

        sorted_keys = sorted(masks_dict.keys())
        n = len(sorted_keys)
        print(f"[SAM3] VideoOutput: assembling {n} masks  ({orig_w}×{orig_h})")

        # ── build binary MASK tensor  (B, H, W) ─────────────
        out_masks = torch.zeros(n, orig_h, orig_w, dtype=torch.float32)
        for i, k in enumerate(sorted_keys):
            m = masks_dict[k].float()
            if m.dim() > 2:
                m = m.squeeze()
            if m.shape[-2:] != (orig_h, orig_w):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0), (orig_h, orig_w),
                    mode="bilinear", align_corners=False,
                ).squeeze()
            out_masks[i] = (m > binary_threshold).float()

        # ── build IMAGE tensors  (B, H, W, 3) float32 0-1 ──
        # frames  = original video
        # visualization = original + coloured mask overlay
        if video_state is not None and "images_np" in video_state:
            imgs = video_state["images_np"]  # uint8 (N, H, W, 3)
            # Build frames for each sorted key
            frame_list = []
            for k in sorted_keys:
                if k < len(imgs):
                    frame_list.append(torch.from_numpy(imgs[k].astype(np.float32) / 255.0))
                else:
                    # Fallback for missing frames
                    frame_list.append(torch.zeros(orig_h, orig_w, 3, dtype=torch.float32))
            frames_out = torch.stack(frame_list) if frame_list else torch.zeros(n, orig_h, orig_w, 3, dtype=torch.float32)
        else:
            frames_out = torch.zeros(n, orig_h, orig_w, 3, dtype=torch.float32)

        if visualize and frames_out.shape[0] == n:
            # Check if masks have any non-zero values
            nonzero_count = (out_masks.sum(dim=(1, 2)) > 0).sum().item()
            print(f"[SAM3] VideoOutput: {nonzero_count}/{n} masks are non-empty, applying visualization")
            vis = self._overlay(frames_out, out_masks)
        else:
            print(f"[SAM3] VideoOutput: visualization disabled or shape mismatch")
            vis = frames_out.clone()

        return (out_masks, frames_out, vis)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _overlay(frames, masks, alpha=0.45, obj_id=0):
        """Overlay a coloured mask on top of the frames."""
        colour = _PALETTE.get(obj_id % len(_PALETTE), (0.2, 0.6, 0.2))
        vis = frames.clone()
        for i in range(len(frames)):
            m = masks[i].unsqueeze(-1)          # (H, W, 1)
            c = torch.tensor(colour, dtype=torch.float32).view(1, 1, 3)
            vis[i] = vis[i] * (1.0 - alpha * m) + c * (alpha * m)
        return vis
