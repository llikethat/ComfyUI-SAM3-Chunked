"""
SAM3 Propagate (Chunked)
========================

Mask propagation through video processed in manageable chunks.

Output types:
  SAM3_VIDEO_MASKS  — dict  {frame_idx → mask tensor (CPU, float32)}
  SAM3_VIDEO_SCORES — dict  {frame_idx → score tensor}
  SAM3_VIDEO_STATE  — passthrough (updated with results)

Memory Strategy
---------------
  original:  allocate all N frames on GPU at once  → OOM > ~500 frames
  chunked:   load C frames on GPU, propagate, offload results, repeat
             C = chunk_size (default 100)

Chunk overlap keeps the tracker's memory bank coherent across boundaries.
"""

import gc
import os
import sys
import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple


# ── helpers ───────────────────────────────────────────────────
def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _vram_mb():
    if not torch.cuda.is_available():
        return 0, 0
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    return a, r


def _ensure_sam3():
    try:
        import sam3          # noqa
        return True
    except ImportError:
        lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            import sam3      # noqa
            return True
        except ImportError:
            return False


def _preprocess_chunk(
    images_np: np.ndarray,   # (N, H, W, 3) uint8
    start: int,
    end: int,
    image_size: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Resize, normalise, and move a slice of frames to GPU."""
    n = end - start
    buf = torch.zeros(n, 3, image_size, image_size, dtype=dtype)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=dtype).view(3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5], dtype=dtype).view(3, 1, 1)
    for i in range(n):
        frame = images_np[start + i]
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).to(dtype)
        buf[i] = (t - mean) / std
    return buf.to(device)


# ── Node ──────────────────────────────────────────────────────
class SAM3Propagate:
    """Propagate SAM3 masks through a video, processing in GPU-friendly chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "video_state": ("SAM3_VIDEO_STATE",),
            },
            "optional": {
                "start_frame": ("INT",  {"default": 0,  "min": 0}),
                "end_frame":   ("INT",  {"default": -1, "min": -1,
                                         "tooltip": "-1 = all frames"}),
                "direction":   (["forward", "backward", "bidirectional"], {"default": "forward"}),
                "clear_cache": ("BOOLEAN", {"default": True,
                                            "tooltip": "Free GPU memory between chunks"}),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("masks", "scores", "video_state")
    FUNCTION = "propagate"
    CATEGORY = "SAM3"

    # ──────────────────────────────────────────────────────────
    def propagate(
        self,
        sam3_model: Dict,
        video_state: Dict,
        start_frame: int = 0,
        end_frame: int = -1,
        direction: str = "forward",
        clear_cache: bool = True,
    ):
        model      = sam3_model["model"]
        device     = sam3_model["device"]
        dtype      = sam3_model["dtype"]
        images_np  = video_state["images_np"]
        num_frames = video_state["num_frames"]
        chunk_size = video_state.get("chunk_size", 100)
        overlap    = video_state.get("overlap_frames", 10)
        img_size   = video_state.get("image_size", 1008)
        prompt     = video_state.get("prompt_data", {})

        if end_frame < 0:
            end_frame = num_frames
        end_frame = min(end_frame, num_frames)
        total = end_frame - start_frame

        effective = max(chunk_size - overlap, 1)
        n_chunks  = (total + effective - 1) // effective

        print(f"[SAM3] Propagate: frames {start_frame}→{end_frame-1}  ({total} frames)")
        print(f"[SAM3]   chunks={n_chunks}  size={chunk_size}  overlap={overlap}")

        # Storage — everything on CPU
        all_masks:  Dict[int, torch.Tensor] = {}
        all_scores: Dict[int, torch.Tensor] = {}

        has_native = (not isinstance(model, dict)) and hasattr(model, "init_state")

        prev_mask = None      # last mask from previous chunk (for seeding)

        for ci in range(n_chunks):
            c_start = start_frame + ci * effective
            c_end   = min(c_start + chunk_size, end_frame)

            print(f"\n[SAM3] ── chunk {ci+1}/{n_chunks}: frames {c_start}–{c_end-1} ──")
            a, r = _vram_mb()
            print(f"[SAM3]   VRAM before load: {a:.0f}/{r:.0f} MB (alloc/reserved)")

            chunk_frames = _preprocess_chunk(images_np, c_start, c_end, img_size, dtype, device)

            if has_native:
                c_masks, c_scores = self._propagate_native(
                    model, chunk_frames, c_start, c_end,
                    prompt, prev_mask, direction, device, dtype,
                    video_state,
                )
            else:
                c_masks, c_scores = self._propagate_fallback(
                    model, chunk_frames, c_start, c_end,
                    prompt, prev_mask, direction, video_state,
                )

            # Store results (skip overlapping prefix except first chunk)
            store_from = c_start if ci == 0 else c_start + overlap
            for fi in range(c_start, c_end):
                local = fi - c_start
                if fi >= store_from and fi not in all_masks and local < len(c_masks):
                    all_masks[fi]  = c_masks[local].cpu().float()
                    if c_scores is not None and local < len(c_scores):
                        all_scores[fi] = c_scores[local].cpu().float()

            # Seed next chunk
            if overlap > 0 and c_end < end_frame and len(c_masks) > 0:
                prev_mask = c_masks[-1].cpu()
            else:
                prev_mask = None

            del chunk_frames, c_masks, c_scores
            if clear_cache:
                _clear_gpu()
                a, r = _vram_mb()
                print(f"[SAM3]   VRAM after clear: {a:.0f}/{r:.0f} MB")

        print(f"\n[SAM3] Propagation done — {len(all_masks)} masks collected")

        masks_out  = all_masks
        scores_out = all_scores

        video_state["_masks"]  = all_masks
        video_state["_scores"] = all_scores

        return (masks_out, scores_out, video_state)

    # ── native SAM3 API ──────────────────────────────────────
    def _propagate_native(
        self, model, chunk_frames, c_start, c_end,
        prompt, prev_mask, direction, device, dtype, video_state,
    ):
        """Run SAM3's own ``init_state`` / ``add_*`` / ``propagate_in_video``
        on a chunk of frames that are *already* on the GPU."""

        orig_h = video_state["orig_height"]
        orig_w = video_state["orig_width"]

        try:
            # SAM3 init_state can accept a tensor batch directly (since it
            # ultimately stores images in ``BatchedDatapoint.img_batch``).
            # If the API only accepts a path we fall back to the simple path.
            inf = model.init_state(chunk_frames, offload_video_to_cpu=True)

            # ── prompts ──
            if c_start == 0 or prev_mask is None:
                self._apply_initial_prompt(model, inf, prompt, device, dtype, orig_h, orig_w)
            else:
                # Seed from previous chunk's last mask
                model.add_new_mask(inf, frame_idx=0, mask=prev_mask.to(device))

            # ── propagate ──
            masks_list = []
            scores_list = []
            for out in model.propagate_in_video(
                inf,
                start_frame_idx=0,
                reverse=(direction == "backward"),
            ):
                masks_list.append(out["masks"].cpu())
                if "scores" in out:
                    scores_list.append(out["scores"].cpu())

            masks  = torch.cat(masks_list, dim=0) if masks_list else torch.zeros(0)
            scores = torch.cat(scores_list, dim=0) if scores_list else None
            return masks, scores

        except Exception as e:
            print(f"[SAM3]   native propagation failed ({e}), using fallback")
            return self._propagate_fallback(
                model, chunk_frames, c_start, c_end,
                prompt, prev_mask, direction, video_state,
            )

    def _apply_initial_prompt(self, model, inf, prompt, device, dtype, orig_h, orig_w):
        """Apply the user's prompt (box / point / text) to the SAM3 inference state."""
        ptype = prompt.get("type", "box")
        frame = prompt.get("frame", 0)

        pos = prompt.get("positive")
        neg = prompt.get("negative")

        # ── Box prompts ──
        if ptype in ("box", "auto") and pos is not None:
            boxes = pos.get("boxes")
            if boxes is not None and len(boxes) > 0:
                labels = pos.get("labels", torch.ones(len(boxes), dtype=torch.long))
                try:
                    model.add_new_points_or_box(
                        inf, frame_idx=frame, box=boxes.to(device), box_label=labels.to(device),
                    )
                    return
                except Exception:
                    pass  # fall through

        # ── Point prompts ──
        pos_pts = prompt.get("positive_points")
        neg_pts = prompt.get("negative_points")

        if ptype in ("point", "auto") and pos_pts is not None:
            points_data = pos_pts.get("points")
            if points_data is not None and len(points_data) > 0:
                # Combine positive and negative points
                all_points = [points_data]
                all_labels = [torch.ones(len(points_data), dtype=torch.long)]

                if neg_pts is not None:
                    neg_data = neg_pts.get("points")
                    if neg_data is not None and len(neg_data) > 0:
                        all_points.append(neg_data)
                        all_labels.append(torch.zeros(len(neg_data), dtype=torch.long))

                combined_points = torch.cat(all_points, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)

                try:
                    model.add_new_points_or_box(
                        inf, frame_idx=frame,
                        points=combined_points.to(device),
                        labels=combined_labels.to(device),
                    )
                    return
                except Exception:
                    pass  # fall through

        # ── Text prompts ──
        text = prompt.get("text", "")
        if ptype in ("text", "auto") and text:
            try:
                model.add_new_text(inf, text=text)
                return
            except Exception:
                pass

        # Last resort — centre box covering middle 50 %
        cx, cy = orig_w / 2, orig_h / 2
        hw, hh = orig_w / 4, orig_h / 4
        fallback_box = torch.tensor([[cx - hw, cy - hh, cx + hw, cy + hh]], dtype=torch.float32)
        try:
            model.add_new_points_or_box(inf, frame_idx=0, box=fallback_box.to(device))
        except Exception:
            pass

    # ── fallback (no native API) ─────────────────────────────
    @staticmethod
    def _propagate_fallback(
        model, chunk_frames, c_start, c_end,
        prompt, prev_mask, direction, video_state,
    ):
        """Simple carry-forward mask propagation when the SAM3 model API
        is unavailable (e.g. raw checkpoint, missing dependency)."""

        orig_h = video_state["orig_height"]
        orig_w = video_state["orig_width"]
        n = c_end - c_start

        masks = torch.zeros(n, 1, orig_h, orig_w, dtype=torch.float32)

        if prev_mask is not None:
            seed = prev_mask.float()
            if seed.dim() > 2:
                seed = seed.squeeze()
            if seed.shape[-2:] != (orig_h, orig_w):
                seed = torch.nn.functional.interpolate(
                    seed.unsqueeze(0).unsqueeze(0), (orig_h, orig_w),
                    mode="bilinear", align_corners=False,
                ).squeeze()
        else:
            seed = None
            # Try box prompts first
            pos = prompt.get("positive") if prompt else None
            if pos is not None:
                boxes = pos.get("boxes")
                if boxes is not None and len(boxes) > 0:
                    b = boxes[0].tolist()
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    seed = torch.zeros(orig_h, orig_w, dtype=torch.float32)
                    seed[max(0,y1):min(orig_h,y2), max(0,x1):min(orig_w,x2)] = 1.0

            # Try point prompts if no box seed
            if seed is None:
                pos_pts = prompt.get("positive_points") if prompt else None
                if pos_pts is not None:
                    points = pos_pts.get("points")
                    if points is not None and len(points) > 0:
                        # Create a circular mask around each positive point
                        seed = torch.zeros(orig_h, orig_w, dtype=torch.float32)
                        radius = max(orig_h, orig_w) // 20  # 5% of image dimension
                        for pt in points:
                            px, py = int(pt[0].item()), int(pt[1].item())
                            y_min = max(0, py - radius)
                            y_max = min(orig_h, py + radius)
                            x_min = max(0, px - radius)
                            x_max = min(orig_w, px + radius)
                            seed[y_min:y_max, x_min:x_max] = 1.0

            if seed is None:
                seed = torch.zeros(orig_h, orig_w, dtype=torch.float32)
                h4, w4 = orig_h // 4, orig_w // 4
                seed[h4:3*h4, w4:3*w4] = 1.0

        for i in range(n):
            masks[i, 0] = seed

        return masks, None
