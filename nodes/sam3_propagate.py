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
from PIL import Image
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


def _np_to_pil_list(images_np: np.ndarray, start: int, end: int) -> List[Image.Image]:
    """Convert a slice of numpy frames to a list of PIL Images for SAM3 init_state."""
    pil_list = []
    for i in range(start, end):
        frame = images_np[i]  # (H, W, 3) uint8
        pil_list.append(Image.fromarray(frame))
    return pil_list


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

            if has_native:
                # Pass images_np - the native method will convert to PIL and handle GPU
                c_masks, c_scores = self._propagate_native(
                    model, images_np, c_start, c_end,
                    prompt, prev_mask, direction, device, dtype,
                    video_state,
                )
            else:
                c_masks, c_scores = self._propagate_fallback(
                    model, images_np, c_start, c_end,
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

            del c_masks, c_scores
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
        self, model, images_np, c_start, c_end,
        prompt, prev_mask, direction, device, dtype, video_state,
    ):
        """Run SAM3's own ``init_state`` / ``add_prompt`` / ``propagate_in_video``
        on a chunk of frames."""

        orig_h = video_state["orig_height"]
        orig_w = video_state["orig_width"]
        n_frames = c_end - c_start

        try:
            # Convert numpy frames to PIL Images (SAM3 init_state accepts list of PIL)
            pil_frames = _np_to_pil_list(images_np, c_start, c_end)
            
            # Initialize SAM3 inference state with PIL images
            inf = model.init_state(pil_frames, offload_video_to_cpu=True)

            # ── prompts ──
            if c_start == 0 or prev_mask is None:
                self._apply_initial_prompt(model, inf, prompt, device, dtype, orig_h, orig_w)
            else:
                # Seed from previous chunk's last mask - try to add as box prompt
                try:
                    # Find bounding box of previous mask
                    mask_np = prev_mask.cpu().numpy() if torch.is_tensor(prev_mask) else prev_mask
                    if mask_np.ndim > 2:
                        mask_np = mask_np.squeeze()
                    
                    # Get bounding box from mask
                    ys, xs = np.where(mask_np > 0.5)
                    if len(ys) > 0:
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        # Normalize to 0-1
                        box_xywh = torch.tensor([[
                            x1 / orig_w,
                            y1 / orig_h,
                            (x2 - x1) / orig_w,
                            (y2 - y1) / orig_h
                        ]], dtype=torch.float32)
                        model.add_prompt(inf, frame_idx=0, boxes_xywh=box_xywh, box_labels=torch.tensor([1]))
                    else:
                        # Fallback: re-apply initial prompt
                        self._apply_initial_prompt(model, inf, prompt, device, dtype, orig_h, orig_w)
                except Exception as e:
                    print(f"[SAM3]   Could not add previous mask: {e}")
                    # Fallback: re-apply initial prompt
                    self._apply_initial_prompt(model, inf, prompt, device, dtype, orig_h, orig_w)

            # ── propagate ──
            masks_list = []
            scores_list = []
            
            for frame_idx, out in model.propagate_in_video(
                inf,
                start_frame_idx=0,
                max_frame_num_to_track=n_frames,
                reverse=(direction == "backward"),
            ):
                if out is None:
                    # No output for this frame - add placeholder
                    masks_list.append(torch.zeros(orig_h, orig_w, dtype=torch.float32))
                    scores_list.append(torch.tensor(0.0))
                    continue
                    
                # out contains: out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks
                binary_masks = out.get("out_binary_masks")  # numpy (N_objects, H, W) bool
                probs = out.get("out_probs")  # numpy (N_objects,)
                
                if binary_masks is not None and len(binary_masks) > 0:
                    # Combine all object masks into one (union)
                    combined_mask = np.any(binary_masks, axis=0)  # (H, W)
                    mask_tensor = torch.from_numpy(combined_mask.astype(np.float32))
                    
                    # Resize to original resolution if needed
                    mask_h, mask_w = mask_tensor.shape
                    if mask_h != orig_h or mask_w != orig_w:
                        mask_tensor = torch.nn.functional.interpolate(
                            mask_tensor.unsqueeze(0).unsqueeze(0),
                            size=(orig_h, orig_w),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze()
                    
                    masks_list.append(mask_tensor)
                    scores_list.append(torch.tensor(probs.max() if probs is not None and len(probs) > 0 else 1.0))
                else:
                    # No masks for this frame - add placeholder
                    masks_list.append(torch.zeros(orig_h, orig_w, dtype=torch.float32))
                    scores_list.append(torch.tensor(0.0))

            if masks_list:
                masks = torch.stack(masks_list, dim=0).unsqueeze(1)  # (N, 1, H, W)
                scores = torch.stack(scores_list) if scores_list else None
                # Debug: check if we got real masks
                nonzero_masks = sum(1 for m in masks_list if m.sum() > 0)
                print(f"[SAM3]   Generated {len(masks_list)} masks, {nonzero_masks} non-empty")
            else:
                masks = torch.zeros(n_frames, 1, orig_h, orig_w, dtype=torch.float32)
                scores = None
                print(f"[SAM3]   WARNING: No masks generated!")
                
            return masks, scores

        except Exception as e:
            print(f"[SAM3]   native propagation failed ({e}), using fallback")
            import traceback
            traceback.print_exc()
            return self._propagate_fallback(
                model, images_np, c_start, c_end,
                prompt, prev_mask, direction, video_state,
            )

    def _apply_initial_prompt(self, model, inf, prompt, device, dtype, orig_h, orig_w):
        """Apply the user's prompt (box / point / text) to the SAM3 inference state.
        
        SAM3's add_prompt expects:
        - text_str: optional text prompt
        - boxes_xywh: optional boxes in [xmin, ymin, width, height] NORMALIZED (0-1)
        - box_labels: labels for boxes (1=positive, 0=negative)
        """
        ptype = prompt.get("type", "box")
        frame = prompt.get("frame", 0)

        pos = prompt.get("positive")
        neg = prompt.get("negative")
        
        # Prepare box prompts (normalized to 0-1)
        boxes_xywh = None
        box_labels = None
        
        if ptype in ("box", "auto") and pos is not None:
            boxes = pos.get("boxes")  # Expected: (N, 4) in xyxy format, pixel coords
            if boxes is not None and len(boxes) > 0:
                # Convert xyxy to xywh and normalize
                boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
                boxes_xywh_list = []
                labels_list = []
                
                for box in boxes_np:
                    x1, y1, x2, y2 = box
                    # Convert to normalized xywh
                    w = x2 - x1
                    h = y2 - y1
                    # Normalize to 0-1
                    boxes_xywh_list.append([
                        x1 / orig_w,
                        y1 / orig_h,
                        w / orig_w,
                        h / orig_h
                    ])
                    labels_list.append(1)  # positive
                
                boxes_xywh = torch.tensor(boxes_xywh_list, dtype=torch.float32)
                box_labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Get text prompt
        text = prompt.get("text", "")
        text_str = text if ptype in ("text", "auto") and text else None
        
        # Call SAM3's add_prompt
        try:
            if boxes_xywh is not None or text_str:
                model.add_prompt(
                    inf,
                    frame_idx=frame,
                    text_str=text_str,
                    boxes_xywh=boxes_xywh,
                    box_labels=box_labels,
                )
                return
        except Exception as e:
            print(f"[SAM3]   add_prompt failed: {e}")
        
        # Fallback: centre box covering middle 50%
        fallback_box = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)  # normalized xywh
        try:
            model.add_prompt(inf, frame_idx=0, boxes_xywh=fallback_box, box_labels=torch.tensor([1]))
        except Exception as e:
            print(f"[SAM3]   fallback prompt also failed: {e}")

    # ── fallback (no native API) ─────────────────────────────
    @staticmethod
    def _propagate_fallback(
        model, images_np, c_start, c_end,
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
