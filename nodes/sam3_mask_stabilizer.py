"""
SAM3 Mask Temporal Stabilizer
=============================

Fix edge flicker and temporal instability in SAM3 video masks.

Pipeline (Quality Mode):
    1. Logit/Confidence Smoothing (EMA on soft masks) - BOUNDARY ONLY
    2. Optical Flow Warping (temporal consistency) - BOUNDARY ONLY
    3. Distance Field Smoothing (smooth edges via SDF) - BOUNDARY ONLY
    4. Temporal Hysteresis (stable pixels unless confidence changes)
    5. Final Binarization (clean up any ghosting)

Usage:
    SAM3Propagate → SAM3MaskTemporalStabilizer → SAM3VideoOutput
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List, Union, Any

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _to_tensor(x) -> torch.Tensor:
    """Convert numpy array or tensor to torch.Tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        return torch.tensor(x, dtype=torch.float32)


def _get_video_state_attr(video_state: Any, key: str, default=None):
    """Get attribute from video_state regardless of whether it's a dict or dataclass."""
    if isinstance(video_state, dict):
        return video_state.get(key, default)
    
    if hasattr(video_state, key):
        return getattr(video_state, key, default)
    
    key_mapping = {
        "orig_height": "height",
        "orig_width": "width",
        "height": "orig_height",
        "width": "orig_width",
    }
    
    if key in key_mapping:
        alt_key = key_mapping[key]
        if hasattr(video_state, alt_key):
            return getattr(video_state, alt_key, default)
    
    return default


def _get_images_from_state(video_state: Any) -> Optional[np.ndarray]:
    """Get images array from video_state."""
    if isinstance(video_state, dict):
        return video_state.get("images_np")
    
    if hasattr(video_state, 'temp_dir'):
        temp_dir = video_state.temp_dir
        num_frames = getattr(video_state, 'num_frames', 0)
        
        if temp_dir and num_frames > 0:
            import os
            from PIL import Image
            
            frames = []
            for i in range(num_frames):
                frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    frames.append(np.array(img))
            
            if frames:
                return np.stack(frames, axis=0)
    
    return None


def _get_boundary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Get binary mask of boundary region."""
    mask_bool = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
    dilated = cv2.dilate(mask_bool, kernel)
    eroded = cv2.erode(mask_bool, kernel)
    return (dilated != eroded)


class SAM3MaskTemporalStabilizer:
    """Temporal stabilization for SAM3 video masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS",),
                "video_state": ("SAM3_VIDEO_STATE",),
            },
            "optional": {
                "enable_confidence_smoothing": ("BOOLEAN", {"default": True}),
                "confidence_ema_alpha": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA smoothing factor. Lower = more temporal smoothing"
                }),
                
                "enable_flow_warping": ("BOOLEAN", {"default": True}),
                "flow_blend_alpha": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend factor between warped previous and current mask"
                }),
                "optical_flow": ("OPTICAL_FLOW", {
                    "tooltip": "Optional: External flow from RAFT/GMFlow node"
                }),
                
                "enable_distance_field": ("BOOLEAN", {"default": True}),
                "sdf_smoothing_sigma": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Gaussian sigma for SDF temporal smoothing"
                }),
                
                "enable_hysteresis": ("BOOLEAN", {"default": True}),
                "hysteresis_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Minimum confidence change to flip a pixel"
                }),
                
                "enable_birefnet": ("BOOLEAN", {"default": False,
                    "tooltip": "Use BiRefNet for edge refinement (requires ComfyUI-BiRefNet)"
                }),
                "birefnet_model": ("BIREFNET_MODEL", {
                    "tooltip": "Optional: BiRefNet model from Load BiRefNet node"
                }),
                
                "boundary_pixels": ("INT", {"default": 20, "min": 5, "max": 50,
                    "tooltip": "Pixel width of boundary region to stabilize"
                }),
                
                "binarize_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Final binarization threshold to remove ghosting"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE", "IMAGE")
    RETURN_NAMES = ("stabilized_masks", "scores", "video_state", "debug_visualization")
    FUNCTION = "stabilize"
    CATEGORY = "SAM3"

    def stabilize(
        self,
        masks: Dict[int, torch.Tensor],
        video_state,
        enable_confidence_smoothing: bool = True,
        confidence_ema_alpha: float = 0.3,
        enable_flow_warping: bool = True,
        flow_blend_alpha: float = 0.5,
        optical_flow: Optional[torch.Tensor] = None,
        enable_distance_field: bool = True,
        sdf_smoothing_sigma: float = 1.5,
        enable_hysteresis: bool = True,
        hysteresis_threshold: float = 0.15,
        enable_birefnet: bool = False,
        birefnet_model = None,
        boundary_pixels: int = 20,
        binarize_threshold: float = 0.5,
    ):
        """Apply multi-stage temporal stabilization to masks - BOUNDARY ONLY to prevent ghosting."""
        
        images_np = _get_images_from_state(video_state)
        orig_h = _get_video_state_attr(video_state, "orig_height") or _get_video_state_attr(video_state, "height", 480)
        orig_w = _get_video_state_attr(video_state, "orig_width") or _get_video_state_attr(video_state, "width", 640)
        
        sorted_keys = sorted(masks.keys())
        n_frames = len(sorted_keys)
        
        print(f"[SAM3 Stabilizer] Processing {n_frames} frames ({orig_w}x{orig_h})")
        
        # Convert to tensor and binarize input
        mask_tensor = self._dict_to_tensor(masks, sorted_keys, orig_h, orig_w)
        original_binary = (mask_tensor > 0.5).float()  # Keep original binary for interior
        
        # Working copy for boundary smoothing
        working_masks = mask_tensor.clone()
        
        # Pre-compute boundary masks for all frames
        boundary_masks = torch.zeros_like(mask_tensor, dtype=torch.bool)
        for t in range(n_frames):
            boundary_masks[t] = torch.from_numpy(_get_boundary_mask(mask_tensor[t].numpy(), boundary_pixels))
        
        # Stage 1: Confidence/Logit EMA Smoothing (BOUNDARY ONLY)
        if enable_confidence_smoothing:
            print(f"[SAM3 Stabilizer] Stage 1: Confidence EMA (alpha={confidence_ema_alpha})")
            smoothed = self._apply_confidence_ema(working_masks, alpha=confidence_ema_alpha)
            # Apply only to boundaries
            for t in range(n_frames):
                working_masks[t] = torch.where(boundary_masks[t], smoothed[t], original_binary[t])
        
        # Stage 2: Optical Flow Warping (BOUNDARY ONLY)
        if enable_flow_warping and images_np is not None:
            print(f"[SAM3 Stabilizer] Stage 2: Optical Flow Warping (blend={flow_blend_alpha})")
            if optical_flow is not None:
                flow_result = self._apply_external_flow(working_masks, optical_flow, blend_alpha=flow_blend_alpha)
            else:
                flow_result = self._apply_farneback_flow(working_masks, images_np, sorted_keys, blend_alpha=flow_blend_alpha)
            # Apply only to boundaries
            for t in range(n_frames):
                working_masks[t] = torch.where(boundary_masks[t], flow_result[t], original_binary[t])
        elif enable_flow_warping and images_np is None:
            print(f"[SAM3 Stabilizer] Stage 2: Skipped (no images available)")
        
        # Stage 3: Distance Field Smoothing (BOUNDARY ONLY)
        if enable_distance_field and HAS_SCIPY:
            print(f"[SAM3 Stabilizer] Stage 3: Distance Field (sigma={sdf_smoothing_sigma})")
            sdf_result = self._apply_distance_field_smoothing(working_masks, sigma=sdf_smoothing_sigma, boundary_pixels=boundary_pixels)
            # Apply only to boundaries
            for t in range(n_frames):
                working_masks[t] = torch.where(boundary_masks[t], sdf_result[t], original_binary[t])
        
        # Stage 4: Temporal Hysteresis
        if enable_hysteresis:
            print(f"[SAM3 Stabilizer] Stage 4: Temporal Hysteresis (threshold={hysteresis_threshold})")
            working_masks = self._apply_hysteresis(working_masks, original_binary, threshold=hysteresis_threshold)
        
        # Stage 5: BiRefNet Edge Refinement (Optional)
        if enable_birefnet and birefnet_model is not None and images_np is not None:
            print(f"[SAM3 Stabilizer] Stage 5: BiRefNet Refinement")
            working_masks = self._apply_birefnet_refinement(working_masks, images_np, sorted_keys, birefnet_model)
        
        # Stage 6: FINAL BINARIZATION - Remove all ghosting
        print(f"[SAM3 Stabilizer] Stage 6: Final Binarization (threshold={binarize_threshold})")
        final_masks = (working_masks > binarize_threshold).float()
        
        stabilized_masks = self._tensor_to_dict(final_masks, sorted_keys)
        
        if images_np is not None:
            debug_vis = self._create_debug_visualization(original_binary, final_masks, images_np, sorted_keys)
        else:
            debug_vis = self._create_simple_debug_visualization(original_binary, final_masks, sorted_keys)
        
        scores = {k: torch.tensor(1.0) for k in sorted_keys}
        
        print(f"[SAM3 Stabilizer] Complete!")
        return (stabilized_masks, scores, video_state, debug_vis)

    def _dict_to_tensor(self, masks: Dict, keys: List[int], H: int, W: int) -> torch.Tensor:
        """Convert mask dict to (N, H, W) tensor."""
        out = torch.zeros(len(keys), H, W, dtype=torch.float32)
        for i, k in enumerate(keys):
            m = _to_tensor(masks[k])
            if m.dim() > 2:
                m = m.squeeze()
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W), mode="bilinear", align_corners=False).squeeze()
            out[i] = m
        return out
    
    def _tensor_to_dict(self, tensor: torch.Tensor, keys: List[int]) -> Dict[int, torch.Tensor]:
        """Convert (N, H, W) tensor back to dict."""
        return {k: tensor[i] for i, k in enumerate(keys)}

    def _apply_confidence_ema(self, masks: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply Exponential Moving Average to soft masks."""
        N, H, W = masks.shape
        smoothed = masks.clone()
        
        for t in range(1, N):
            smoothed[t] = alpha * masks[t] + (1 - alpha) * smoothed[t-1]
        
        backward = masks.clone()
        for t in range(N - 2, -1, -1):
            backward[t] = alpha * masks[t] + (1 - alpha) * backward[t+1]
        
        return 0.5 * (smoothed + backward)

    def _apply_farneback_flow(self, masks: torch.Tensor, images_np: np.ndarray, keys: List[int], blend_alpha: float) -> torch.Tensor:
        """Apply Farneback optical flow for temporal consistency."""
        N, H, W = masks.shape
        stabilized = masks.clone()
        
        for t in range(1, N):
            k_prev, k_curr = keys[t-1], keys[t]
            
            if k_prev >= len(images_np) or k_curr >= len(images_np):
                continue
            
            prev_gray = cv2.cvtColor(images_np[k_prev], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(images_np[k_curr], cv2.COLOR_RGB2GRAY)
            
            if prev_gray.shape != (H, W):
                prev_gray = cv2.resize(prev_gray, (W, H))
                curr_gray = cv2.resize(curr_gray, (W, H))
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            prev_mask = stabilized[t-1].numpy()
            warped = self._warp_with_flow(prev_mask, flow)
            
            current = masks[t].numpy()
            blended = blend_alpha * current + (1 - blend_alpha) * warped
            stabilized[t] = torch.from_numpy(blended.astype(np.float32))
        
        return stabilized
    
    def _apply_external_flow(self, masks: torch.Tensor, flow: torch.Tensor, blend_alpha: float) -> torch.Tensor:
        """Apply pre-computed optical flow from RAFT/GMFlow."""
        N, H, W = masks.shape
        stabilized = masks.clone()
        
        if flow.shape[-1] == 2:
            flow = flow.permute(0, 3, 1, 2)
        
        for t in range(1, min(N, flow.shape[0] + 1)):
            f = flow[t-1]
            
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f.unsqueeze(0), (H, W), mode="bilinear", align_corners=False).squeeze(0)
                f[0] *= W / flow.shape[-1]
                f[1] *= H / flow.shape[-2]
            
            prev_mask = stabilized[t-1].unsqueeze(0).unsqueeze(0)
            warped = self._warp_tensor_with_flow(prev_mask, f)
            
            stabilized[t] = blend_alpha * masks[t] + (1 - blend_alpha) * warped.squeeze()
        
        return stabilized
    
    def _warp_with_flow(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp mask using optical flow (numpy)."""
        H, W = mask.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)
        return cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def _warp_tensor_with_flow(self, mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp mask tensor using optical flow (PyTorch grid_sample)."""
        B, C, H, W = mask.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        
        flow_norm = torch.zeros_like(grid)
        flow_norm[..., 0] = flow[0] / (W / 2)
        flow_norm[..., 1] = flow[1] / (H / 2)
        
        return F.grid_sample(mask, grid + flow_norm, mode='bilinear', padding_mode='border', align_corners=True)

    def _apply_distance_field_smoothing(self, masks: torch.Tensor, sigma: float, boundary_pixels: int) -> torch.Tensor:
        """Smooth masks via signed distance field with temporal smoothing."""
        if not HAS_SCIPY:
            return masks
            
        N, H, W = masks.shape
        
        # Convert each mask to SDF
        sdf_stack = torch.zeros(N, H, W)
        for t in range(N):
            sdf_stack[t] = torch.from_numpy(self._mask_to_sdf(masks[t].numpy()))
        
        # Temporal smoothing on SDF
        if sigma > 0:
            kernel_size = int(6 * sigma) | 1
            half_k = kernel_size // 2
            kernel = self._gaussian_kernel_1d(kernel_size, sigma)
            
            sdf_smoothed = torch.zeros_like(sdf_stack)
            for t in range(N):
                weighted_sum = torch.zeros(H, W)
                weight_total = 0.0
                for k_idx, k_weight in enumerate(kernel):
                    src_t = t - half_k + k_idx
                    if 0 <= src_t < N:
                        weighted_sum += k_weight * sdf_stack[src_t]
                        weight_total += k_weight
                if weight_total > 0:
                    sdf_smoothed[t] = weighted_sum / weight_total
                else:
                    sdf_smoothed[t] = sdf_stack[t]
        else:
            sdf_smoothed = sdf_stack
        
        # Threshold SDF to get binary mask
        return (sdf_smoothed > 0).float()
    
    def _mask_to_sdf(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to signed distance field."""
        mask_bool = mask > 0.5
        dist_inside = ndimage.distance_transform_edt(mask_bool)
        dist_outside = ndimage.distance_transform_edt(~mask_bool)
        return (dist_inside - dist_outside).astype(np.float32)
    
    def _gaussian_kernel_1d(self, size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        x = torch.arange(size) - size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        return kernel / kernel.sum()

    def _apply_hysteresis(self, smoothed: torch.Tensor, original: torch.Tensor, threshold: float) -> torch.Tensor:
        """Keep pixels stable unless confidence changes significantly."""
        N, H, W = smoothed.shape
        result = smoothed.clone()
        
        for t in range(1, N):
            diff = torch.abs(smoothed[t] - result[t-1])
            stable_mask = diff < threshold
            result[t] = torch.where(stable_mask, result[t-1], smoothed[t])
        
        return result

    def _apply_birefnet_refinement(self, masks: torch.Tensor, images_np: np.ndarray, keys: List[int], birefnet_model) -> torch.Tensor:
        """Use BiRefNet for edge refinement."""
        try:
            if birefnet_model is None:
                return masks
            
            N, H, W = masks.shape
            refined = masks.clone()
            
            for t, k in enumerate(keys):
                if k >= len(images_np):
                    continue
                
                image = images_np[k]
                mask = masks[t].numpy()
                
                try:
                    if hasattr(birefnet_model, 'refine'):
                        refined_mask = birefnet_model.refine(image, mask)
                        refined[t] = torch.from_numpy(refined_mask.astype(np.float32))
                except Exception as e:
                    print(f"[SAM3 Stabilizer] BiRefNet frame {t} failed: {e}")
            
            return refined
        except Exception as e:
            print(f"[SAM3 Stabilizer] BiRefNet refinement failed: {e}")
            return masks

    def _create_debug_visualization(self, original: torch.Tensor, stabilized: torch.Tensor, images_np: np.ndarray, keys: List[int]) -> torch.Tensor:
        """Create side-by-side visualization."""
        N, H, W = original.shape
        vis = torch.zeros(N, H, W * 2, 3, dtype=torch.float32)
        
        for t, k in enumerate(keys):
            if k < len(images_np):
                frame = images_np[k].astype(np.float32) / 255.0
                frame = cv2.resize(frame, (W, H))
            else:
                frame = np.zeros((H, W, 3), dtype=np.float32)
            
            orig_edges = self._get_edges(original[t].numpy())
            left = frame.copy()
            left[orig_edges > 0] = [1.0, 0.0, 0.0]
            
            stab_edges = self._get_edges(stabilized[t].numpy())
            right = frame.copy()
            right[stab_edges > 0] = [0.0, 1.0, 0.0]
            
            vis[t, :, :W, :] = torch.from_numpy(left)
            vis[t, :, W:, :] = torch.from_numpy(right)
        
        return vis
    
    def _create_simple_debug_visualization(self, original: torch.Tensor, stabilized: torch.Tensor, keys: List[int]) -> torch.Tensor:
        """Create simple visualization when no source frames available."""
        N, H, W = original.shape
        vis = torch.zeros(N, H, W * 2, 3, dtype=torch.float32)
        
        for t in range(N):
            vis[t, :, :W, 0] = original[t]
            vis[t, :, W:, 1] = stabilized[t]
        
        return vis
    
    def _get_edges(self, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Extract edges from binary mask."""
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        edges = cv2.Canny(mask_uint8, 50, 150)
        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            edges = cv2.dilate(edges, kernel)
        return edges > 0


class SAM3MaskTemporalMedian:
    """Simple temporal median filter for SAM3 masks."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS",),
                "video_state": ("SAM3_VIDEO_STATE",),
            },
            "optional": {
                "window_size": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2}),
                "boundary_only": ("BOOLEAN", {"default": True}),
                "boundary_pixels": ("INT", {"default": 10, "min": 3, "max": 30}),
            },
        }
    
    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("filtered_masks", "video_state")
    FUNCTION = "filter"
    CATEGORY = "SAM3"
    
    def filter(self, masks: Dict, video_state, window_size: int = 5, boundary_only: bool = True, boundary_pixels: int = 10):
        sorted_keys = sorted(masks.keys())
        N = len(sorted_keys)
        
        if N == 0:
            return (masks, video_state)
        
        first = _to_tensor(masks[sorted_keys[0]])
        if first.dim() > 2:
            first = first.squeeze()
        H, W = first.shape[-2:]
        
        stack = torch.zeros(N, H, W)
        for i, k in enumerate(sorted_keys):
            m = _to_tensor(masks[k])
            if m.dim() > 2:
                m = m.squeeze()
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W), mode="bilinear", align_corners=False).squeeze()
            stack[i] = m
        
        # Binarize input
        original_binary = (stack > 0.5).float()
        
        half = window_size // 2
        result = original_binary.clone()
        
        for t in range(N):
            t_start = max(0, t - half)
            t_end = min(N, t + half + 1)
            window = original_binary[t_start:t_end]
            
            median_mask = torch.median(window, dim=0).values
            
            if boundary_only:
                boundary = torch.from_numpy(_get_boundary_mask(original_binary[t].numpy(), boundary_pixels))
                result[t] = torch.where(boundary, median_mask, original_binary[t])
            else:
                result[t] = median_mask
        
        # Final binarization
        result = (result > 0.5).float()
        
        return ({k: result[i] for i, k in enumerate(sorted_keys)}, video_state)


NODE_CLASS_MAPPINGS = {
    "SAM3MaskTemporalStabilizer": SAM3MaskTemporalStabilizer,
    "SAM3MaskTemporalMedian": SAM3MaskTemporalMedian,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3MaskTemporalStabilizer": "🎭 SAM3 Mask Temporal Stabilizer",
    "SAM3MaskTemporalMedian": "🎭 SAM3 Mask Temporal Median",
}
