"""
SAM3 Mask Stabilizer - Post-processing for temporal consistency

Stabilizes masks across frames to reduce flickering and improve temporal coherence.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class SAM3MaskStabilizer:
    """
    Post-process SAM3 propagation masks for temporal stability.
    
    Reduces flickering by:
    - Temporal smoothing across frames
    - Morphological operations to clean up edges
    - Optional mask interpolation for missing frames
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_MASKS",),
                "video_state": ("SAM3_VIDEO_STATE",),
            },
            "optional": {
                "temporal_smooth": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Temporal smoothing weight (0=no smoothing, 1=max smoothing)"
                }),
                "morph_iterations": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "tooltip": "Morphological cleanup iterations (0=none)"
                }),
                "fill_missing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Interpolate masks for frames with no detection"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Binarization threshold after smoothing"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "stabilize"
    CATEGORY = "SAM3/video"

    def stabilize(
        self,
        masks: Dict,
        video_state: Dict,
        temporal_smooth: float = 0.3,
        morph_iterations: int = 1,
        fill_missing: bool = True,
        threshold: float = 0.5,
    ):
        """Stabilize masks across video frames."""
        
        orig_h = video_state.get("orig_height", 720)
        orig_w = video_state.get("orig_width", 1280)
        num_frames = video_state.get("num_frames", len(masks))
        
        print(f"[SAM3 Stabilizer] Processing {num_frames} frames ({orig_w}x{orig_h})")
        
        # Get sorted frame indices
        if isinstance(masks, dict):
            sorted_keys = sorted(masks.keys())
        else:
            # Already a tensor
            return (masks,)
        
        if len(sorted_keys) == 0:
            print("[SAM3 Stabilizer] No masks to process")
            return (torch.zeros(num_frames, orig_h, orig_w, dtype=torch.float32),)
        
        # Convert dict to tensor, handling various mask shapes
        mask_tensor = self._dict_to_tensor(masks, sorted_keys, orig_h, orig_w)
        
        # Apply temporal smoothing
        if temporal_smooth > 0 and mask_tensor.shape[0] > 1:
            mask_tensor = self._temporal_smooth(mask_tensor, temporal_smooth)
        
        # Apply morphological cleanup
        if morph_iterations > 0:
            mask_tensor = self._morph_cleanup(mask_tensor, morph_iterations)
        
        # Fill missing frames
        if fill_missing:
            mask_tensor = self._fill_missing_frames(mask_tensor, num_frames)
        
        # Binarize
        mask_tensor = (mask_tensor > threshold).float()
        
        print(f"[SAM3 Stabilizer] Output shape: {mask_tensor.shape}")
        
        return (mask_tensor,)

    def _dict_to_tensor(
        self,
        masks: Dict,
        sorted_keys: List,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Convert mask dictionary to tensor, handling various input shapes."""
        
        out = torch.zeros(len(sorted_keys), orig_h, orig_w, dtype=torch.float32)
        
        for i, key in enumerate(sorted_keys):
            m = masks[key]
            
            # Skip None masks
            if m is None:
                continue
            
            # Convert numpy to tensor
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m.astype(np.float32))
            
            # Handle empty masks (shape [0, H, W] - no objects detected)
            if m.numel() == 0:
                # Leave as zeros
                continue
            
            # Check for empty first dimension (no objects)
            if m.dim() >= 1 and m.shape[0] == 0:
                # No objects detected for this frame, leave as zeros
                continue
            
            # Ensure float type
            if m.dtype == torch.bool:
                m = m.float()
            elif m.dtype != torch.float32:
                m = m.to(torch.float32)
            
            # Handle different dimensionalities
            # Expected: [H, W] or [1, H, W] or [N, H, W] or [1, 1, H, W] etc.
            while m.dim() > 2:
                if m.shape[0] == 1:
                    # Single object/batch - squeeze
                    m = m.squeeze(0)
                elif m.shape[0] == 0:
                    # Empty - no objects, use zeros
                    m = torch.zeros(orig_h, orig_w, dtype=torch.float32)
                    break
                else:
                    # Multiple objects - combine them (union)
                    m = m.any(dim=0).float() if m.max() <= 1 else m.max(dim=0)[0]
            
            # Handle 1D tensor (shouldn't happen but be safe)
            if m.dim() == 1:
                continue
            
            # Resize if dimensions don't match
            if m.shape[-2:] != (orig_h, orig_w):
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            out[i] = m
        
        return out

    def _temporal_smooth(
        self,
        masks: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Apply temporal smoothing across frames."""
        
        if masks.shape[0] < 2:
            return masks
        
        smoothed = masks.clone()
        
        for i in range(1, masks.shape[0]):
            smoothed[i] = (1 - weight) * masks[i] + weight * smoothed[i - 1]
        
        # Backward pass for bidirectional smoothing
        for i in range(masks.shape[0] - 2, -1, -1):
            smoothed[i] = (1 - weight * 0.5) * smoothed[i] + (weight * 0.5) * smoothed[i + 1]
        
        return smoothed

    def _morph_cleanup(
        self,
        masks: torch.Tensor,
        iterations: int,
    ) -> torch.Tensor:
        """Apply morphological operations to clean up mask edges."""
        
        if iterations <= 0:
            return masks
        
        # Create a small kernel for morphological ops
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=masks.dtype, device=masks.device)
        kernel = kernel / kernel.numel()
        
        # Add batch and channel dims for conv2d
        m = masks.unsqueeze(1)  # [N, 1, H, W]
        
        for _ in range(iterations):
            # Erosion-like (min pooling approximation via threshold after blur)
            m_eroded = F.conv2d(m, kernel, padding=kernel_size // 2)
            m_eroded = (m_eroded > 0.7).float()
            
            # Dilation-like
            m_dilated = F.conv2d(m_eroded, kernel, padding=kernel_size // 2)
            m_dilated = (m_dilated > 0.3).float()
            
            m = m_dilated
        
        return m.squeeze(1)

    def _fill_missing_frames(
        self,
        masks: torch.Tensor,
        total_frames: int,
    ) -> torch.Tensor:
        """Fill in frames where mask is all zeros by interpolating from neighbors."""
        
        # Find frames with valid masks
        valid_frames = []
        for i in range(masks.shape[0]):
            if masks[i].sum() > 0:
                valid_frames.append(i)
        
        if len(valid_frames) == 0:
            return masks
        
        # Fill missing frames
        for i in range(masks.shape[0]):
            if masks[i].sum() == 0:
                # Find nearest valid frames
                prev_valid = None
                next_valid = None
                
                for v in valid_frames:
                    if v < i:
                        prev_valid = v
                    elif v > i and next_valid is None:
                        next_valid = v
                        break
                
                if prev_valid is not None and next_valid is not None:
                    # Interpolate
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    masks[i] = (1 - alpha) * masks[prev_valid] + alpha * masks[next_valid]
                elif prev_valid is not None:
                    masks[i] = masks[prev_valid]
                elif next_valid is not None:
                    masks[i] = masks[next_valid]
        
        return masks


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3MaskStabilizer": SAM3MaskStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3MaskStabilizer": "SAM3 Mask Stabilizer",
}
