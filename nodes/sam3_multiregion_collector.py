"""
SAM3 Multi-Region Collector
============================

Collect multiple prompt regions (each with its own points + boxes) for
SAM3 segmentation. Supports up to 8 independent prompt regions with
colour-coded tabs in the frontend widget.

Compatible output type: SAM3_MULTI_PROMPT.

Includes interactive canvas widget support (web/sam3_multiregion_widget.js).
"""

import io
import json
import base64
import torch
import numpy as np


class SAM3MultiRegionCollector:
    """Multi-region prompt collector for SAM3.

    Each prompt region can contain positive/negative points and
    positive/negative boxes. The frontend widget shows tabs for
    switching between regions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Hidden storage widget — the JS frontend serializes the
                # full prompt state here as JSON.
                "multi_prompts_store": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON store for all prompt regions"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_MULTI_PROMPT",)
    RETURN_NAMES = ("multi_prompts",)
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "SAM3"

    def collect(self, image, multi_prompts_store="[]"):
        prompts = self._parse(multi_prompts_store)

        # Send the first frame as a base64 image to the frontend widget
        bg_image = self._encode_image(image)

        return {
            "ui": {"bg_image": [bg_image]},
            "result": (prompts,),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(json_str):
        """Parse JSON into a list of prompt region dicts.

        Each region dict contains:
            positive_points: Tensor (N, 2)
            negative_points: Tensor (N, 2)
            positive_boxes:  Tensor (N, 4)
            negative_boxes:  Tensor (N, 4)
        """
        try:
            regions_raw = json.loads(json_str)
        except json.JSONDecodeError:
            regions_raw = []

        if not isinstance(regions_raw, list):
            regions_raw = []

        result = []
        for region in regions_raw:
            if not isinstance(region, dict):
                continue

            parsed = {}
            for key in ("positive_points", "negative_points"):
                pts = region.get(key, [])
                if pts:
                    coords = []
                    for p in pts:
                        if isinstance(p, dict):
                            coords.append([p.get("x", 0), p.get("y", 0)])
                        elif isinstance(p, (list, tuple)):
                            coords.append(list(p[:2]))
                    parsed[key] = torch.tensor(coords, dtype=torch.float32) if coords else torch.zeros(0, 2)
                else:
                    parsed[key] = torch.zeros(0, 2)

            for key in ("positive_boxes", "negative_boxes"):
                boxes = region.get(key, [])
                if boxes:
                    box_coords = []
                    for b in boxes:
                        if isinstance(b, dict):
                            box_coords.append([
                                b.get("x1", 0), b.get("y1", 0),
                                b.get("x2", 0), b.get("y2", 0),
                            ])
                        elif isinstance(b, (list, tuple)):
                            box_coords.append(list(b[:4]))
                    parsed[key] = torch.tensor(box_coords, dtype=torch.float32) if box_coords else torch.zeros(0, 4)
                else:
                    parsed[key] = torch.zeros(0, 4)

            result.append(parsed)

        # Always return at least one empty region
        if not result:
            result.append({
                "positive_points": torch.zeros(0, 2),
                "negative_points": torch.zeros(0, 2),
                "positive_boxes": torch.zeros(0, 4),
                "negative_boxes": torch.zeros(0, 4),
            })

        return result

    @staticmethod
    def _encode_image(image):
        """Encode the first frame of an IMAGE batch as base64 JPEG."""
        try:
            from PIL import Image as PILImage

            if torch.is_tensor(image):
                arr = image[0].cpu().numpy()
            else:
                arr = np.asarray(image[0])

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""
