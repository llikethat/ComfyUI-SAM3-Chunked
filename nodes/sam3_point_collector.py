"""
SAM3 Point Collector
====================

Collect positive/negative point prompts for SAM3 segmentation.
Compatible output type: SAM3_POINTS_PROMPT.

Includes interactive canvas widget support (web/sam3_points_widget.js).
"""

import io
import json
import base64
import torch
import numpy as np


class SAM3PointCollector:
    """Interactive point collection for SAM3 prompts.

    The frontend widget (sam3_points_widget.js) stores JSON-encoded
    point arrays. Each point is ``{"x", "y"}`` in pixel coordinates.
    Left-click = positive, Shift/Right-click = negative.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Widget names MUST match what sam3_points_widget.js expects
                "coordinates": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of positive points [{x,y}, ...]"
                }),
                "neg_coordinates": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of negative points [{x,y}, ...]"
                }),
                "points_store": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "Combined store for serialization"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points")
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "SAM3"

    def collect(self, image, coordinates="[]", neg_coordinates="[]", points_store="{}"):
        pos = self._parse_points(coordinates)
        neg = self._parse_points(neg_coordinates)

        # Send the first frame as a base64 image to the frontend widget
        bg_image = self._encode_image(image)

        return {
            "ui": {"bg_image": [bg_image]},
            "result": (pos, neg),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_points(json_str):
        """Parse JSON into a dict compatible with SAM3_POINTS_PROMPT."""
        try:
            pts_raw = json.loads(json_str)
        except json.JSONDecodeError:
            pts_raw = []

        if not pts_raw:
            return {
                "points": torch.zeros(0, 2),
                "labels": torch.zeros(0, dtype=torch.long),
            }

        coords = []
        for p in pts_raw:
            if isinstance(p, dict):
                coords.append([p["x"], p["y"]])
            elif isinstance(p, (list, tuple)):
                coords.append(list(p[:2]))
        if not coords:
            return {
                "points": torch.zeros(0, 2),
                "labels": torch.zeros(0, dtype=torch.long),
            }

        return {
            "points": torch.tensor(coords, dtype=torch.float32),
            "labels": torch.ones(len(coords), dtype=torch.long),
        }

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
