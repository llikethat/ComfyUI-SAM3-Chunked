"""
SAM3 BBox Collector
===================

Collect bounding boxes for SAM3 segmentation prompts.
Compatible output type: SAM3_BOXES_PROMPT.

Includes interactive canvas widget support (web/sam3_bbox_widget.js).
"""

import io
import json
import base64
import torch
import numpy as np


class SAM3BBoxCollector:
    """Interactive bounding-box collection for SAM3 prompts.

    The frontend widget (sam3_bbox_widget.js) stores JSON-encoded box
    arrays. Each box is ``{"x1", "y1", "x2", "y2"}`` in pixel coords.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Widget names MUST match what sam3_bbox_widget.js expects
                "bboxes": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of positive bounding boxes [{x1,y1,x2,y2}, ...]"
                }),
                "neg_bboxes": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of negative bounding boxes [{x1,y1,x2,y2}, ...]"
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Optional text prompt (e.g. 'person')"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("positive_bboxes", "negative_bboxes")
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "SAM3"

    def collect(self, image, bboxes="[]", neg_bboxes="[]", text_prompt=""):
        pos = self._parse(bboxes)
        neg = self._parse(neg_bboxes)

        # Send the first frame as a base64 image to the frontend widget
        bg_image = self._encode_image(image)

        return {
            "ui": {"bg_image": [bg_image]},
            "result": (pos, neg),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(json_str):
        """Parse JSON into a dict compatible with SAM3_BOXES_PROMPT."""
        try:
            boxes_raw = json.loads(json_str)
        except json.JSONDecodeError:
            boxes_raw = []

        if not boxes_raw:
            return {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}

        boxes = []
        for b in boxes_raw:
            if isinstance(b, dict):
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            elif isinstance(b, (list, tuple)):
                boxes.append(list(b[:4]))
        if not boxes:
            return {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.long),
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
