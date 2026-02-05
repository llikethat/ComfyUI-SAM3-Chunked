"""
SAM3 BBox Collector
===================

Collect bounding boxes for SAM3 segmentation prompts.
Compatible output type: SAM3_BOXES_PROMPT.
"""

import json
import torch
import numpy as np


class SAM3BBoxCollector:
    """Interactive bounding-box collection for SAM3 prompts.

    The widget stores JSON-encoded box arrays.  Each box is
    ``{"x1", "y1", "x2", "y2"}`` in pixel coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "positive_bboxes_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of positive bounding boxes [{x1,y1,x2,y2}, ...]"
                }),
                "negative_bboxes_json": ("STRING", {
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
    FUNCTION = "collect"
    CATEGORY = "SAM3"

    def collect(self, image, positive_bboxes_json="[]", negative_bboxes_json="[]", text_prompt=""):
        pos = self._parse(positive_bboxes_json)
        neg = self._parse(negative_bboxes_json)
        return (pos, neg)

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
