"""
SAM3 Model Loader
=================

Load SAM3 model checkpoint with HuggingFace auto-download.
Outputs SAM3_MODEL — same type as the original ComfyUI-SAM3 package.
"""

import os
import sys
import torch
import folder_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAM3_KNOWN_CHECKPOINTS = {
    "sam3.pt": {
        "hf_repo": "facebook/sam3",
        "hf_file": "sam3.pt",
    },
    "sam3_hiera_large.pt": {
        "hf_repo": "facebook/sam3",
        "hf_file": "sam3_hiera_large.pt",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_sam3_importable():
    """Make the bundled ``lib/sam3`` importable if the system package is absent."""
    try:
        import sam3  # noqa: F401 — already installed
        return
    except ImportError:
        pass

    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)


def _get_model_dir():
    d = os.path.join(folder_paths.models_dir, "sam3")
    os.makedirs(d, exist_ok=True)
    return d


def _list_checkpoints():
    d = _get_model_dir()
    local = [f for f in os.listdir(d) if f.endswith((".pt", ".pth", ".safetensors"))]
    return sorted(set(local) | set(SAM3_KNOWN_CHECKPOINTS.keys())) or ["sam3.pt"]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class LoadSAM3Model:
    """Load a SAM3 model checkpoint.

    Outputs ``SAM3_MODEL`` — a dictionary that carries the model, device, and
    dtype so that downstream nodes can use it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_checkpoints(), {
                    "tooltip": "Checkpoint file inside models/sam3/. Auto-downloads if missing."
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token (only for gated repos)"
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load"
    CATEGORY = "SAM3"

    # ── main entry ────────────────────────────────────────────
    def load(self, model_name, hf_token="", device="auto", precision="auto"):
        device = self._resolve_device(device)
        dtype = self._resolve_dtype(precision)
        model_path = self._resolve_path(model_name, hf_token)

        print(f"[SAM3] Loading {model_name} → {device}  dtype={dtype}")

        _ensure_sam3_importable()

        try:
            from sam3.model_builder import build_sam3_video_predictor
            model = build_sam3_video_predictor(model_path)
            model = model.to(device=device, dtype=dtype).eval()
        except Exception as e:
            print(f"[SAM3] build_sam3_video_predictor failed ({e}), loading raw checkpoint")
            model = {"_raw_checkpoint": torch.load(model_path, map_location="cpu"),
                     "_path": model_path}

        return ({
            "model": model,
            "model_path": model_path,
            "device": str(device),
            "dtype": dtype,
        },)

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _resolve_device(dev):
        if dev == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return dev

    @staticmethod
    def _resolve_dtype(prec):
        if prec == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[prec]

    @staticmethod
    def _resolve_path(model_name, hf_token):
        d = _get_model_dir()
        path = os.path.join(d, model_name)
        if os.path.isfile(path):
            return path

        # Try HuggingFace
        info = SAM3_KNOWN_CHECKPOINTS.get(model_name)
        if info is None:
            raise FileNotFoundError(f"Model not found: {path}")

        try:
            from huggingface_hub import hf_hub_download
            print(f"[SAM3] Downloading {info['hf_repo']}/{info['hf_file']} …")
            return hf_hub_download(
                repo_id=info["hf_repo"],
                filename=info["hf_file"],
                local_dir=d,
                token=hf_token or None,
            )
        except ImportError:
            pass

        # Direct URL fallback
        url = f"https://huggingface.co/{info['hf_repo']}/resolve/main/{info['hf_file']}"
        print(f"[SAM3] Direct download: {url}")
        import urllib.request
        urllib.request.urlretrieve(url, path)
        return path
