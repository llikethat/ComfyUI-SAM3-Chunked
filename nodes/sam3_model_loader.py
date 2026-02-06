"""
SAM3 Model Loader
=================

Load SAM3 model checkpoint with HuggingFace auto-download.
Outputs SAM3_MODEL — same type as the original ComfyUI-SAM3 package.

IMPORTANT: The facebook/sam3 model is gated. You must:
1. Accept the license at https://huggingface.co/facebook/sam3
2. Set HF_TOKEN environment variable or run `huggingface-cli login`
"""

import os
import sys
import torch
import folder_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAM3_HF_REPO = "facebook/sam3"
SAM3_CHECKPOINT = "sam3.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_sam3_importable():
    """Make the bundled ``lib/sam3`` importable if the system package is absent."""
    try:
        import sam3  # noqa: F401 — already installed
        return True
    except ImportError:
        pass

    lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model_dir():
    d = os.path.join(folder_paths.models_dir, "sam3")
    os.makedirs(d, exist_ok=True)
    return d


def _list_checkpoints():
    d = _get_model_dir()
    local = [f for f in os.listdir(d) if f.endswith((".pt", ".pth", ".safetensors"))]
    # Always include the HF checkpoint option
    all_options = sorted(set(local) | {SAM3_CHECKPOINT})
    return all_options or [SAM3_CHECKPOINT]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class LoadSAM3Model:
    """Load a SAM3 model checkpoint.

    Outputs ``SAM3_MODEL`` — a dictionary that carries the model, device, and
    dtype so that downstream nodes can use it.
    
    NOTE: The facebook/sam3 model is gated. You must accept the license at
    https://huggingface.co/facebook/sam3 and authenticate with HuggingFace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_checkpoints(), {
                    "tooltip": "Checkpoint file inside models/sam3/. Auto-downloads from HuggingFace if missing."
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for gated model access (or set HF_TOKEN env var)"
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
        
        # Set HF token if provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        print(f"[SAM3] Loading {model_name} → {device}  dtype={dtype}")

        if not _ensure_sam3_importable():
            raise ImportError("[SAM3] Cannot import sam3 library. Please check installation.")

        # Check if local file exists
        model_dir = _get_model_dir()
        local_path = os.path.join(model_dir, model_name)
        
        if os.path.isfile(local_path):
            checkpoint_path = local_path
            print(f"[SAM3] Using local checkpoint: {checkpoint_path}")
        else:
            # Let SAM3 library handle the download from HuggingFace
            checkpoint_path = None
            print(f"[SAM3] Checkpoint not found locally, will download from HuggingFace...")
            print(f"[SAM3] Make sure you have accepted the license at https://huggingface.co/facebook/sam3")

        try:
            from sam3.model_builder import build_sam3_video_model
            
            # build_sam3_video_model will auto-download if checkpoint_path is None
            model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=(checkpoint_path is None),
                device=str(device),
            )
            model = model.eval()
            
            # Move to specified dtype if different
            if hasattr(model, 'to'):
                model = model.to(dtype=dtype)
                
            print(f"[SAM3] Model loaded successfully!")
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "gated" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError(
                    f"[SAM3] HuggingFace authentication required!\n"
                    f"The facebook/sam3 model is gated. Please:\n"
                    f"1. Visit https://huggingface.co/facebook/sam3 and accept the license\n"
                    f"2. Either:\n"
                    f"   a) Set your HF token in the node's hf_token input, OR\n"
                    f"   b) Set HF_TOKEN environment variable, OR\n"
                    f"   c) Run 'huggingface-cli login' in terminal\n"
                    f"\nOriginal error: {e}"
                )
            else:
                raise RuntimeError(f"[SAM3] Failed to load model: {e}")

        return ({
            "model": model,
            "model_path": checkpoint_path or "huggingface:facebook/sam3",
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
