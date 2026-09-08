import json
import hashlib
import platform
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict):
    """Append a single record as a JSON line (creates file/dirs as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    """Load all records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, step: int, config: dict):
    """Save model + optimizer state with manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step":      step,
            "config":    config,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    # M2 completion (v3.5): committed checkpoints are first-party artifacts
    # (plain dict/str/int/float/tensors — verified) so the restrictive
    # unpickler is safe and preferred.
    return torch.load(path, map_location=device, weights_only=True)


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def hash_dict(d: dict) -> str:
    """Deterministic SHA-256 hash of a JSON-serialisable dict."""
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def hash_file(path: Path) -> str:
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_id(config_dict: dict, seed: int) -> str:
    """Create a short run ID from config hash + seed."""
    cfg_hash = hash_dict(config_dict)
    return f"{cfg_hash}_s{seed}"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def capture_environment() -> Dict[str, Any]:
    """Capture runtime environment metadata for reproducibility."""
    env: Dict[str, Any] = {
        "python":     sys.version,
        "platform":   platform.platform(),
        "torch":      torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device_count"] = torch.cuda.device_count()
        env["cuda_device_name"]  = torch.cuda.get_device_name(0)
        env["cuda_capability"]   = list(torch.cuda.get_device_capability(0))
    return env


def write_manifest(out_dir: Path, config_dict: dict, run_id: str):
    """Write manifest.json containing config + environment snapshot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id":      run_id,
        "config":      config_dict,
        "environment": capture_environment(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
