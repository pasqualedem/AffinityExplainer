"""
cache.py — hierarchical caching system for model outputs and attribution maps.

Key features:
- Explicit parameters (no model/algo objects)
- Safe under concurrent SLURM jobs (uses file locks)
- Caches both model outputs and attributions
- Automatically invalidates attributions when model output changes
- Directory layout: dataset → model → algo → mask_logic → version
"""

import os
import json
import time
import hashlib
import torch
import shutil
import logging
from datetime import datetime
from contextlib import contextmanager

try:
    from filelock import FileLock, Timeout
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("cache")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Utility: hashing
# -----------------------------------------------------------------------------
def _hash_dict(data):
    """Deterministic JSON hash of any nested structure."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def tensor_hash(tensor):
    if isinstance(tensor, torch.Tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
    return None


# -----------------------------------------------------------------------------
# Locking
# -----------------------------------------------------------------------------
@contextmanager
def acquire_lock(lock_path, timeout=120):
    """File or directory-based lock for concurrency safety."""
    if _HAS_FILELOCK:
        lock = FileLock(lock_path)
        try:
            with lock.acquire(timeout=timeout):
                yield
        except Timeout:
            logger.warning(f"Timeout waiting for lock {lock_path}, continuing read-only.")
            yield
    else:
        start = time.time()
        while True:
            try:
                os.mkdir(lock_path)
                break
            except FileExistsError:
                if time.time() - start > timeout:
                    logger.warning(f"Timeout waiting for lock {lock_path}")
                    break
                time.sleep(2)
        try:
            yield
        finally:
            if os.path.isdir(lock_path):
                shutil.rmtree(lock_path)


# -----------------------------------------------------------------------------
# Core atomic load/compute
# -----------------------------------------------------------------------------
def compute_or_load(path, compute_fn, meta_path=None, timeout=120):
    """Load from cache or compute atomically (safe across jobs)."""
    lock_path = path + ".lock"
    tmp_path = path + ".tmp"

    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    with acquire_lock(lock_path, timeout):
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)

        result = compute_fn()
        torch.save(result, tmp_path)

        if meta_path:
            meta = {
                "timestamp": datetime.now().isoformat(),
                "tensor_hash": tensor_hash(result),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        os.replace(tmp_path, path)
        return result


# -----------------------------------------------------------------------------
# Model output caching
# -----------------------------------------------------------------------------
def get_cached_model_output(
    dataset_id,
    model_name,
    image_ids,
    classes,
    cache_dir,
    compute_fn,
    timeout=120,
):
    """
    Cache the raw model output (e.g., logits).

    Args:
        dataset_id: str, e.g. 'val_pascal_N1K5'
        model_name: str
        image_ids: list[str]
        classes: list[str]
        cache_dir: str
        compute_fn: callable returning the model output tensor
    Returns:
        (output_tensor, output_hash)
    """
    output_hash = _hash_dict({
        "dataset": dataset_id,
        "model": model_name,
        "images": image_ids,
        "classes": classes,
    })

    subdir = os.path.join(cache_dir, f"dataset_{dataset_id}", f"model_{model_name}", "outputs")
    os.makedirs(subdir, exist_ok=True)

    path = os.path.join(subdir, f"{output_hash}.pt")
    meta_path = path + ".meta.json"

    result = compute_or_load(path, compute_fn, meta_path, timeout=timeout)
    return result, output_hash


# -----------------------------------------------------------------------------
# Attribution caching
# -----------------------------------------------------------------------------
def get_cached_attribution(
    dataset_id,
    model_name,
    algo_name,
    algo_params,
    mask_logic,
    output_hash,
    cache_dir,
    compute_fn,
    version="v1",
    timeout=120,
):
    """
    Cache attribution maps that depend on a model output.

    Args:
        dataset_id: str
        model_name: str
        algo_name: str
        algo_params: dict
        mask_logic: str ('logits' or 'ground_truth')
        output_hash: str (from get_cached_model_output)
        compute_fn: callable returning attribution tensor
    """
    key = _hash_dict({
        "dataset": dataset_id,
        "model": model_name,
        "algo": algo_name,
        "params": algo_params,
        "mask_logic": mask_logic,
        "output_hash": output_hash,
        "version": version,
    })

    subdir = os.path.join(
        cache_dir,
        f"dataset_{dataset_id}",
        f"model_{model_name}",
        f"algo_{algo_name}",
        f"mask_{mask_logic}",
        f"version_{version}",
    )
    os.makedirs(subdir, exist_ok=True)

    path = os.path.join(subdir, f"{key}.pt")
    meta_path = path + ".meta.json"

    return compute_or_load(path, compute_fn, meta_path, timeout=timeout)


# -----------------------------------------------------------------------------
# Clear cache
# -----------------------------------------------------------------------------
def clear_cache(cache_dir, dataset_id=None, model_name=None, algo_name=None,
                mask_logic=None, version=None, older_than_days=None):
    removed = 0
    for root, _, files in os.walk(cache_dir, topdown=False):
        if not any(f.endswith(".pt") for f in files):
            continue

        if dataset_id and f"dataset_{dataset_id}" not in root:
            continue
        if model_name and f"model_{model_name}" not in root:
            continue
        if algo_name and f"algo_{algo_name}" not in root:
            continue
        if mask_logic and f"mask_{mask_logic}" not in root:
            continue
        if version and f"version_{version}" not in root:
            continue

        for f in files:
            if not f.endswith(".pt"):
                continue
            full = os.path.join(root, f)
            if older_than_days:
                age_days = (time.time() - os.path.getmtime(full)) / 86400
                if age_days < older_than_days:
                    continue
            os.remove(full)
            meta = full + ".meta.json"
            if os.path.exists(meta):
                os.remove(meta)
            removed += 1

    logger.info(
        f"Removed {removed} cache files "
        f"(dataset={dataset_id}, model={model_name}, algo={algo_name}, mask={mask_logic}, version={version})"
    )
    return removed


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    def dummy_model_forward():
        time.sleep(1)
        return torch.rand(1, 3, 224, 224)

    def dummy_attribution():
        time.sleep(1)
        return torch.rand(1, 3, 224, 224)

    dataset_id = "val_pascal_N1K5"
    model_name = "resnet50"
    algo_name = "gradcam"
    cache_dir = "/tmp/attr_cache"

    image_ids = ["img001"]
    classes = ["cat"]

    # 1️⃣ Cache model output
    output, out_hash = get_cached_model_output(
        dataset_id, model_name, image_ids, classes, cache_dir, compute_fn=dummy_model_forward
    )

    # 2️⃣ Cache attribution (depends on output hash)
    attribution = get_cached_attribution(
        dataset_id, model_name, algo_name, {"steps": 10},
        mask_logic="logits", output_hash=out_hash,
        cache_dir=cache_dir, compute_fn=dummy_attribution
    )

    print("Attribution shape:", attribution.shape)
