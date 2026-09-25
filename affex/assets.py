r"""Lazily fetched assets: datasets and model weights.

Nothing in this repository has to be installed by hand. Every model builder and every
dataset asks for what it needs through `ensure()`, which downloads and unpacks the asset
the first time it is missing and is a no-op afterwards. Running an experiment on COCO
therefore downloads COCO, and building DCAMA downloads DCAMA's weights.

Destinations default to `data/` and `checkpoints/` inside the repository; set
AFFEX_DATA_DIR or AFFEX_CHECKPOINTS to point at a shared copy on a cluster. Set
AFFEX_NO_DOWNLOAD=1 to turn a missing asset into an error instead of a download, which
is what you want on a compute node with no outbound network.
"""

import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional


def data_dir() -> str:
    return os.environ.get("AFFEX_DATA_DIR", "data")


def checkpoints_dir() -> str:
    return os.environ.get("AFFEX_CHECKPOINTS", "checkpoints")


def resolve(path: str) -> str:
    """Map a configured path onto the directories this installation actually uses.

    Parameter files say `data/coco/...` and builders say `checkpoints/...`; both are
    rewritten to AFFEX_DATA_DIR and AFFEX_CHECKPOINTS when those are set, so a shared
    read-only copy on a cluster needs no edit to any config. Absolute paths pass through.
    """
    if not path or os.path.isabs(path):
        return path
    parts = path.replace("\\", "/").split("/")
    if parts[0] == "data":
        return os.path.join(data_dir(), *parts[1:])
    if parts[0] == "checkpoints":
        return os.path.join(checkpoints_dir(), *parts[1:])
    return path


@dataclass
class Asset:
    """One downloadable thing.

    `probe` is the path that proves the asset is present; `dest` is where the payload
    lands. `url` is fetched directly, `gdrive` goes through gdown (Google Drive, which
    is where the DCAMA and DMTNet authors published their weights), and `manual` marks
    an asset behind a licence click that no script may fetch on the user's behalf.
    """

    probe: Callable[[], str]
    dest: Callable[[], str]
    url: Optional[str] = None                 # one URL, or several to fetch in turn
    gdrive: Optional[str] = None
    archive: Optional[str] = None          # "zip" | "tar"
    strip_dirs: tuple = ()                 # directories to lift into dest after unpacking
    manual: Optional[str] = None           # instructions, for licence-gated assets
    post: Optional[Callable[[str], None]] = None   # runs on dest once the payload is in place
    note: str = ""


ASSETS = {
    "coco-annotations": Asset(
        probe=lambda: os.path.join(data_dir(), "coco/annotations/instances_val2014.json"),
        dest=lambda: os.path.join(data_dir(), "coco"),
        url="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        archive="zip",
        post=lambda dest: _coco_match_2017_filenames(dest),
        note="the demo streams COCO images per episode, so only the annotations are stored",
    ),
    "coco-images": Asset(
        probe=lambda: os.path.join(data_dir(), "coco/train_val_2017"),
        dest=lambda: os.path.join(data_dir(), "coco"),
        url=("http://images.cocodataset.org/zips/train2017.zip",
             "http://images.cocodataset.org/zips/val2017.zip"),
        archive="zip",
        post=lambda dest: _coco_merge_2017_splits(dest),
        note="about 20 GB, needed only to evaluate on COCO from local files",
    ),
    "pascal-voc": Asset(
        probe=lambda: os.path.join(data_dir(), "pascal/JPEGImages"),
        dest=lambda: os.path.join(data_dir(), "pascal"),
        url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        archive="tar",
        strip_dirs=("VOCdevkit/VOC2012",),
        note="about 2 GB",
    ),
    "dcama-backbone": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "dcama/swin_base_patch4_window12_384.pth"),
        dest=lambda: os.path.join(checkpoints_dir(), "dcama"),
        gdrive="1NlX0IFgcrjdHmbTBrXrqAdzYJ7Ul9ux6",
    ),
    "dmtnet": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "dmtnet.pt"),
        dest=lambda: checkpoints_dir(),
        gdrive="12oY79kWTLKoIoSkHn8HfM0paw-PClIVV",
    ),
    "dcama-pascal-fold0": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "dcama/pascal/swin_fold0.pt"),
        dest=lambda: os.path.join(checkpoints_dir(), "dcama/pascal"),
        gdrive="1cEXdqRAjorPx30xnB0hwcNBwYN8JYcKx",
        note="the released fold-0 weights; other folds are linked from the DCAMA repo",
    ),
    "dinov2-vitl14": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "dinov2_vitl14_pretrain.pth"),
        dest=lambda: checkpoints_dir(),
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    ),
    "sam-vit-b": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "sam_vit_b_01ec64.pth"),
        dest=lambda: checkpoints_dir(),
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "sam-vit-h": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "sam_vit_h_4b8939.pth"),
        dest=lambda: checkpoints_dir(),
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    "sam2-hiera-large": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "sam2_hiera_large.pt"),
        dest=lambda: checkpoints_dir(),
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    ),
    "dinov3-vitb16": Asset(
        probe=lambda: os.path.join(checkpoints_dir(),
                                   "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        dest=lambda: checkpoints_dir(),
        manual=("DINOv3 weights are licence-gated. Accept the licence at "
                "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and place "
                "the .pth in {dest}, or export HF_TOKEN and rerun."),
    ),
    "sansa-adapter": Asset(
        probe=lambda: os.path.join(checkpoints_dir(), "sansa_universal.pth"),
        dest=lambda: checkpoints_dir(),
        manual=("The SANSA adapter is published on the authors' Hugging Face page and "
                "requires accepting their terms. Download sansa_universal.pth into {dest}."),
    ),
}


class AssetMissing(RuntimeError):
    pass


def _coco_match_2017_filenames(dest: str) -> None:
    """Point the 2014 annotations at the 2017 image files.

    The episodes are defined on the 2014 annotations but the images are the 2017 release,
    whose files drop the `COCO_val2014_` prefix. Rewriting the names once here is what
    lets `img_dir` find them. Splitting an already-rewritten name changes nothing, so
    this is safe to run again.
    """
    import json

    for split in ("train2014", "val2014"):
        path = os.path.join(dest, "annotations", f"instances_{split}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            anns = json.load(f)
        for image in anns["images"]:
            image["file_name"] = image["file_name"].split("_")[-1]
        with open(path, "w") as f:
            json.dump(anns, f)


def _coco_merge_2017_splits(dest: str) -> None:
    """Put train2017 and val2017 images side by side, the layout the episodes assume."""
    merged = os.path.join(dest, "train_val_2017")
    os.makedirs(merged, exist_ok=True)
    for split in ("train2017", "val2017"):
        folder = os.path.join(dest, split)
        if not os.path.isdir(folder):
            continue
        for item in os.listdir(folder):
            shutil.move(os.path.join(folder, item), os.path.join(merged, item))
        os.rmdir(folder)


def _download(url: str, out: str) -> None:
    import requests

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {os.path.basename(out)}: {pct:5.1f}% "
                          f"({done >> 20}/{total >> 20} MB)", end="", file=sys.stderr)
        print("", file=sys.stderr)


def _unpack(path: str, dest: str, kind: str, strip_dirs=()) -> None:
    os.makedirs(dest, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    else:
        with tarfile.open(path) as t:
            t.extractall(dest)
    for sub in strip_dirs:
        inner = os.path.join(dest, sub)
        if os.path.isdir(inner):
            for item in os.listdir(inner):
                shutil.move(os.path.join(inner, item), os.path.join(dest, item))
            shutil.rmtree(os.path.join(dest, sub.split("/")[0]), ignore_errors=True)


def ensure(name: str) -> str:
    """Return the path to asset `name`, downloading it first if it is not there."""
    asset = ASSETS[name]
    probe = asset.probe()
    if os.path.exists(probe):
        return probe
    if asset.manual:
        raise AssetMissing(f"{name}: " + asset.manual.format(dest=asset.dest()))
    if os.environ.get("AFFEX_NO_DOWNLOAD"):
        raise AssetMissing(
            f"{name} is missing at {probe} and AFFEX_NO_DOWNLOAD is set. "
            f"Fetch it on a machine with network access and copy it over.")

    dest = asset.dest()
    os.makedirs(dest, exist_ok=True)
    print(f"[affex] fetching {name}"
          + (f" ({asset.note})" if asset.note else ""), file=sys.stderr)
    if asset.gdrive:
        import gdown
        gdown.download(id=asset.gdrive, output=os.path.join(dest, ""), quiet=False)
    else:
        urls = asset.url if isinstance(asset.url, tuple) else (asset.url,)
        for url in urls:
            tmp = os.path.join(dest, os.path.basename(url))
            _download(url, tmp)
            if asset.archive:
                _unpack(tmp, dest, asset.archive, asset.strip_dirs)
                os.remove(tmp)
    if asset.post:
        asset.post(dest)
    if not os.path.exists(probe):
        raise AssetMissing(f"{name}: download finished but {probe} is still missing.")
    return probe


def available(name: str) -> bool:
    return os.path.exists(ASSETS[name].probe())
