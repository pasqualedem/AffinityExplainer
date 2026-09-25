# app.py
# Streamlit app to replicate the notebook: load model & dataset, pick a sample, run prediction and explanation,
# and visualize with lovely-tensors.

import copy
from pathlib import Path
import os
import traceback
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import torch
import torch.nn.functional as F
import lovely_tensors as lt
from PIL import Image
from io import BytesIO, StringIO
import zipfile

# streamlit-drawable-canvas still imports image_to_url from where it lived before
# Streamlit 1.40; point the old name at the current one before importing it.
import streamlit.elements.image as _st_image
from streamlit.elements.lib.image_utils import image_to_url as _image_to_url

if not hasattr(_st_image, "image_to_url"):
    _st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

from affex.explainer import EXPLAINER_REGISTRY

# --- Project Imports ---
try:
    from affex.explainer.affinity import MODEL_EXPLAINER_REGISTRY
    from affex.metrics import FSSCausalMetric
    from affex.data.utils import BatchKeys, min_max_scale
    from affex.utils.torch import to_device
    from affex.utils.utils import ResultDict
    from affex.utils.grid import create_experiment, load_yaml
    from affex.models import build_model_preconfigured
    from affex.data import get_dataloaders
    from affex.explainer import build_explainer
    from affex.substitution import Substitutor
    from affex.utils.segmentation import unnormalize
    
    # Optional saliency
    try:
        import saliency.core as saliency 
    except Exception:
        saliency = None
        
    _imports_error = None
except Exception as e:
    _imports_error = e

lt.monkey_patch()

# --- Configuration Dicts (Kept Intact) ---
config_coco = {
    "datasets": {
        "val_coco20i_N1K1": {
            "name": "coco",
            "instances_path": "data/coco/annotations/instances_val2014.json",
            "all_example_categories": False,
            "img_dir": "data/coco/train_val_2017" if os.path.exists("data/coco/train_val_2017") else None,
            "n_shots": 1,
            "n_ways": 1,
            "do_subsample": False,
            "add_box_noise": False,
            "val_fold_idx": None,
            "n_folds": None,
            "split": "val",
        }
    },
    "common": {
        "remove_small_annotations": True,
        "custom_preprocess": False,
        "maintain_gt_shape": False,
    },
}

config_pascal = {
    "datasets": {
        "val_pascal5i_N1K1": {
            "name": "pascal",
            "data_dir": "data/pascal",
            "split": "val",
            "val_fold_idx": None,
            "n_folds": 4,
            "n_shots": 1,
            "n_ways": 1,
            "do_subsample": False,
            "val_num_samples": 1000,
            "maintain_gt_shape": False,
        }
    },
    "common": {
        "remove_small_annotations": True,
        "ignore_borders": True,
        "custom_preprocess": False,
    }
}

config_dataloader = {
    "num_workers": 0,
    "batch_size": 1,
    "csv_folder": "data_csv"
}

# --- Visual Styling Helpers ---
def local_css():
    st.markdown("""
    <style>
        .main-header {font-size: 2.5rem; font-weight: 700; margin-bottom: 0rem;}
        .sub-header {font-size: 1.5rem; font-weight: 600; margin-top: 1rem;}
        /* Use Streamlit theme variables for colors to support Dark Mode */
        .highlight-box {
            background-color: var(--secondary-background-color);
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid var(--primary-color);
            color: var(--text-color);
        }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Helpers (Kept Intact) ---
def error_box(msg: str, exc: Exception | None = None):
    st.error(msg)
    if exc is not None:
        # No expander here: error_box can be called from inside one, and Streamlit
        # forbids nested expanders (which would mask the real error).
        st.code("".join(traceback.format_exception(None, exc, exc.__traceback__)))

def tensor_to_pil(tensor):
    tensor = (tensor * 255).permute(1, 2, 0).detach().type(torch.uint8).cpu()
    return Image.fromarray(tensor.numpy())

def tensor_to_heatmap(tensor):
    array = tensor.detach().cpu().numpy()
    cmap = plt.get_cmap("jet")
    rgba = cmap(array)
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(heat_rgb)

def to_png(pil):
    """Encode once, at full resolution, and hand st.image the bytes.

    st.image re-encodes a PIL image on every rerun; at 1024 px that is ~0.1 s per
    panel, so touching any sidebar widget re-encoded the whole page. Identical
    bytes also get the same media URL, so the browser re-uses what it already has.
    """
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def fetch_batch(loader, index):
    """Build batch `index` directly instead of iterating the dataloader up to it.

    The sampler is sequential and its metadata already pins the episode, so the
    batch can be assembled from the dataset alone. Iterating cost ~20 ms per
    skipped episode, which is where "load sample 300" spent its seconds.
    """
    sampler = loader.batch_sampler
    if index >= len(sampler.batch_sizes):
        return None
    size = sampler.batch_sizes[index]
    start = sum(sampler.batch_sizes[:index])
    metadata = {k: v[index] for k, v in sampler.batch_metadata.items()}
    samples = [loader.dataset[(start + j, metadata)] for j in range(size)]
    return loader.collate_fn(samples)


def build_episode(raw_batch, batch_index, dataset, image_size, n_shots, device):
    """Wrap a fetched batch into the session episode, tagged with what it was built for.

    The tag matters: every model has its own input resolution (DCAMA 384, DMTNet 400,
    SANSA and INSID3 1024), and feeding a model an episode built for another one fails
    deep inside the correlation layers with a shape mismatch.
    """
    chosen, dataset_name = raw_batch
    substitutor = Substitutor(substitute=True)
    substitutor.reset(batch=chosen)
    chosen, gt = next(substitutor)
    return {
        "batch": to_device(chosen, device),
        "gt": gt.to(device),
        "dataset_name": dataset_name,
        "batch_index": batch_index,
        "signature": (dataset, image_size, n_shots),
    }


def batch_class_names(loader, index):
    """Class names of batch `index` read from the sampler metadata (no image loading).

    Returns None when the dataloader runs without an episode csv, where the classes
    are only known after the sample is built.
    """
    sampler = loader.batch_sampler
    classes = sampler.batch_metadata.get(BatchKeys.CLASSES)
    if classes is None or index >= len(classes):
        return None
    categories = next(iter(loader.dataset.categories.values()))
    query_classes = classes[index][0] if classes[index] else []
    return [categories[c]["name"] for c in query_classes if c in categories]

PASCAL_PARTS = ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation")
COCO_ANNOTATIONS = "data/coco/annotations/instances_val2014.json"


def dataset_ready(dataset: str) -> bool:
    """Whether the demo can build episodes for this dataset.

    COCO needs only the annotations: the loader streams the images from coco_url
    when no local img_dir is set. Pascal has no such fallback, so the images and
    masks have to be on disk.
    """
    if dataset == "coco":
        return os.path.exists(COCO_ANNOTATIONS)
    return all(os.path.isdir(os.path.join("data/pascal", part)) for part in PASCAL_PARTS)


def download_dataset(dataset: str) -> bool:
    """Run the download script for `dataset`, streaming its output into the page."""
    script = "scripts/download_coco_jsononly.sh" if dataset == "coco" else "scripts/download_pascal.sh"
    with st.status(f"Downloading {dataset}...", expanded=True) as status:
        proc = subprocess.Popen(["bash", script], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        box, tail = st.empty(), []
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-10:]
            box.code("\n".join(tail))
        ok = proc.wait() == 0
        status.update(label=f"{dataset} ready" if ok else f"{dataset} download failed",
                      state="complete" if ok else "error")
    return ok


@st.cache_resource(show_spinner="Loading model...")
def load_model(model: str, device: str):
    # SANSA is built differentiable so gradient-based explainers work in the app;
    # plain forwards still run under torch.no_grad, so eval cost is unchanged.
    extra = {"differentiable": True} if model == "sansa" else {}
    model_obj, image_size = build_model_preconfigured(model_name=model, **extra)
    model_obj.eval()
    model_obj = model_obj.to(device)
    return model_obj, image_size


@st.cache_resource(show_spinner="Loading dataset...")
def load_loader(dataset: str, image_size: int, n_shots: int):
    """Build the dataloader for one (dataset, image_size, n_shots) triple.

    The episode csv in data_csv pins the image ids of every episode, so the number of
    shots comes from which csv is read, not from the n_shots parameter: reading the
    1-shot csv gives 1-shot episodes whatever n_shots says. We therefore pick the csv
    that matches the requested shots, and fall back to sampling episodes on the fly
    (no csv) for shot counts that have no csv.
    """
    dataset_cfg = copy.deepcopy(config_coco if dataset == "coco" else config_pascal)
    dataloader_cfg = copy.deepcopy(config_dataloader)

    base = "val_coco20i" if dataset == "coco" else "val_pascal5i"
    (old_key,) = list(dataset_cfg["datasets"].keys())
    params = dataset_cfg["datasets"].pop(old_key)
    params["n_shots"] = n_shots
    params["image_size"] = image_size

    csv_key = f"{base}_N1K{n_shots}"
    csv_folder = dataloader_cfg.get("csv_folder")
    if csv_folder and os.path.exists(os.path.join(csv_folder, f"{csv_key}.csv")):
        dataset_cfg["datasets"][csv_key] = params
    else:
        # No fixed episode list for this shot count: sample episodes instead, which
        # honours n_shots directly.
        dataset_cfg["datasets"][f"{base}_N1K{n_shots}"] = params
        dataloader_cfg.pop("csv_folder", None)

    dataset_cfg["preprocess"] = {"image_size": image_size}

    val_loader = get_dataloaders(
        dataset_cfg,
        dataloader_cfg,
        num_processes=1,
    )
    return val_loader

def run_model_on_batch(model, batch, gt):
    # The batch may be cached from a previous device selection; always move it to
    # wherever the (possibly rebuilt) model actually lives.
    model_device = next(model.parameters()).device
    batch = to_device(batch, model_device)
    if gt is not None:
        gt = gt.to(model_device)
    target_shape = batch[BatchKeys.IMAGES][:, 0].shape[2:]
    with torch.no_grad():
        result = model(batch, postprocess=False)
    logits = F.interpolate(
        result[ResultDict.LOGITS],
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    pred_seg = logits.argmax(dim=1)
    return result, logits, pred_seg

def tint_foreground(image_3chw, mask_hw, tint=(0.2, -0.1, -0.1), tint_border=(0.5, -0.3, -0.3)):
    thickness = 3
    image_3chw[:, mask_hw.cpu().bool()] += torch.tensor(tint).unsqueeze(-1)
    image_3chw = torch.clamp(image_3chw, 0, 1)
    query_mask = (mask_hw > 0.5).unsqueeze(0).unsqueeze(0).float()
    kernel = 2 * thickness + 1
    erosion = -F.max_pool2d(-query_mask, kernel_size=kernel, stride=1, padding=thickness)
    border_mask = (query_mask - erosion) > 0
    border_mask = border_mask.squeeze(0).squeeze(0).cpu().bool()
    image_3chw[:, border_mask] += torch.tensor(tint_border).unsqueeze(-1)
    image_3chw = torch.clamp(image_3chw, 0, 1)
    return image_3chw

def visualize_episode_header(input_dict):
    # Rendered PILs are cached inside the episode dict, so sidebar changes do not redo
    # the tensor->image conversions on every rerun.
    episode = st.session_state.get("episode", {})
    pils = episode.get("header_pils")
    if pils is None:
        rgb_images = unnormalize(input_dict[BatchKeys.IMAGES])
        support_images = rgb_images[0, 1:].clone()
        support_masks = input_dict[BatchKeys.PROMPT_MASKS][0, 0:, 1].clone()
        n_shots = support_masks.shape[0]
        pils = {
            "query": to_png(tensor_to_pil(rgb_images[0, 0])),
            "supports": [
                to_png(tensor_to_pil(tint_foreground(support_images[i].cpu(), support_masks[i].cpu().bool())))
                for i in range(n_shots)
            ],
        }
        if episode:
            episode["header_pils"] = pils
    n_shots = len(pils["supports"])

    st.markdown('<p class="sub-header">1. The Few-Shot Episode</p>', unsafe_allow_html=True)
    st.caption("The model is presented with a Query Image and a Support Set (Images + Masks) to learn the class on-the-fly.")

    col_query, col_support = st.columns([1, 2])

    with col_query:
        st.markdown("**Query Image** (To Segment)")
        st.image(pils["query"], use_container_width=True)

    with col_support:
        st.markdown(f"**Support Set** ({n_shots}-shot)")
        cols = st.columns(n_shots)
        for i in range(n_shots):
            with cols[i]:
                st.image(pils["supports"][i], caption=f"Shot {i}", use_container_width=True)

def show_overlay(rgb_images, logits, seg, gt):
    # Render once per inference result; sidebar reruns reuse the cached PILs.
    cache = st.session_state.get("result", {})
    pils = cache.get("overlay_pils")
    if pils is None:
        query_image = rgb_images[0, 0].clone()
        # Fill and border share the colour: blue is the ground truth, red is the prediction.
        red, red_border = (0.7, 0, 0), (0.9, -0.3, -0.3)
        blue, blue_border = (0, 0, 0.7), (-0.3, -0.3, 0.9)
        tinted_seg = tint_foreground(query_image.clone(), seg[0].cpu().bool(), tint=red, tint_border=red_border)
        tinted_gt = tint_foreground(query_image.clone(), gt.cpu().bool(), tint=blue, tint_border=blue_border)
        tinted_both = tint_foreground(tinted_gt.clone(), seg[0].cpu().bool(), tint=red, tint_border=red_border)
        pils = {
            "seg": to_png(tensor_to_pil(tinted_seg)),
            "gt": to_png(tensor_to_pil(tinted_gt)),
            "both": to_png(tensor_to_pil(tinted_both)),
            "logits": to_png(tensor_to_heatmap(logits[0, 1].cpu())),
        }
        if cache:
            cache["overlay_pils"] = pils

    st.markdown('<p class="sub-header">2. Segmentation Result</p>', unsafe_allow_html=True)

    tabs = st.tabs(["Comparison", "Detailed Masks"])

    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(pils["seg"], caption="Prediction (Red)", use_container_width=True)
        with col2:
            st.image(pils["gt"], caption="Ground Truth (Blue)", use_container_width=True)
        with col3:
            st.image(pils["both"], caption="Overlay (Both)", use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        col1.image(pils["logits"], caption="Logits Heatmap", use_container_width=True)
        col2.info("The logits heatmap represents the raw confidence of the model before argmax.")

def apply_custom_support_masks(batch, custom_masks: dict):
    """Return a shallow copy of batch with PROMPT_MASKS replaced for given shot indices."""
    if not custom_masks:
        return batch
    import copy as _copy
    batch = _copy.copy(batch)
    masks = batch[BatchKeys.PROMPT_MASKS].clone()
    for shot_idx, mask_tensor in custom_masks.items():
        masks[0, shot_idx, 1] = mask_tensor.float()
    batch[BatchKeys.PROMPT_MASKS] = masks
    return batch


MODEL_NAMES = {
    "dcama": "DCAMA",
    "dmtnet": "DMTNet",
    "insid3": "INSID3",
    "sansa": "SANSA",
    "gfsam": "GF-SAM",
    "matcher": "Matcher",
    "panet": "PANet",
}

# Paper names, not code names: `signed_affinity` is what the paper calls AffEx, and the
# mask-free `affinity` is Unmasked AffEx.
EXPLAINER_NAMES = {
    "signed_affinity": "AffEx",
    "affinity": "Unmasked AffEx",
    "random": "Random",
    "gaussian_noise": "Gaussian Noise Mask",
    "saliency": "Saliency",
    "integrated_gradients": "Integrated Gradients",
    "guided_ig": "Guided IG",
    "blur_ig": "Blur IG",
    "xrai": "XRAI",
    "deep_lift": "Deep Lift",
    "lime": "LIME",
    "gradcam": "Grad-CAM",
    "gradient_shap": "Gradient SHAP",
    "prototype_readout": "Prototype Read-out",
    "smoothed_affinity": "Smoothed AffEx",
    "masked_affinity": "Masked AffEx",
    "reverse_signed_affinity": "Reverse-signed AffEx",
    "encoder_affinity": "Encoder AffEx",
}

# Order the picker by what a visitor should try first.
EXPLAINER_ORDER = ["signed_affinity", "affinity", "lime", "saliency", "blur_ig", "xrai",
                   "integrated_gradients", "guided_ig", "deep_lift", "gaussian_noise", "random"]

AFFINITY_DEFAULTS = {
    "explanation_size": 256,
    "mask_blur_kernel_size": 7,
    "mask_blur_sigma": 50,
    "mask_dilation_radius": 5,
    "mask_dilation_kernel": 7,
}

# Per-model starting point. INSID3 and SANSA keep the paper settings; on DCAMA and DMTNet
# the dilation is dropped, which leaves the attribution maps sharper to look at.
AFFINITY_BY_MODEL = {
    "insid3": dict(AFFINITY_DEFAULTS, explanation_size=64),
    "sansa": dict(AFFINITY_DEFAULTS, explanation_size=64),
    "gfsam": dict(AFFINITY_DEFAULTS, explanation_size=64),
    "matcher": dict(AFFINITY_DEFAULTS, explanation_size=64),
    "dcama": dict(AFFINITY_DEFAULTS, mask_dilation_radius=0, mask_dilation_kernel=None),
    "dmtnet": dict(AFFINITY_DEFAULTS, mask_dilation_radius=0, mask_dilation_kernel=None),
}


def affinity_defaults(model_name):
    return dict(AFFINITY_BY_MODEL.get(model_name, AFFINITY_DEFAULTS))


def label_for(key, table):
    return table.get(key, key.replace("_", " ").title())


def debug_mode():
    """Advanced controls are hidden unless debug is on (AFFEX_DEBUG=1, or the toggle)."""
    return st.session_state.get("debug_mode", os.environ.get("AFFEX_DEBUG") == "1")


def debug_extra_explainers():
    """The variants that only make sense when poking at the method itself."""
    if not debug_mode():
        return []
    return [k for k in sorted(EXPLAINER_REGISTRY) if k not in EXPLAINER_ORDER]


CANVAS_WIDTH = 640


@st.fragment
def custom_region_picker(query_chw, shape):
    """Draw the region to explain directly on the query image.

    The drawing happens in the browser, so strokes cost nothing; the mask is only sent
    back when the drawing is released. Freehand for quick blobs, rectangle for a crop,
    polygon for precise outlines.
    """
    img_h, img_w = shape
    width = min(CANVAS_WIDTH, img_w)
    height = max(1, round(img_h * width / img_w))

    col_canvas, col_side = st.columns([3, 1], vertical_alignment="top")
    with col_side:
        mode = st.radio("Tool", ["freedraw", "rect", "polygon"], key="canvas_mode",
                        format_func=lambda m: {"freedraw": "Brush", "rect": "Rectangle",
                                               "polygon": "Polygon"}[m])
        stroke = st.slider("Brush size", 5, 120, 40, step=5, key="canvas_stroke",
                           disabled=mode != "freedraw")

    background = tensor_to_pil(query_chw).resize((width, height), Image.BILINEAR)
    with col_canvas:
        result = st_canvas(
            fill_color="rgba(220, 40, 40, 0.45)",
            stroke_color="rgba(220, 40, 40, 0.85)",
            stroke_width=stroke if mode == "freedraw" else 2,
            background_image=background,
            update_streamlit=True,     # only fires when a stroke is finished
            height=height, width=width,
            drawing_mode=mode,
            display_toolbar=True,
            key="custom_canvas",
        )

    mask = torch.zeros(img_h, img_w, dtype=torch.bool)
    if result is not None and result.image_data is not None:
        drawn = torch.from_numpy(result.image_data[..., 3] > 0)   # alpha of the drawing
        if drawn.any():
            mask = F.interpolate(drawn[None, None].float(), size=(img_h, img_w),
                                 mode="nearest")[0, 0] > 0.5
    st.session_state["custom_mask"] = mask
    if not mask.any():
        st.caption("Draw on the image to choose the region to explain.")
    else:
        st.caption(f"Region: {int(mask.sum())} pixels "
                   f"({100 * mask.float().mean().item():.1f}% of the query).")


def build_and_run_explainer(
    name, parameters, model, input_dict, device, explanation_mask=None, affinity_params=None
):
    # Affinity-family explainers use the paper's mask processing (dilation + Gaussian
    # smoothing of the support mask before sign assignment); constructor defaults are
    # the unprocessed binary mask.
    explainer_params = dict(affinity_params or AFFINITY_DEFAULTS) if "affinity" in name else {}
    explainer = build_explainer(name=name, model=model, params=explainer_params, device=device)
    n_ways = parameters.get("n_ways", 1)
    input_dict = to_device(input_dict, device)
    target_shape = input_dict[BatchKeys.IMAGES][:, 0].shape[2:]

    with torch.no_grad():
        result = model(input_dict, postprocess=False)

    logits = F.interpolate(
        result[ResultDict.LOGITS],
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    pred_seg = logits.argmax(dim=1)

    if explanation_mask is None:
        explanation_mask = (
            F.one_hot(pred_seg, num_classes=n_ways + 1).permute(0, 3, 1, 2)[0].bool()[1]
        )
    else:
        explanation_mask = explanation_mask.to(device)

    model_expl = explainer.explain(
        input_dict=input_dict,
        explanation_mask=explanation_mask,
    )[0]
    return model_expl, explanation_mask

def download_model(model_name: str):
    # Stream output of the download script into Streamlit
                            cmd = ["bash", f"scripts/download_{model_name.lower()}.sh"]
                            proc = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                            )
                            log_placeholder = st.empty()
                            logs = ""
                            for line in proc.stdout:
                                logs += line
                                # update a code block so newlines render nicely
                                log_placeholder.code(logs, language="bash", height=100)
                            proc.stdout.close()
                            ret = proc.wait()
                            if ret != 0:
                                raise subprocess.CalledProcessError(ret, cmd, output=logs)

import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# --- Main App ---
def main():
    st.set_page_config(page_title="Affinity Explainer", layout="wide", page_icon="imgs/icon.svg")
    local_css()
    
    # --- Header ---
    # Load the svg file as base64
    svg_file_path = "imgs/icon.svg"
    svg_base64 = get_base64_of_bin_file(svg_file_path)
    
    # Inject the base64 string into the source
    st.markdown(
        f'# <img src="data:image/svg+xml;base64,{svg_base64}" alt="icon" width="64" style="vertical-align: middle;"/> AffinityExplainer',
        unsafe_allow_html=True
    )
    st.markdown("""
    Interpretability for **Matching-Based Few-Shot Semantic Segmentation**.
    This tool visualizes how support images influence the segmentation of the query image using the *Affinity Explainer* method.
    """)
    
    if _imports_error is not None:
        error_box("Project-specific modules are missing. Please check environment.", _imports_error)
        st.stop()
        
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("Dataset Settings", expanded=True):
            dataset = st.selectbox("Dataset", ["coco", "pascal"], index=0,
                                   format_func=lambda k: {"coco": "COCO-20$^i$", "pascal": "Pascal-5$^i$"}[k])
            num_shots = st.number_input("N-Shots", min_value=1, value=2)

        with st.expander("Model Settings", expanded=True):
            model_name = st.selectbox(
                "Model", ["dcama", "dmtnet", "insid3", "sansa", "gfsam"],
                format_func=lambda k: label_for(k, MODEL_NAMES))
            
            # insid3/sansa load their weights from their own repos, not checkpoints/<name>/
            if model_name in ("dcama", "dmtnet") and not os.path.exists(f"checkpoints/{model_name}/"):
                if st.button(f"📥 Download {model_name.upper()} Checkpoints"):
                    with st.spinner("Downloading..."):
                        download_model(model_name)
            
            cuda_count = torch.cuda.device_count()
            device_options = ["cpu"] + [f"cuda:{i}" for i in range(cuda_count)] if cuda_count > 0 else ["cpu"]
            device = st.selectbox("Compute Device", device_options, index=1 if cuda_count > 0 else 0)
            
        with st.expander("Explanation Settings", expanded=True):
            options = ([k for k in EXPLAINER_ORDER if k in EXPLAINER_REGISTRY] + debug_extra_explainers())
            name = st.selectbox("Explanation Method", options, index=0,
                                format_func=lambda k: label_for(k, EXPLAINER_NAMES))
            affinity_params = affinity_defaults(model_name)
            if "affinity" in name and debug_mode():
                st.caption("Affinity hyperparameters")
                affinity_params["explanation_size"] = st.select_slider(
                    "Similarity Map Resolution",
                    options=[32, 64, 128, 256],
                    value=affinity_params["explanation_size"],
                    help="Resolution the per-layer similarity maps are interpolated to.",
                )
                col_dil, col_blur = st.columns(2)
                affinity_params["mask_dilation_radius"] = col_dil.number_input(
                    "Dilation Radius", min_value=0, max_value=20,
                    value=affinity_params["mask_dilation_radius"],
                )
                affinity_params["mask_dilation_kernel"] = col_dil.number_input(
                    "Dilation Kernel", min_value=1, max_value=15, step=2,
                    value=affinity_params["mask_dilation_kernel"] or 7,
                )
                affinity_params["mask_blur_kernel_size"] = col_blur.number_input(
                    "Blur Kernel", min_value=1, max_value=31, step=2,
                    value=affinity_params["mask_blur_kernel_size"],
                )
                affinity_params["mask_blur_sigma"] = col_blur.number_input(
                    "Blur Sigma", min_value=0.1, max_value=200.0, step=1.0,
                    value=float(affinity_params["mask_blur_sigma"]),
                )

        st.divider()
        st.toggle("Debug mode", key="debug_mode",
                  value=os.environ.get("AFFEX_DEBUG") == "1",
                  help="Show the attribution hyperparameters and every explainer variant.")
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.clear()
            if "cuda" in device: torch.cuda.empty_cache()
            st.rerun()

    # A missing dataset is a dead end for everything below, so stop here and offer
    # the download rather than failing later inside the dataloader.
    if not dataset_ready(dataset):
        size = "about 200 MB" if dataset == "coco" else "about 2 GB"
        st.warning(f"The {dataset} dataset is not installed. The demo needs it to build episodes.")
        if dataset == "pascal":
            st.caption("COCO works without any download: its images are streamed on demand.")
        if st.button(f"📥 Download {dataset} ({size})", type="primary"):
            if download_dataset(dataset):
                st.rerun()
        st.stop()

    # --- Main Logic ---
    
    # Setup Parameters
    parameters = {
        "dataset": copy.deepcopy(config_coco if dataset == "coco" else config_pascal),
        "dataloader": copy.deepcopy(config_dataloader),
        "model": {"name": model_name},
        "explainer": {"name": name},
    }
    for k in parameters["dataset"]["datasets"]:
        parameters["dataset"]["datasets"][k]["n_shots"] = num_shots

    # 1. Load Model and data, cached independently of each other and of the sidebar
    # settings they do not depend on.
    try:
        model, image_size = load_model(model_name, device)
        loader = load_loader(dataset, image_size, int(num_shots))
        loader_iter = next(iter(loader.values()))
    except Exception as e:
        error_box("Failed to build model.", e)
        st.stop()

    # 2. Episode picker, in the main window: the sample is the thing a visitor changes
    # most, so it sits above the images and loads on change rather than behind a button.
    n_episodes = len(loader_iter.batch_sampler.batch_sizes)
    st.session_state.setdefault("sample_index", 0)

    def _step(delta):
        st.session_state["sample_index"] = max(0, min(n_episodes - 1,
                                                      st.session_state["sample_index"] + delta))

    pick = st.container()
    with pick:
        col_prev, col_idx, col_next, col_filter = st.columns([1, 2, 1, 3], vertical_alignment="bottom")
        col_prev.button("◀ Previous", use_container_width=True, disabled=st.session_state["sample_index"] <= 0,
                        on_click=_step, args=(-1,))
        sample_index = col_idx.number_input(
            f"Episode (0 to {n_episodes - 1})", min_value=0, max_value=max(0, n_episodes - 1),
            step=1, key="sample_index")
        col_next.button("Next ▶", use_container_width=True,
                        disabled=st.session_state["sample_index"] >= n_episodes - 1,
                        on_click=_step, args=(1,))
        favorite_class = col_filter.text_input("Filter by class", value="",
                                               placeholder="e.g. person", key="class_filter")

    # Load whenever the request differs from what is in hand: no button to press.
    wanted = (dataset, image_size, int(num_shots), int(sample_index), favorite_class)
    loaded = st.session_state.get("episode", {}).get("request")
    load_batch = wanted != loaded

    if load_batch:
        with st.spinner("Fetching batch data..."):
            chosen = None
            resolved_index = None
            try:
                if favorite_class == "":
                    resolved_index = int(sample_index)
                    chosen = fetch_batch(loader_iter, resolved_index)
                else:
                    # Class filtering reads the episode metadata, so only the matching
                    # episode is decoded instead of every episode up to it.
                    n_batches = len(loader_iter.batch_sampler.batch_sizes)
                    matches, metadata_filtering = [], True
                    for b in range(n_batches):
                        names = batch_class_names(loader_iter, b)
                        if names is None:
                            metadata_filtering = False
                            break
                        if favorite_class in names:
                            matches.append(b)

                    if metadata_filtering:
                        if int(sample_index) < len(matches):
                            resolved_index = matches[int(sample_index)]
                            chosen = fetch_batch(loader_iter, resolved_index)
                        else:
                            st.warning(
                                f"Only {len(matches)} episodes of class '{favorite_class}' "
                                f"(asked for index {int(sample_index)})."
                            )
                    else:
                        i = 0
                        progress_bar = st.progress(0, text=f"Searching for {favorite_class}...")
                        for batch in loader_iter:
                            categories = loader_iter.dataset.datasets[batch[1][0]].categories
                            classes = batch[0][0][BatchKeys.CLASSES][0][0]
                            class_names = [categories[c]["name"] for c in classes]
                            if favorite_class in class_names:
                                if i == int(sample_index):
                                    chosen = batch
                                    break
                                i += 1
                            progress_bar.progress(min(i/50, 1.0)) # Arbitrary max for visuals
                        progress_bar.empty()

                if chosen:
                    st.session_state["episode"] = build_episode(
                        chosen, resolved_index, dataset, image_size, int(num_shots), device
                    )
                    st.session_state["episode"]["request"] = wanted
                    # Reset downstream results
                    for key in ["result", "explanation", "explanation_mask", "metrics", "explanation_pils"]:
                        st.session_state.pop(key, None)
                else:
                    st.warning("Sample not found (check index or class name).")

            except Exception as e:
                error_box("Dataloader error.", e)
                st.stop()

    # 3. Display & Run
    # An episode loaded for one model is unusable by a model with a different input
    # resolution, so re-fetch the same sample whenever the signature no longer matches.
    if "episode" in st.session_state:
        stale = st.session_state["episode"].get("signature") != (dataset, image_size, int(num_shots))
        if stale:
            idx = st.session_state["episode"].get("batch_index")
            st.session_state.pop("episode", None)
            for key in ["result", "explanation", "explanation_mask", "metrics", "explanation_pils"]:
                st.session_state.pop(key, None)
            if idx is not None:
                raw = fetch_batch(loader_iter, idx)
                if raw is not None:
                    st.session_state["episode"] = build_episode(
                        raw, idx, dataset, image_size, int(num_shots), device
                    )
                    st.info(
                        f"Episode {idx} reloaded: {model_name}, {image_size}px, "
                        f"{int(num_shots)}-shot."
                    )
            if "episode" not in st.session_state:
                st.warning("Load the batch again: the previous episode was built for another model.")

    if "episode" in st.session_state:
        episode = st.session_state["episode"]
        chosen = episode["batch"]
        gt = episode["gt"]
        
        visualize_episode_header(chosen)

        # Custom support mask uploads
        n_shots_ep = chosen[BatchKeys.PROMPT_MASKS].shape[1]
        mask_H, mask_W = chosen[BatchKeys.PROMPT_MASKS].shape[-2:]
        custom_support_masks: dict = {}
        with st.expander("🖼️ Custom Support Masks (optional)"):
            st.caption(
                "Upload a binary image (white = foreground) to replace the support mask for any shot. "
                "Leave empty to keep the dataset mask."
            )
            upload_cols = st.columns(n_shots_ep)
            for i in range(n_shots_ep):
                with upload_cols[i]:
                    uploaded = st.file_uploader(
                        f"Shot {i}", type=["png", "jpg", "jpeg"], key=f"supp_mask_{i}"
                    )
                    if uploaded is not None:
                        mask_img = Image.open(uploaded).convert("L").resize(
                            (mask_W, mask_H), Image.NEAREST
                        )
                        mask_arr = np.array(mask_img)
                        mask_tensor = torch.from_numpy(mask_arr > 128)
                        custom_support_masks[i] = mask_tensor
                        tinted = tint_foreground(
                            unnormalize(chosen[BatchKeys.IMAGES])[0, i + 1].clone().cpu(),
                            mask_tensor,
                        )
                        st.image(tensor_to_pil(tinted), caption=f"Shot {i} (custom)", use_container_width=True)

        # The region whose attribution we ask for: it belongs next to the images it
        # refers to, not in the sidebar.
        st.markdown("#### Explanation region")
        mask_source = st.radio(
            "Explanation Mask",
            ["Prediction", "Ground Truth", "Custom Area"],
            horizontal=True,
            label_visibility="collapsed",
            key="mask_source",
            help="Which part of the query the attribution is computed for.",
        )

        explanation_mask_input = None
        if mask_source == "Ground Truth":
            # Positive labels only. Pascal marks VOC's border band with the ignore label
            # -100, which is truthy, so .bool() would fold the outline of every annotated
            # object in the image (people, bicycles) into the mask.
            explanation_mask_input = gt[0] > 0
        elif mask_source == "Custom Area":
            custom_region_picker(
                unnormalize(chosen[BatchKeys.IMAGES])[0, 0].clone().cpu(),
                (chosen[BatchKeys.IMAGES].shape[3], chosen[BatchKeys.IMAGES].shape[4]),
            )
            explanation_mask_input = st.session_state.get("custom_mask")

        if st.button("🚀 Run Inference & Explanation", type="primary", use_container_width=True):
            with st.status("Processing...", expanded=True) as status:
                st.write("Running Model Forward Pass...")
                try:
                    batch_to_run = apply_custom_support_masks(chosen, custom_support_masks)
                    result, logits, pred_seg = run_model_on_batch(model, batch_to_run, gt)
                    st.write("Computing Affinity Explanation...")
                    exp, explanation_mask = build_and_run_explainer(
                        name, parameters, model, batch_to_run, device, explanation_mask_input,
                        affinity_params=affinity_params,
                    )
                except Exception as e:
                    error_box("Model Error", e)
                    st.stop()

                st.session_state["result"] = {"logits": logits, "pred_seg": pred_seg}
                st.session_state["explanation"] = exp
                st.session_state.pop("explanation_pils", None)  # invalidate cached renders
                st.session_state["explanation_mask"] = explanation_mask
                st.session_state.pop("metrics", None)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

    # 4. Results & Explainability
    if "result" in st.session_state:
        batch_result = st.session_state["result"]
        # Segmentation Results (tensor->PIL conversions run only on the first display;
        # afterwards show_overlay reuses the cached images)
        if batch_result.get("overlay_pils") is not None:
            show_overlay(None, None, None, None)
        else:
            rgb_images = unnormalize(chosen[BatchKeys.IMAGES])
            show_overlay(rgb_images.cpu(), batch_result["logits"].cpu(), batch_result["pred_seg"].cpu(), (gt[0] > 0).cpu())

        # Explanation Heatmaps
        st.markdown('<p class="sub-header">3. Affinity Explanation</p>', unsafe_allow_html=True)
        st.markdown("""
        The explanation highlights regions in the **Support Set** that were most influential for the query segmentation.
        """)
        
        exp_pils = st.session_state.get("explanation_pils")
        if exp_pils is None:
            exp = min_max_scale(st.session_state["explanation"])  # Normalize for vis
            exp_pils = [to_png(tensor_to_heatmap(exp[0, i])) for i in range(exp.shape[1])]
            st.session_state["explanation_pils"] = exp_pils

        cols = st.columns(len(exp_pils))
        for i, pil in enumerate(exp_pils):
            with cols[i]:
                st.image(pil, caption=f"Support Attribution {i}", use_container_width=True)

        st.divider()
        
        # 5. Causal Metrics (IAUC / DAUC)
        st.markdown('<p class="sub-header">4. Causal Evaluation (IAUC & DAUC)</p>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="highlight-box">
            <strong>Metric Logic:</strong> We progressively insert or delete pixels from the support set based on their attribution score 
            and measure the impact on the model's prediction (mIoU or Logits). 
            <br>
            <ul>
            <li><strong>IAUC (Insertion AUC):</strong> Does adding "important" pixels improve the result quickly?</li>
            <li><strong>DAUC (Deletion AUC):</strong> Does removing "important" pixels degrade the result quickly?</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            

            col_param1, col_param2, col_run = st.columns([1, 1, 2])
            n_steps = col_param1.number_input("Steps", 10, 100, 30)
            measure = col_param2.selectbox("Measure", ["miou", "logits"])
            
            if col_run.button("📉 Calculate Metrics", use_container_width=True):
                metrics = {}
                metrics["iauc"] = FSSCausalMetric(
                    model=model, mode="ins", n_steps=n_steps, measure=measure,
                    n_mid_statuses=n_steps, mid_statuses_distribution="linear",
                )
                metrics["dauc"] = FSSCausalMetric(
                    model=model, mode="del", n_steps=n_steps, measure=measure,
                    n_mid_statuses=n_steps, mid_statuses_distribution="linear",
                )
                
                explanation_mask = st.session_state["explanation_mask"]
                explanation = st.session_state["explanation"]
                st.session_state["metrics"] = {}

                for name, metric in metrics.items():
                    progress = st.progress(0, "Calculating " + name.upper())
                    for _, i, _ in metric.evaluate_interactive(
                        chosen, explanation, explanation_mask, gt=gt
                    ):
                        # Note: metric.computed_n_steps is only available *inside* loop or after first step
                        progress.progress(
                            min(i / (metric.computed_n_steps + 1), 1.0),
                            f"Calculating {name.upper()}, step {i}/{metric.computed_n_steps}",
                        )
                    st.session_state["metrics"][name] = {
                        "auc": metric.xauc,
                        "scores": metric.scores,
                        "mid_statuses": metric.mid_statuses,
                    }

        # 6. Interactive Dashboard
        if "metrics" in st.session_state:
            st.markdown("### Interactive Analysis")
            
            metric_names = list(st.session_state["metrics"].keys())
            tabs = st.tabs([f"📊 {n.upper()} Analysis" for n in metric_names])

            for i, name in enumerate(metric_names):
                with tabs[i]:
                    result = st.session_state["metrics"][name]
                    mid_statuses = result["mid_statuses"]
                    scores = result["scores"]
                    auc_val = result['auc']

                    # Layout: Top row stats, Bottom row plots + slider
                    st.metric(f"{name.upper()} Score (AUC)", f"{auc_val:.4f}")

                    # Slider
                    if len(mid_statuses) > 0:
                        selected_idx = st.slider(
                            f"Scrub through {name.upper()} Perturbation Steps",
                            0, len(mid_statuses) - 1, 0,
                            key=f"slider_{name}"
                        )
                        current_step_int, mid_images, mid_masks, probs, _ = mid_statuses[selected_idx]
                    else:
                        st.warning("No status data.")
                        continue

                    # Plotting
                    col_viz, col_plot = st.columns([1, 1])
                    
                    with col_plot:
                        st.markdown("**Performance Curve**")
                        batch_element = 0 
                        x = np.linspace(0, 1, scores.shape[0])
                        y = scores[:, batch_element]
                        curr_x = x[min(current_step_int, len(x)-1)]
                        curr_y = y[min(current_step_int, len(y)-1)]

                        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                        ax.plot(x, y, linewidth=2, color='#2C3E50')
                        ax.fill_between(x, 0, y, alpha=0.2, color='#2C3E50')
                        ax.axvline(x=curr_x, color='#E74C3C', linestyle='--')
                        ax.plot(curr_x, curr_y, 'o', color='#E74C3C', markersize=8)
                        ax.set_xlabel("Perturbation Ratio")
                        ax.set_ylabel("Score")
                        ax.set_title(f"{name.upper()} Trajectory")
                        ax.grid(True, alpha=0.3)
                        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Downloads
                        with st.expander("📥 Download Plot Data"):
                            csv_buf = StringIO()
                            pd.DataFrame(scores.numpy()).to_csv(csv_buf, index=False)
                            st.download_button("Download CSV", csv_buf.getvalue(), f"{name}.csv", "text/csv")

                    with col_viz:
                        st.markdown(f"**Visual State @ Step {current_step_int}**")
                        
                        # Image tabs
                        viz_tabs = st.tabs(["Modified Support", "Model Segmentation", "Probabilities"])
                        
                        with viz_tabs[0]:
                            # Show just the first shot for brevity if multiple
                            masked_shot = tint_foreground(
                                unnormalize(mid_images[0, 1])[0].cpu(),
                                mid_masks[0, 0, 1].cpu().bool(),
                            )
                            st.image(tensor_to_pil(masked_shot), caption=f"Support Shot 0 (Perturbed)", use_container_width=True)
                        
                        with viz_tabs[1]:
                            mid_pred_seg = probs.argmax(dim=1)[0]
                            seg_rgb = tint_foreground(
                                unnormalize(mid_images[0, 0])[0].cpu(),
                                mid_pred_seg.cpu().bool(),
                            )
                            st.image(tensor_to_pil(seg_rgb), caption="Resulting Segmentation", use_container_width=True)
                            
                        with viz_tabs[2]:
                            st.image(tensor_to_heatmap(probs[0, 1].cpu()), caption="Class Probability", use_container_width=True)
                    
                    # --- Re-added Bulk Download ---
                    st.divider()
                    if st.button(f"📦 Generate ZIP of all images ({name.upper()})", key=f"zip_{name}"):
                        try:
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as zf:
                                # Iterate over ALL statuses just for the zip
                                for s_idx, m_stat in enumerate(mid_statuses):
                                    j_step, _, _, m_probs, _ = m_stat
                                    
                                    # Save Probability Map
                                    prob_pil = tensor_to_heatmap(m_probs[0, 1].cpu())
                                    p_buf = BytesIO()
                                    prob_pil.save(p_buf, format="PNG")
                                    zf.writestr(f"{name}_step{j_step}_probs.png", p_buf.getvalue())
                                    
                                    # (Can add segmentation or support images here similarly if needed)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download ZIP now",
                                data=zip_buffer.getvalue(),
                                file_name=f"{name}_mid_statuses.zip",
                                mime="application/zip",
                            )
                        except Exception as e:
                            st.error(f"Error creating zip: {e}")

            st.success("Complete.")

def launch():
    # Resolve the script through the package, so `uvx --from <repo> app` works from
    # anywhere and not only from a clone's working directory.
    cmd = ["streamlit", "run", str(Path(__file__).resolve())]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logs = ""
    for line in proc.stdout:
        logs += line
        # print logs to console as well
        print(line, end="")
    proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output=logs)

if __name__ == "__main__":
    main()