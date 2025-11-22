# app.py
# Streamlit app to replicate the notebook: load model & dataset, pick a sample, run prediction and explanation,
# and visualize with lovely-tensors.

import copy
import traceback
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt           
import subprocess


# Core deps used in the original notebook
import torch
import torch.nn.functional as F

# Lovely Tensors for visualization (must be installed)
import lovely_tensors as lt

from affex.explainer.affinity import MODEL_EXPLAINER_REGISTRY
from affex.metrics import FSSCausalMetric

lt.monkey_patch()

from PIL import Image

# Optional saliency
try:
    import saliency.core as saliency  # google-research/tcav style API
except Exception:
    saliency = None

from io import BytesIO, StringIO

# Project-specific imports (same names used in the notebook)
try:
    from affex.data.utils import BatchKeys, min_max_scale
    from affex.utils.torch import to_device
    from affex.utils.utils import ResultDict
    from affex.utils.grid import create_experiment, load_yaml
    from affex.models import build_model_preconfigured
    from affex.data import get_dataloaders
    from affex.explainer import build_explainer
    from affex.substitution import Substitutor
    from affex.utils.segmentation import unnormalize
except Exception as e:
    # Defer raising; we'll show a visible error block in the UI
    _imports_error = e
else:
    _imports_error = None


# --------------------------
# Small utility helpers
# --------------------------
def error_box(msg: str, exc: Exception | None = None):
    st.error(msg)
    if exc is not None:
        with st.expander("Traceback / details"):
            st.code(
                "".join(traceback.format_exception(None, exc, exc.__traceback__))
                if exc
                else msg
            )


def tensor_to_pil(tensor):
    tensor = (tensor * 255).permute(1, 2, 0).detach().type(torch.uint8).cpu()
    return Image.fromarray(tensor.numpy())


def tensor_to_heatmap(tensor):
    array = tensor.detach().cpu().numpy()
    cmap = plt.get_cmap("jet")  # diverging; zero ≈ white
    rgba = cmap(array)  # [H, W, 4], float in [0,1]
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)  # [H, W, 3], uint8
    return Image.fromarray(heat_rgb)


@st.cache_resource(show_spinner=False)
def load_parameters(dataset: str, param_relpath: str):
    """
    Load YAML parameter file from the project tree.
    """
    return load_yaml(param_relpath)


@st.cache_resource(show_spinner=True)
def build_model_and_data(parameters: dict, model: str, device: str):
    """
    Build the model and dataloaders using the same utilities as the notebook.
    """
    model, image_size = build_model_preconfigured(model_name=model)
    model.eval()
    model = model.to(device)

    # Adjust image_size into dataset/preprocess blocks to match notebook's behavior
    if "dataset" in parameters:
        if "datasets" in parameters["dataset"]:
            for k in parameters["dataset"]["datasets"]:
                parameters["dataset"]["datasets"][k]["image_size"] = image_size
        if "preprocess" not in parameters["dataset"]:
            parameters["dataset"]["preprocess"] = {}
        parameters["dataset"]["preprocess"]["image_size"] = image_size

    # Data
    _, val_loader, _ = get_dataloaders(
        copy.deepcopy(parameters["dataset"]),
        copy.deepcopy(parameters["dataloader"]),
        num_processes=1,
    )

    return model, val_loader


def choose_loader(split: str, loaders: tuple):
    if split == "train":
        return loaders[0]
    if split == "val":
        return loaders[1]
    return loaders[2]  # test


def run_model_on_batch(model, batch, gt):

    # Infer original shape for resizing logits to the query image
    target_shape = batch[BatchKeys.IMAGES][:, 0].shape[2:]

    with torch.no_grad():
        result = model(batch, postprocess=False)

    # Resize logits (B, C, H, W) to target shape
    logits = F.interpolate(
        result[ResultDict.LOGITS],
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )

    # Pred segmentation
    pred_seg = logits.argmax(dim=1)  # (B, H, W)
    return result, logits, pred_seg


def visualize_episode(input_dict):
    """
    Show query image, support images & masks, and predicted segmentation
    """
    rgb_images = unnormalize(input_dict[BatchKeys.IMAGES])

    # Support set
    support_images = rgb_images[0, 1:].clone()
    support_masks = input_dict[BatchKeys.PROMPT_MASKS][0, 0:, 1].clone()
    n_shots = support_masks.shape[0]

    st.subheader("Episode")
    cols = st.columns(n_shots + 1) if n_shots > 0 else [st.container()]

    cols[0].image(
        tensor_to_pil(rgb_images[0, 0]), caption="Query", use_container_width=True
    )

    for i in range(n_shots):
        with cols[i + 1]:
            tinted_support = tint_foreground(
                support_images[i].cpu(), support_masks[i].cpu().bool()
            )
            st.image(
                tensor_to_pil(tinted_support),
                caption=f"Support {i}",
                use_container_width=True,
            )

    with st.expander("Support set images and masks"):
        for i in range(n_shots):
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    tensor_to_pil(support_images[i].cpu()),
                    caption=f"Support Image {i}",
                    use_container_width=True,
                )
            with col2:
                empty_image = torch.zeros(
                    3, support_masks[i].shape[0], support_masks[i].shape[1]
                )
                tinted_mask = tint_foreground(
                    empty_image.cpu(),
                    support_masks[i].cpu().bool(),
                    tint=(0.7, 0, 0),
                    tint_border=(0.9, 0, 0),
                )
                st.image(
                    tensor_to_pil(tinted_mask),
                    caption=f"Support Mask {i}",
                    use_container_width=True,
                )


def tint_foreground(
    image_3chw, mask_hw, tint=(0.2, -0.1, -0.1), tint_border=(0.5, -0.3, -0.3)
):
    thickness = 3  # desired border thickness in pixels
    # Add red tint to segmented area
    image_3chw[:, mask_hw.cpu().bool()] += torch.tensor(tint).unsqueeze(-1)
    image_3chw = torch.clamp(image_3chw, 0, 1)

    query_mask = (mask_hw > 0.5).unsqueeze(0).unsqueeze(0).float()  # 1x1xHxW
    kernel = 2 * thickness + 1
    erosion = -F.max_pool2d(
        -query_mask, kernel_size=kernel, stride=1, padding=thickness
    )  # morphological erosion
    border_mask = (
        query_mask - erosion
    ) > 0  # True where mask is present but erosion removed it (the border)
    border_mask = border_mask.squeeze(0).squeeze(0).cpu().bool()  # HxW bool tensor

    image_3chw[:, border_mask] += torch.tensor(tint_border).unsqueeze(-1)

    image_3chw = torch.clamp(image_3chw, 0, 1)

    return image_3chw


def show_overlay(rgb_images, logits, seg, gt):
    query_image = rgb_images[0, 0].clone()
    segmentation = seg[0]
    tinted_seg = tint_foreground(query_image.clone(), segmentation.cpu().bool())
    tinted_gt = tint_foreground(
        query_image.clone(),
        gt.cpu().bool(),
        tint=[-0.1, -0.1, 0.2],
        tint_border=[-0.3, -0.3, 0.5],
    )
    tinted_both = tint_foreground(tinted_gt.clone(), segmentation.cpu().bool())
    st.subheader(
        "Query with segmentation overlay (red tint) and gt overlay (blue tint)"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.subheader("Segmentation overlay")
    col1.image(tensor_to_pil(tinted_seg), use_container_width=True)
    col2.subheader("GT overlay")
    col2.image(tensor_to_pil(tinted_gt), use_container_width=True)
    col3.subheader("Both overlays")
    col3.image(tensor_to_pil(tinted_both), use_container_width=True)
    
    col4.subheader("Logits heatmap")
    # Also show logits heatmap
    col4.image(
        tensor_to_heatmap(logits[0, 1].cpu()),
        use_container_width=True,
    )

    with st.expander("Segmentation and GT masks"):
        empty_image = torch.zeros(3, segmentation.shape[0], segmentation.shape[1])
        tinted_seg_mask = tint_foreground(
            empty_image.clone().cpu(),
            segmentation.cpu().bool(),
            tint=(0.7, 0, 0),
            tint_border=(0.9, 0, 0),
        )
        tinted_gt_mask = tint_foreground(
            empty_image.clone().cpu(),
            gt.cpu().bool(),
            tint=(0, 0, 0.7),
            tint_border=(0, 0, 0.9),
        )
        col1, col2 = st.columns(2)
        col1.subheader("Segmentation mask")
        col1.image(tensor_to_pil(tinted_seg_mask), use_container_width=True)
        col2.subheader("GT mask")
        col2.image(tensor_to_pil(tinted_gt_mask), use_container_width=True)



def build_and_run_explainer(parameters, model, input_dict, device):
    """
    Build explainer via project utility if present; otherwise attempt a simple saliency baseline.
    Returns an explanation tensor scaled to [0,1] in HxW for display.
    """
    name = "signed_affinity"
    explainer = build_explainer(name=name, model=model, params={}, device=device)
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
    ).argmax(dim=1)
    explanation_mask = (
        F.one_hot(logits, num_classes=n_ways + 1).permute(0, 3, 1, 2)[0].bool()[1]
    )

    model_expl = explainer.explain(
        input_dict=input_dict,
        explanation_mask=explanation_mask,
    )[0]

    return model_expl, explanation_mask


def main():
    st.set_page_config(page_title="Qualit-Eff Streamlit", layout="wide")
    st.title("Qualit-Eff: Prediction & Explanation Demo")

    if _imports_error is not None:
        error_box(
            "Project-specific modules are missing. Install the 'affex' project and dependencies, then restart.",
            _imports_error,
        )
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        dataset = st.text_input(
            "Dataset folder name",
            value="coco",
            help="Used to locate parameters/<dataset>/cut_iauc_miou.yaml",
        )
        config = st.text_input(
            "Config code",
            value="N1K5",
            help="Used with grid/experiment utilities if your project expects it.",
        )
        model_name = st.text_input(
            "Model name", value="dmtnet", help="Name of the preconfigured model to use."
        )
        # Offer CPU plus every available CUDA device (cuda:0, cuda:1, ...)
        cuda_count = torch.cuda.device_count()
        device_options = (
            ["cpu"] + [f"cuda:{i}" for i in range(cuda_count)]
            if cuda_count > 0
            else ["cpu"]
        )
        default_idx = 1 if cuda_count > 0 else 0
        device = st.selectbox(
            "Device",
            options=device_options,
            index=default_idx,
            help="Choose CPU or a specific CUDA device",
        )
        sample_split = st.selectbox("Split", options=["val", "test", "train"], index=0)
        sample_index = st.number_input("Sample index", min_value=0, value=0, step=1)
        favorite_class = st.text_input(
            "Favorite class",
            value="",
            help="Choose a batched item containing this class if possible",
        )
        run_col, clear_col = st.columns(2)
        with clear_col:
            if st.button("Clear GPU cache"):
                st.session_state.pop("episode", None)
                st.session_state.pop("result", None)
                st.session_state.pop("explanation", None)
                st.session_state.pop("explanation_mask", None)
                st.session_state.pop("metrics", None)
                if device != "cpu":
                    torch.cuda.empty_cache()
        with run_col:
            show_batch = st.button("Show Batch")

    # Parameter file path (same as notebook used)
    param_relpath = f"parameters/{dataset}/cut_iauc_miou.yaml"

    try:
        parameters = load_parameters(dataset, param_relpath)
    except Exception as e:
        error_box(f"Failed to load parameters from '{param_relpath}'.", e)
        st.stop()
    runs_parameters = create_experiment(parameters)
    parameters = runs_parameters[0]  # Take the first set of parameters

    st.json(parameters, expanded=False)

    try:
        model, loader = build_model_and_data(
            parameters, model=model_name, device=device
        )
        loader = next(iter(loader.values()))
    except Exception as e:
        error_box("Failed to build model and dataloaders.", e)
        st.stop()

    if loader is None:
        error_box(f"No dataloader for split '{sample_split}'.")
        st.stop()

    # Allow user to pick a deterministic item from the loader by index
    # We iterate until we reach the requested index (streaming from loader).
    if show_batch:
        with st.spinner("Getting batch from dataloader..."):
            try:
                if favorite_class == "":
                    for i, batch in enumerate(loader):
                        if i == int(sample_index):
                            chosen = batch

                            chosen, dataset_name = chosen

                            substitutor = Substitutor(substitute=True)
                            substitutor.reset(batch=chosen)
                            chosen, gt = next(substitutor)
                            chosen = to_device(chosen, device)
                            gt = gt.to(device)
                            break
                else:
                    i = 0
                    with st.spinner("Searching for favorite class in batches..."):
                        for batch in loader:
                            categories = loader.dataset.datasets[batch[1][0]].categories
                            classes = batch[0][0][BatchKeys.CLASSES][0][0]
                            class_names = [categories[c]["name"] for c in classes]
                            if favorite_class in class_names:
                                if i == int(sample_index):
                                    st.write(
                                        f"Found favorite class '{favorite_class}' in batch index {i} with classes: {class_names}"
                                    )
                                    chosen = batch

                                    chosen, dataset_name = chosen

                                    substitutor = Substitutor(substitute=True)
                                    substitutor.reset(batch=chosen)
                                    chosen, gt = next(substitutor)
                                    chosen = to_device(chosen, device)
                                    break
                                i += 1
            except Exception as e:
                error_box("Could not iterate the dataloader.", e)
                st.stop()
    
        st.session_state["episode"] = {
            "batch": chosen,
            "gt": gt,
            "dataset_name": dataset_name,
        }
        st.session_state.pop("result", None)
        st.session_state.pop("explanation", None)
        st.session_state.pop("explanation_mask", None)
        st.session_state.pop("metrics", None)
        
    if "episode" in st.session_state:
        episode = st.session_state["episode"]
        chosen = episode["batch"]
        gt = episode["gt"]
        dataset_name = episode["dataset_name"]
        visualize_episode(chosen)
        
        if st.button("Run"):
            with st.spinner("Running model on selected sample..."):
                try:
                    result, logits, pred_seg = run_model_on_batch(model, chosen, gt)
                except Exception as e:
                    error_box("Model forward pass failed.", e)
                    st.stop()
                    
                st.session_state["result"] = {
                        "logits": logits,
                        "pred_seg": pred_seg,
                }
                
                    
            with st.spinner("Running explainer on selected sample..."):
                exp, explanation_mask = build_and_run_explainer(
                    parameters, model, chosen, device
                )

            st.session_state["explanation"] = exp
            st.session_state["explanation_mask"] = explanation_mask
            st.session_state.pop("metrics", None)  # Clear previous metrics if any
        
    if "result" in st.session_state:
        batch_result = st.session_state["result"]
        logits = batch_result["logits"]
        pred_seg = batch_result["pred_seg"]

        # Visuals
        try:
            rgb_images = unnormalize(chosen[BatchKeys.IMAGES])
            show_overlay(rgb_images.cpu(), logits.cpu(), pred_seg.cpu(), gt[0].cpu().bool())
        except Exception as e:
            error_box("Visualization failed.", e)

        # Explanation
        st.header("Explanation")
        try:
            exp = st.session_state["explanation"]
            explanation_mask = st.session_state["explanation_mask"]
            # Normalize and show
            exp = min_max_scale(exp)
            cols = st.columns(exp.shape[1])
            for i in range(exp.shape[1]):
                with cols[i]:
                    st.subheader(f"Explanation channel {i}")
                    st.image(
                        tensor_to_heatmap(exp[0, i]),
                        caption="Explanation (normalized)",
                        use_container_width=True,
                    )

            st.session_state["explanation"] = exp
            st.session_state["explanation_mask"] = explanation_mask
        except Exception as e:
            error_box("Failed to compute/show explanation.", e)

        st.success("Done.")

        st.header("IAUC and DAUC Calculation")
        n_steps = st.number_input("Number of steps", min_value=1, value=75, step=1)
        measure = st.selectbox("Measure", options=["miou", "logits"], index=0)
        n_mid_statuses = st.number_input(
            "Number of mid statuses", min_value=1, value=30, step=1
        )

        if st.button("Calculate IAUC and DAUC"):

            metrics = {}
            metrics["iauc"] = FSSCausalMetric(
                model=model,
                mode="ins",
                n_steps=n_steps,
                measure=measure,
                n_mid_statuses=n_mid_statuses,
            )
            metrics["dauc"] = FSSCausalMetric(
                model=model,
                mode="del",
                n_steps=n_steps,
                measure=measure,
                n_mid_statuses=n_mid_statuses,
            )

            if (
                "explanation" not in st.session_state
                or "explanation_mask" not in st.session_state
            ):
                error_box("No explanation available. Run explanation first.")
                st.stop()
            explanation_mask = st.session_state["explanation_mask"]
            explanation = st.session_state["explanation"]
            st.session_state["metrics"] = {}

            for name, metric in metrics.items():
                progress = st.progress(0, "Calculating " + name.upper())
                for _, i, _ in metric.evaluate_interactive(
                    chosen, explanation, explanation_mask, gt=gt
                ):
                    progress.progress(
                        min(i / (metric.computed_n_steps + 1), 1.0),
                        f"Calculating {name.upper()}, step {i}/{metric.computed_n_steps}",
                    )
                st.session_state["metrics"][name] = {
                    "auc": metric.xauc,
                    "scores": metric.scores,
                    "mid_statuses": metric.mid_statuses,
                }

        if "metrics" in st.session_state:
            st.subheader("IAUC-DAUC Results")
            cols = st.columns(2)
            for name, result in st.session_state["metrics"].items():
                with cols[0 if name == "iauc" else 1]:
                    st.write(f"**{name.upper()} AUC:** {result['auc']:.4f}")
                    scores = result["scores"]
                    # Plot scores as AUC curve
                    # Plot scores
                    batch_element = 0  # Change this to plot a different batch element
                    x = np.linspace(0, 1, scores.shape[0])
                    y = scores[:, batch_element]

                    fig, ax = plt.subplots(figsize=(45/5, 15/5), dpi=150)

                    ax.plot(
                        x,
                        y,
                        linewidth=2,
                    )
                    ax.fill_between(
                        x,
                        0,
                        y,
                        alpha=0.3,
                    )

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.05)

                    ax.set_xlabel(f"Pixels {'inserted' if name == 'iauc' else 'deleted'}")
                    ax.set_ylabel(f"Score ({'mIoU' if measure == 'miou' else 'Logits'})")
                    ax.set_title(f"{name.upper()} Curve")
                    
                    # Write AUC value on the plot
                    ax.text(0.8, 0.8, f"AUC = {(result['auc']*100):.2f}", fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.6))

                    # Cleaner grid
                    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

                    # Light frame tweaks
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)

                    st.pyplot(fig)
                    plt.close(fig)

                    # Provide download buttons for the plot and the numeric scores

                    # Save SVG in addition to PNG
                    svg_buf = BytesIO()
                    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
                    svg_buf.seek(0)

                    st.download_button(
                        label="Download plot (SVG)",
                        data=svg_buf.getvalue(),
                        file_name=f"{name}_curve.svg",
                        mime="image/svg+xml",
                    )

                
            for name, result in st.session_state["metrics"].items():              
                with st.expander(f"{name.upper()} Mid Statuses Visualization"):
                    mid_statuses = result["mid_statuses"]
                    for mid_status in mid_statuses:
                        j, mid_images, mid_masks, probs, top_pred = mid_status
                        st.subheader(f"{name.upper()} Mid Status {j}")
                        
                        assert mid_images.shape[0] == 1, "Batch size > 1 not supported for mid status visualization"
                        n_shots = mid_images.shape[1] - 1
                        
                        shot_cols = st.columns(n_shots)
                        for shot_idx in range(n_shots):
                            with shot_cols[shot_idx]:
                                st.subheader(f"Shot {shot_idx}")
                                masked_shot = tint_foreground(
                                    unnormalize(mid_images[0, shot_idx+1])[0].cpu(),
                                    mid_masks[0, shot_idx, 1].cpu().bool(),
                                )
                                st.image(
                                    tensor_to_pil(masked_shot),
                                    use_container_width=True,
                                )
                                
                        prob_col, seg_col = st.columns(2)
                        
                        with prob_col:
                            st.subheader(f"{name.upper()} Mid Status {j} - Probs")
                            st.image(
                                tensor_to_heatmap(probs[0, 1].cpu()),
                                use_container_width=True,
                            )
                        with seg_col:
                            mid_pred_seg = probs.argmax(dim=1)[0]  # (B, n_ways, H, W)
                            st.subheader(f"{name.upper()} Mid Status {j} - Segmentation")
                            seg_rgb = tint_foreground(
                                unnormalize(mid_images[0, 0])[0].cpu(),
                                mid_pred_seg.cpu().bool(),
                            )
                            st.image(
                                tensor_to_pil(seg_rgb),
                                use_container_width=True,
                                )


def launch():
    subprocess.run(["streamlit", "run", "app.py"], check=True)


if __name__ == "__main__":
    main()
