# app.py
# Streamlit app to replicate the notebook: load model & dataset, pick a sample, run prediction and explanation,
# and visualize with lovely-tensors.

import copy
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
    "dataset": {
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
        },
    }
}

config_dataloader = {
    "num_workers": 0,
    "batch_size": 1,
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
        with st.expander("Traceback / details"):
            st.code("".join(traceback.format_exception(None, exc, exc.__traceback__)) if exc else msg)

def tensor_to_pil(tensor):
    tensor = (tensor * 255).permute(1, 2, 0).detach().type(torch.uint8).cpu()
    return Image.fromarray(tensor.numpy())

def tensor_to_heatmap(tensor):
    array = tensor.detach().cpu().numpy()
    cmap = plt.get_cmap("jet")
    rgba = cmap(array)
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(heat_rgb)

@st.cache_resource(show_spinner=True)
def build_model_and_data(parameters: dict, model: str, device: str):
    model_obj, image_size = build_model_preconfigured(model_name=model)
    model_obj.eval()
    model_obj = model_obj.to(device)

    if "dataset" in parameters:
        if "datasets" in parameters["dataset"]:
            for k in parameters["dataset"]["datasets"]:
                parameters["dataset"]["datasets"][k]["image_size"] = image_size
        if "preprocess" not in parameters["dataset"]:
            parameters["dataset"]["preprocess"] = {}
        parameters["dataset"]["preprocess"]["image_size"] = image_size

    _, val_loader, _ = get_dataloaders(
        copy.deepcopy(parameters["dataset"]),
        copy.deepcopy(parameters["dataloader"]),
        num_processes=1,
    )
    return model_obj, val_loader

def run_model_on_batch(model, batch, gt):
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
    rgb_images = unnormalize(input_dict[BatchKeys.IMAGES])
    support_images = rgb_images[0, 1:].clone()
    support_masks = input_dict[BatchKeys.PROMPT_MASKS][0, 0:, 1].clone()
    n_shots = support_masks.shape[0]

    st.markdown('<p class="sub-header">1. The Few-Shot Episode</p>', unsafe_allow_html=True)
    st.caption("The model is presented with a Query Image and a Support Set (Images + Masks) to learn the class on-the-fly.")

    col_query, col_support = st.columns([1, 2])
    
    with col_query:
        st.markdown("**Query Image** (To Segment)")
        st.image(tensor_to_pil(rgb_images[0, 0]), use_container_width=True)

    with col_support:
        st.markdown(f"**Support Set** ({n_shots}-shot)")
        cols = st.columns(n_shots)
        for i in range(n_shots):
            with cols[i]:
                tinted_support = tint_foreground(
                    support_images[i].cpu(), support_masks[i].cpu().bool()
                )
                st.image(tensor_to_pil(tinted_support), caption=f"Shot {i}", use_container_width=True)

def show_overlay(rgb_images, logits, seg, gt):
    query_image = rgb_images[0, 0].clone()
    
    # Create overlays
    tinted_seg = tint_foreground(query_image.clone(), seg[0].cpu().bool(), tint=(0.7, 0, 0))
    tinted_gt = tint_foreground(query_image.clone(), gt.cpu().bool(), tint=(0, 0, 0.7))
    tinted_both = tint_foreground(tinted_gt.clone(), seg[0].cpu().bool(), tint=(0.7, 0, 0))
    
    st.markdown('<p class="sub-header">2. Segmentation Result</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Comparison", "Detailed Masks"])
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(tensor_to_pil(tinted_seg), caption="Prediction (Red)", use_container_width=True)
        with col2:
            st.image(tensor_to_pil(tinted_gt), caption="Ground Truth (Blue)", use_container_width=True)
        with col3:
            st.image(tensor_to_pil(tinted_both), caption="Overlay (Both)", use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        col1.image(tensor_to_heatmap(logits[0, 1].cpu()), caption="Logits Heatmap", use_container_width=True)
        col2.info("The logits heatmap represents the raw confidence of the model before argmax.")

def build_and_run_explainer(parameters, model, input_dict, device):
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
    )
    pred_seg = logits.argmax(dim=1)
    
    # FIX: Pass pred_seg (LongTensor) to one_hot, NOT logits (FloatTensor)
    explanation_mask = (
        F.one_hot(pred_seg, num_classes=n_ways + 1).permute(0, 3, 1, 2)[0].bool()[1]
    )
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
        f'# <img src="data:image/svg+xml;base64,{svg_base64}" alt="icon" width="64" style="vertical-align: middle;"/> [AffinityExplainer](https://pasqualedem.github.io/AffinityExplainer/)',
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
            dataset = st.selectbox("Dataset", ["coco", "pascal"], index=0)
            num_shots = st.number_input("N-Shots", min_value=1, value=2)
            
            if not os.path.exists("data/coco/annotations/instances_val2014.json"):
                if st.button("📥 Download COCO Metadata"):
                    with st.spinner("Downloading..."):
                        subprocess.run(["bash", "scripts/download_coco_jsononly.sh"], check=True)

        with st.expander("Model Settings", expanded=True):
            model_name = st.selectbox("Model Architecture", ["dcama", "dmtnet"])
            
            if not os.path.exists(f"checkpoints/{model_name}/"):
                if st.button(f"📥 Download {model_name.upper()} Checkpoints"):
                    with st.spinner("Downloading..."):
                        download_model(model_name)
            
            cuda_count = torch.cuda.device_count()
            device_options = ["cpu"] + [f"cuda:{i}" for i in range(cuda_count)] if cuda_count > 0 else ["cpu"]
            device = st.selectbox("Compute Device", device_options, index=1 if cuda_count > 0 else 0)

        with st.expander("Sample Selection"):
            sample_index = st.number_input("Sample Index", min_value=0, value=0)
            favorite_class = st.text_input("Filter by Class Name", value="", placeholder="e.g. person")
        
        st.divider()
        col_load, col_clear = st.columns(2)
        load_batch = col_load.button("🔄 Load Batch", use_container_width=True)
        if col_clear.button("🗑️ Clear", use_container_width=True):
            st.session_state.clear()
            if "cuda" in device: torch.cuda.empty_cache()
            st.rerun()

    # --- Main Logic ---
    
    # Setup Parameters
    parameters = {
        "dataset": config_coco if dataset == "coco" else config_pascal,
        "dataloader": config_dataloader,
        "model": {"name": model_name},
        "explainer": {"name": "signed_affinity"},
    }
    for k in parameters["dataset"]["datasets"]:
        parameters["dataset"]["datasets"][k]["n_shots"] = num_shots

    # 1. Load Model
    try:
        model, loader = build_model_and_data(parameters, model=model_name, device=device)
        loader_iter = next(iter(loader.values()))
    except Exception as e:
        error_box("Failed to build model.", e)
        st.stop()

    # 2. Get Batch
    if load_batch:
        with st.spinner("Fetching batch data..."):
            chosen = None
            try:
                if favorite_class == "":
                    for i, batch in enumerate(loader_iter):
                        if i == int(sample_index):
                            chosen = batch
                            break
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
                    chosen, dataset_name = chosen
                    substitutor = Substitutor(substitute=True)
                    substitutor.reset(batch=chosen)
                    chosen, gt = next(substitutor)
                    chosen = to_device(chosen, device)
                    gt = gt.to(device)
                    
                    st.session_state["episode"] = {
                        "batch": chosen,
                        "gt": gt,
                        "dataset_name": dataset_name,
                    }
                    # Reset downstream results
                    for key in ["result", "explanation", "explanation_mask", "metrics"]:
                        st.session_state.pop(key, None)
                else:
                    st.warning("Sample not found (check index or class name).")

            except Exception as e:
                error_box("Dataloader error.", e)
                st.stop()

    # 3. Display & Run
    if "episode" in st.session_state:
        episode = st.session_state["episode"]
        chosen = episode["batch"]
        gt = episode["gt"]
        
        visualize_episode_header(chosen)

        if st.button("🚀 Run Inference & Explanation", type="primary", use_container_width=True):
            with st.status("Processing...", expanded=True) as status:
                st.write("Running Model Forward Pass...")
                try:
                    result, logits, pred_seg = run_model_on_batch(model, chosen, gt)
                    st.write("Computing Affinity Explanation...")
                    exp, explanation_mask = build_and_run_explainer(parameters, model, chosen, device)
                except Exception as e:
                    error_box("Model Error", e)
                    st.stop()

                st.session_state["result"] = {"logits": logits, "pred_seg": pred_seg}
                st.session_state["explanation"] = exp
                st.session_state["explanation_mask"] = explanation_mask
                st.session_state.pop("metrics", None)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

    # 4. Results & Explainability
    if "result" in st.session_state:
        batch_result = st.session_state["result"]
        rgb_images = unnormalize(chosen[BatchKeys.IMAGES])
        
        # Segmentation Results
        show_overlay(rgb_images.cpu(), batch_result["logits"].cpu(), batch_result["pred_seg"].cpu(), gt[0].cpu().bool())

        # Explanation Heatmaps
        st.markdown('<p class="sub-header">3. Affinity Explanation</p>', unsafe_allow_html=True)
        st.markdown("""
        The explanation highlights regions in the **Support Set** that were most influential for the query segmentation.
        """)
        
        exp = st.session_state["explanation"]
        exp = min_max_scale(exp) # Normalize for vis
        
        cols = st.columns(exp.shape[1])
        for i in range(exp.shape[1]):
            with cols[i]:
                st.image(tensor_to_heatmap(exp[0, i]), caption=f"Support Attribution {i}", use_container_width=True)

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
    cmd = ["streamlit", "run", "affex/app.py"]
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