import lovely_tensors as lt
from matplotlib import cm
import streamlit as st

import pickle
from einops import rearrange, repeat
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TvT
import torch.nn as nn
import numpy as np
from PIL import Image
from torchmetrics import F1Score, MetricCollection
from torchvision.transforms.functional import resize
from sklearn.decomposition import PCA
from streamlit_image_coordinates import streamlit_image_coordinates

import numpy as np
import torch

from fssweed.utils.segmentation import create_rgb_segmentation, unnormalize
from fssweed.utils.utils import (
    ResultDict,
    StrEnum,
    torch_dict_load,
    torch_dict_save,
    to_device,
)
from fssweed.models import SUPPORTED_MODELS, build_model_preconfigured
from fssweed.data import get_dataloaders, get_preprocessing, get_testloaders
from fssweed.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
    min_max_scale,
)
from fssweed.substitution import Substitutor
import matplotlib.pyplot as plt
from tqdm import tqdm


lt.monkey_patch()

PROMPT_IMAGES = [
    # 'frame0009_2.png',
    'frame0021_2.png',
    "frame0033_3.png",
    'frame0034_1.png',
    'frame0048_0.png',
]

PASCAL_NAME = "val_pascal5i"
PASCAL_PARAMS = {
    "name": "pascal",
    "data_dir": "data/pascal",
    "split": "val",
    "val_fold_idx": 3,
    "n_folds": 4,
    "n_shots": 2,
    "n_ways": 1,
    "do_subsample": False,
    "val_num_samples": 100,
    "ignore_borders": True,
}


parameters = {
    "dataloader": {
        "num_workers": 0,
        "possible_batch_example_nums": [[1, 2, 4]],
        "val_possible_batch_example_nums": [[1, 1]],
        "prompt_types": ["mask"],
        "prompt_choice_level": ["episode"],
        "val_prompt_types": ["mask"],
        },
    "dataset": {
        "preprocess": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "image_size": 384,
        },
        "datasets": {
            PASCAL_NAME: PASCAL_PARAMS,
        },
    },
    "model": {
        "name": "dcama",
        "backbone": "swin",
        # "backbone_checkpoint": "checkpoints/swin_base_patch4_window12_384.pth",
        "model_checkpoint": "checkpoints/swin_fold0_pascal_modcross_soft.pt",
        # 'model_checkpoint': "checkpoints/f4z7ghu7.pt",
        "concat_support": True,
        "image_size": 384,
    }
}


@st.cache_data
def get_data(n_ways, n_shots, image_size):
    parameters["dataset"]["datasets"][PASCAL_NAME]["n_ways"] = n_ways
    parameters["dataset"]["datasets"][PASCAL_NAME]["n_shots"] = n_shots
    parameters["dataset"]["datasets"][PASCAL_NAME]["image_size"] = image_size
    parameters["dataset"]["preprocess"]["image_size"] = image_size
    
    _, val, _ = get_dataloaders(
        parameters["dataset"],
        parameters["dataloader"],
        num_processes=1,
    )
    return val[PASCAL_NAME]
    
    
@st.cache_data
def get_model(model, use_pe, n_shots, device):
    model, image_size = build_model_preconfigured(model_name=model, use_pe=use_pe, n_shots=n_shots)
    model.to(device)
    model.eval()
    return model, image_size


def preprocess_attentions(attentions):
    model_name = st.session_state.get("model_name", None)
    if model_name == "dcama":
        processed_attentions = []
        for class_attns in attentions:
            processed_level_attentions = []
            for level_attn in class_attns:
                level_attn = level_attn.mean(dim=1)
                processed_level_attentions.append(level_attn)
            processed_attentions.append(processed_level_attentions)
        return processed_attentions
    elif model_name == "dmtnet":
        processed_attentions = []
        for class_attns in attentions:
            num_shots = len(class_attns)
            level_attentions = [
                torch.cat(
                    [
                        class_attns[i][0][j] + class_attns[i][1][j]
                        for i in range(num_shots)
                    ],
                    dim=-2
                )
                for j in range(len(class_attns[0][0]))
            ]
            
            processed_level_attentions = []
            for level_attn in level_attentions:
                level_attn = rearrange(level_attn, "b c h1 w1 nh2 w2 -> b c (h1 w1) (nh2 w2)")
                for i in range(level_attn.shape[1]):
                    processed_level_attentions.append(level_attn[:, i])
            processed_attentions.append(processed_level_attentions)
        return processed_attentions
                

def image_blend(image, heatmap):
    alpha = 0.8
    rgb_image = unnormalize(image)[0]
    blended_heatmap = heatmap * rgb_image
    blended = blended_heatmap * alpha + rgb_image * (1 - alpha)
    blended = (blended * 255).type(torch.uint8)
    return blended


def plot_contributions(contributions, support_images, support_mask):
        col1, col2, col3 = st.columns(3)
        
        blended_weighted_contrib = image_blend(support_images, contributions.clamp(0, 1))
        
        col1.write("Blended contribution")
        col1.write(blended_weighted_contrib.rgb.fig)
        col2.write("Contribution")
        col2.write(contributions.chans(cmap="seismic").fig)
        
        sign_weighted_contrib = (contributions * support_mask)
        
        col3.write("Sign weighted contribution")
        col3.write(sign_weighted_contrib.chans(cmap="seismic").fig)
        
        pos_contib = image_blend(support_images, sign_weighted_contrib.clamp(0, 1))
        neg_contib = image_blend(support_images, (-sign_weighted_contrib).clamp(0, 1))
        
        col1, col2, col3 = st.columns(3)
        col1.write("Positive contribution")
        col1.write(sign_weighted_contrib.clamp(0, 1).chans(cmap="seismic").fig)
        col2.write("Negative contribution")
        col2.write((-sign_weighted_contrib).clamp(0, 1).chans(cmap="seismic").fig)
        col3.write(f"Support Image")
        col3.write(unnormalize(support_images).rgb.fig)
        
        col1, col2, col3 = st.columns(3)
        col1.write("Positive Blended contribution")
        col1.write(pos_contib.rgb.fig)
        col2.write("Negative Blended contribution")
        col2.write(neg_contib.rgb.fig)
        col3.write("Support Mask")
        col3.write(support_mask.chans(cmap="seismic").fig)


def explain(model, input_dict, result, num_classes):
    attns = preprocess_attentions(result[ResultDict.ATTENTIONS])
    masks = input_dict[BatchKeys.PROMPT_MASKS]
    flag_examples = input_dict[BatchKeys.FLAG_EXAMPLES]
    target_image = input_dict[BatchKeys.IMAGES][0, 0]
    support_images = input_dict[BatchKeys.IMAGES][0, 1:]
    support_images = rearrange(support_images, "n c h w -> c h (n w)")
    
    target_shape = input_dict[BatchKeys.IMAGES][:, 0].shape[2:]
    st.write(target_shape)
    
    st.write("### Click on the image to select a pixel")
    coords = streamlit_image_coordinates(
        (unnormalize(target_image.unsqueeze(0))[0]*255).numpy().transpose(1, 2, 0).astype(np.uint8),
    )
    
    if coords:
        st.session_state["coords"] = coords
    if st.session_state.get("coords") is not None:
        coords = st.session_state["coords"]
        selected_x = coords["x"]
        selected_y = coords["y"]
    else:
        return
    
    st.write("Coordinates", (coords["x"], coords["y"]))
    for chosen_class in range(num_classes):
        st.write(f"### Class {chosen_class+1} interpretation")
        class_attns = attns[chosen_class]
        class_examples = flag_examples[:, :, chosen_class + 1]
        mask = masks[:, :, chosen_class+1, ::][class_examples]
        class_shots = mask.shape[0]
        
        support_mask = resize(mask, target_shape, interpolation=TvT.InterpolationMode.NEAREST).float()
        support_mask = rearrange(support_mask, "n h w -> h (n w)")
        support_mask = 2*support_mask - 1
        # support_mask = TvT.gaussian_blur(support_mask, (9,9), sigma=5)
        st.write("Support mask", support_mask)

        level_contributions = []
        for level_attn in class_attns:

            hw = level_attn.shape[-2]
            h = w = int(hw ** 0.5)
            level_attn = F.softmax(level_attn, dim=-1)
            level_attn = rearrange(level_attn, "b (h1 w1) (n h2 w2) -> (b h2 w2 n) h1 w1", n=class_shots, h1=h, w1=w, h2=h, w2=w)
            level_attn = resize(level_attn, target_shape)
            level_attn = level_attn[:, selected_y, selected_x]
            level_attn = rearrange(level_attn, "(b h2 w2 n) -> (b n) h2 w2", h2=h, w2=w, n=class_shots)
            resized_level_attn = resize(level_attn, target_shape, interpolation=TvT.InterpolationMode.BILINEAR, antialias=False)
            resized_level_attn = rearrange(resized_level_attn, "(b n) h w -> b h (n w)", n=class_shots)
            level_contributions.append(resized_level_attn)

        contrib_seq = torch.stack(level_contributions)
        contrib = contrib_seq.mean(dim=0)
        
        with st.spinner("Doing Feature Ablation..."):
            cmask_contrib = model.feature_ablation(result, chosen_class, selected_x, selected_y, n_shots=class_shots, image_size=target_shape)
            if cmask_contrib is None:
                cmask_contrib = torch.full((contrib_seq.shape[0],), 1 / contrib_seq.shape[0])
                
        st.write(contrib_seq)
        with st.expander("Full masks"):
            st.write(min_max_scale(contrib_seq).chans(cmap="seismic").fig)
        
        st.write("#### Mean Based Contribution")        
        contrib = min_max_scale(contrib)
        plot_contributions(contrib, support_images, support_mask)
        # col1.write((contrib*support_mask).chans(cmap="seismic").fig)
        
        st.write(f"#### Weighted contribution")
        cmask_contrib = rearrange(cmask_contrib, "c -> c 1 1 1")
        weighted_contrib = min_max_scale((contrib_seq * cmask_contrib).sum(dim=0))
        plot_contributions(weighted_contrib, support_images, support_mask)
        
        
def feature_map_pca_heatmap(feature_map):
    """
    Given a feature map of shape (D, H, W), performs PCA along the feature dimension
    and returns a heatmap based on the first principal component.
    
    Args:
        feature_map (torch.Tensor): Input tensor of shape (D, H, W).
    
    Returns:
        heatmap (torch.Tensor): Heatmap of shape (H, W) based on the first principal component.
    """
    # Check the input dimensions
    D, H, W = feature_map.shape
    feature_map_reshaped = feature_map.view(D, -1).T  # Reshape to (H*W, D)

    # Convert to numpy for PCA
    feature_map_np = feature_map_reshaped.cpu().numpy()
    
    # Perform PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(feature_map_np)
    
    # Reshape the result back to (H, W) and convert it to a tensor
    heatmap = torch.tensor(principal_component).view(H, W)
    
    # Normalize the heatmap for better visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap
        
def attention_summary(result, masks, flag_examples):
    attns_class = result[ResultDict.ATTENTIONS]
    pre_mix = result[ResultDict.PRE_MIX]
    mix = result[ResultDict.MIX]
    mix1 = result[ResultDict.MIX_1]
    mix2 = result[ResultDict.MIX_2]
    sf1  = result[ResultDict.SUPPORT_FEAT_1]
    sf0  = result[ResultDict.SUPPORT_FEAT_0]
    qf0  = result[ResultDict.QUERY_FEAT_0]
    qf1  = result[ResultDict.QUERY_FEAT_1]
    coarse_masks = result[ResultDict.COARSE_MASKS]
    
    masks = masks[:, :, 1:, ::]
    st.write("## Model Summary")
    target_size = 48
    for j, attns in enumerate(attns_class):
        attns = [
            attn.mean(dim=1) for attn in attns
        ]
        class_examples = flag_examples[:, :, j + 1]
        mask = masks[:, :, j, ::][class_examples]
        outs = []
        for attn in attns:
            hw = attn.shape[-1]
            h = w = int(hw ** 0.5)
            # resize mask to attn
            mask = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
            mask = rearrange(mask, "1 h w -> 1 1 (h w)")
            attn = attn * mask
            attn = attn.sum(dim=-1)
            # attn = torch.matmul(attn, mask)
            attn = rearrange(attn, "1 (h w) -> 1 h w", h=h, w=w)
            attn = resize(attn, (target_size, target_size))
            outs.append(attn)
        out = torch.cat(outs).mean(dim=0)
        out = (out - out.min()) / (out.max() - out.min())
        
        cols = st.columns(4)
        with cols[0]:
            st.write("### Coarse Mask 1")
            st.write(coarse_masks[j][0][0])
            coarse1 = coarse_masks[j][0][0].mean(dim=0)
            st.write(coarse1.chans(scale=4).fig)
        with cols[1]:
            st.write("### Coarse Mask 2")
            st.write(coarse_masks[j][1][0])
            coarse2 = coarse_masks[j][1][0].mean(dim=0)
            st.write(coarse2.chans(scale=4).fig)
        with cols[2]:
            st.write("### Coarse Mask 3")
            st.write(coarse_masks[j][2][0])
            coarse3 = coarse_masks[j][2][0].mean(dim=0)
            st.write(coarse3.chans(scale=4).fig)
        with cols[3]:
            st.write("### Coarse Mean")
            coarse3 = coarse_masks[j][2]
            coarse2 = F.interpolate(coarse_masks[j][1], coarse3.size()[-2:], mode='bilinear', align_corners=True)
            coarse1 = F.interpolate(coarse_masks[j][2], coarse3.size()[-2:], mode='bilinear', align_corners=True)
            coarse = torch.cat([coarse1, coarse2, coarse3], dim=1)
            coarse_mean = coarse.mean(dim=1)
            st.write(coarse_mean)
            st.write(coarse_mean.chans(scale=4).fig)
            
        with st.expander("Full Coarse Maps"):
            cols = st.columns(4)
            for i, level_coarse in enumerate(coarse[0]):
                with cols[i % 4]:
                    st.write(f"Coarse Map {i}")
                    st.write(level_coarse.chans.fig)
        
        cols = st.columns(4)
        with cols[0]:
            st.write("### Pre-mix")
            st.write(pre_mix[j][0])
            pre_mix_pca = feature_map_pca_heatmap(pre_mix[j][0])
            st.write(pre_mix_pca.chans(scale=4).fig)
        with cols[1]:
            st.write("### Mix")
            st.write(mix[j][0])
            coarse1 = feature_map_pca_heatmap(mix[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[2]:
            st.write("### Mix Out 1")
            st.write(mix1[j][0])
            coarse1 = feature_map_pca_heatmap(mix1[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[3]:
            st.write("### Mix Out 2")
            st.write(mix2[j][0])
            coarse1 = feature_map_pca_heatmap(mix2[j][0])
            st.write(coarse1.chans(scale=4).fig)
                
        with st.expander("Query Features"):
            cols = st.columns(2)
            with cols[0]:
                st.write("### Query Feature 0")
                qf0_pca = feature_map_pca_heatmap(qf0[j][0])
                st.write(qf0_pca.chans(scale=4).fig)
            with cols[1]:
                st.write("### Query Feature 1")
                qf1_pca = feature_map_pca_heatmap(qf1[j][0])
                st.write(qf1_pca.chans(scale=4).fig)

        # with st.expander("Support Features"):
        #     cols = st.columns(2)
        #     with cols[0]:
        #         st.write("### Support Feature 0")
        #         sf0_pca = feature_map_pca_heatmap(sf0[j][0])
        #         st.write(sf0_pca.chans(scale=4).fig)
        #     with cols[1]:
        #         st.write("### Support Feature 1")
        #         sf1_pca = feature_map_pca_heatmap(sf1[j][0])
        #         st.write(sf1_pca.chans(scale=4).fig)
    
def main():
    with st.sidebar:
        st.write("### Parameters")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        use_pe = st.checkbox("Use PE", value=True)
        n_ways = st.number_input("N ways", value=1, min_value=1, max_value=10)
        n_shots = st.number_input("N shots", value=2, min_value=1, max_value=10)
        model = st.selectbox("Model", list(SUPPORTED_MODELS.keys()), index=0, key="model_name")
    
    model, image_size = get_model(model, use_pe, n_shots, device)
    data = get_data(n_shots=n_shots, n_ways=n_ways, image_size=image_size)
    
    if "iterator" not in st.session_state:
        st.session_state["iterator"] = iter(data)
    if "batch" not in st.session_state:
        st.session_state["batch"] = next(st.session_state["iterator"])
    if st.button("Next"):
        st.session_state["batch"] = next(st.session_state["iterator"])
        st.session_state.pop("result", None)
        st.session_state.pop("coords", None)
    batch = st.session_state["batch"]
    
    batch, dataset_name  = batch
    
    substitutor = Substitutor(substitute=False)
    substitutor.reset(batch=batch)
    batch = next(substitutor)
    input_dict, gt = batch
    
    col1, col2 = st.columns(2)
    col1.write("### Support images")
    col1.write(unnormalize(input_dict[BatchKeys.IMAGES][:, 1:]).rgb.fig)
    
    col2.write("### Support masks")
    col2.write(create_rgb_segmentation(input_dict[BatchKeys.PROMPT_MASKS][0], num_classes=3).rgb.fig)
    
    st.write("Query Image")
    st.write(unnormalize(input_dict[BatchKeys.IMAGES][:, 0]).rgb.fig)
    st.write(data.dataset.datasets[PASCAL_NAME].n_shots)
    
    input_dict = to_device(input_dict, device)
    num_classes = input_dict[BatchKeys.PROMPT_MASKS].shape[2] - 1
    gt = to_device(gt, device)
    st.write(gt)
    
    st.write(input_dict)
    
    if st.button("Predict"):
        with torch.no_grad():
            result = model(input_dict)
            st.write(result[ResultDict.LOGITS])
            
        st.session_state["result"] = result
        
    if "result" in st.session_state:
        result = st.session_state["result"]
        outputs = torch.argmax(result[ResultDict.LOGITS], dim=1)
        pred_col, gt_col = st.columns(2)
        
        if st.checkbox("Show attention summary"):
            attention_summary(result, input_dict[BatchKeys.PROMPT_MASKS], input_dict[BatchKeys.FLAG_EXAMPLES])
        
        pred_col.write("Predictions")
        pred_col.write(create_rgb_segmentation(outputs, num_classes=num_classes+1).rgb.fig)
        
        gt_col.write("Ground Truth")
        gt_col.write(create_rgb_segmentation(gt, num_classes=num_classes+1).rgb.fig)
        
        explain(model, input_dict, result, num_classes)
    
if __name__ == "__main__":
    main()