# AffinityExplainer

![imgs/FSSAffex.svg](imgs/FSSAffex.svg)

Repository for the paper "Matching-Based Few-Shot Semantic Segmentation Models Are Interpretable by Design"

AffinityExplainer is a framework to interpret matching-based few-shot semantic segmentation models. It provides a tool to extract and visualize the contribution of each support pixel to the final prediction.

## One-Line Demo!

To run a quick demo of AffinityExplainer, execute the following command:

```bash
uvx --from https://github.com/pasqualedem/AffinityExplainer app
```
> **💡 You just need [uv](https://docs.astral.sh/uv/) to run this command **

## Install

Use [uv](https://docs.astral.sh/uv/) to install the required packages:

```bash
uv sync
source .venv/bin/activate
```

## Download Datasets

To download the PASCAL VOC12 and COCO datasets, run the following scripts:

```bash
bash scripts/download_pascal.sh
bash scripts/download_coco.sh
```

## Reproduce Results

Refer to the scripts in the `scripts/` directory to reproduce the results presented in the paper. Each script corresponds to a specific experiment or ablation study.

## Some Examples

![imgs/FSSAffex_examples-DCAMA.svg](imgs/FSSAffex_examples-DCAMA.svg)