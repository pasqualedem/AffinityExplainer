from .affinity import AffinityExplainer
from .captum import CaptumExplainer
from .random import GaussianNoiseMask, RandomExplainer
from .saliency import SaliencyExplainer


from ..utils.segmentation import unnormalize


EXPLAINER_REGISTRY = {
    "integrated_gradients": CaptumExplainer,
    "saliency": CaptumExplainer,
    "gradcam": CaptumExplainer,
    "gradient_shap": CaptumExplainer,
    "deep_lift": CaptumExplainer,
    "affinity": AffinityExplainer,
    "random": RandomExplainer,
    "gaussian_noise": GaussianNoiseMask,
    "lime": CaptumExplainer,
    "guided_ig": SaliencyExplainer,
    "blur_ig": SaliencyExplainer,
    "xrai": SaliencyExplainer,
}


def build_explainer(model, name, params, device="cpu"):
    params = params.copy()

    if name not in EXPLAINER_REGISTRY:
        raise ValueError(
            f"Explainer {name} is not supported. Supported explainers: {list(EXPLAINER_REGISTRY.keys())}"
        )

    explainer_class = EXPLAINER_REGISTRY[name]
    if name in ["random", "gaussian_noise", "affinity"]:
        return explainer_class(model, **params)

    return explainer_class(model=model, name=name, **params).to(device)
