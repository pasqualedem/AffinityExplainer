import math
import torch
import torch.nn.functional as F
import typing
import warnings

from captum.attr._core.lime import LimeBase, construct_feature_mask
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum._utils.models.linear_model import (
    SkLearnLasso,
    SkLearnRidge,
    SGDLinearRegression,
)
from skimage.segmentation import slic
from torch import Tensor
from typing import Any, Callable, cast, List, Literal, Optional, Tuple, Union

from captum._utils.common import (
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
    _expand_additional_forward_args,
    _expand_target,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum._utils.progress import progress
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import (
    _format_input_baseline,
)
from captum.log import log_usage
from torch.utils.data import DataLoader, TensorDataset


def compute_feature_mask(
    image: torch.Tensor, num_segments: int = 100, compactness: float = 10.0
):
    """
    Compute interpretable feature mask (superpixels) for LIME on the query image.

    Args:
        image: Tensor [C, H, W], normalized to [0, 1] or [0, 255]
        num_segments: approximate number of superpixels
        compactness: tradeoff between color proximity and spatial proximity

    Returns:
        feature_mask: torch.LongTensor [H, W] with integers in [0, K-1]
    """
    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
    segments = slic(
        img_np, n_segments=num_segments, compactness=compactness, start_label=0
    )
    return torch.from_numpy(segments).long().to(image.device)


def fss_perturb_func(original_input, **kwargs):
    """
    Custom perturbation function for LIME in Few-Shot Segmentation.
    Produces interpretable, structured perturbations rather than random noise.

    Expected original_input: tuple(images, masks)
        - images: [B, 1+M, C, H, W]   (query + supports)
        - masks:  [B, M, Cmask, H, W]
    Returns:
        Binary vector [1, num_interp_features]
    """
    masks, images = original_input

    num_interp_features = kwargs.get("num_interp_features", None)
    feature_mask = kwargs.get("feature_mask", None)

    # --- If feature_mask is provided, perturb based on interpretable regions ---
    if feature_mask is not None:
        feature_mask = feature_mask[0]  # Get support feature mask [H, W]

        # Each unique integer in feature_mask corresponds to an interpretable feature
        unique_features = torch.unique(feature_mask)
        unique_features = unique_features[unique_features >= 0]  # exclude -1 if used
        K = len(unique_features)  # exclude special value for query

        # Binary activation vector: 1 = feature on, 0 = off
        active = torch.randint(0, 2, (1, K)).float().to(images.device)
        # # Add the query support features as always active
        active = torch.cat([active, torch.ones(1, 1).to(active.device)], dim=1)

        # Check if any feature mask is all zeros
        active_masks = (feature_mask * masks).any(dim=(2, 3, 4))  # [B, M]
        # Check any batch has all-zero mask
        if not active_masks.any(dim=1).all():
            for i in range(active_masks.shape[0]):
                if not active_masks[i].any():
                    mask_features = (feature_mask * masks)[i].unique()
                    # Active one random feature
                    rand_feat = mask_features[
                        torch.randint(0, len(mask_features), (1,))
                    ]
                    active[i, rand_feat] = 1.0

        return active

    # --- Fallback: simple random binary vector if no feature_mask ---
    if num_interp_features is None:
        # if unspecified, fall back to a small fixed number
        num_interp_features = 8
    return torch.randint(0, 2, (1, num_interp_features), device=images.device).float()


class CleanTensorDataset(TensorDataset):
    """
    TensorDataset that removes nans and infs from the data.
    """

    def __init__(self, *tensors: Tensor) -> None:
        masks = torch.stack(
            [torch.isfinite(t).all(dim=tuple(range(1, t.dim()))) for t in tensors]
        ).all(dim=0)
        clean_tensors = tuple(t[masks] for t in tensors)
        super().__init__(*clean_tensors)


class FSSLime(LimeBase):
    def __init__(
        self,
        forward_func,
        interpretable_model=None,
        similarity_func=None,
        perturb_func=None,
        model_type: str = None,
        use_baselines: bool = False,
        kernel_width: float = 1.0,
        image_weight: float = 0.2,
        slic_num_segments: int = 80,
    ):
        """
        Few-Shot Semantic Segmentation (FSS) variant of LIME.

        Automatically handles baselines for both query/support images and support masks.

        Args:
            forward_func: model forward function
            interpretable_model: optional custom surrogate model
            similarity_func: optional similarity kernel
            perturb_func: optional sampling function
            model_type: "ridge", "lasso", or "linear" (SGD)
            use_baselines: if True, generate and use baselines for images and masks;
                           if False, do not use baselines (standard LIME behavior)
            kernel_width: float = 1.0, kernel width for similarity function
            image_weight: float = 0.2, weight for image similarity vs mask similarity
        """

        if interpretable_model is None:
            if model_type == "ridge":
                interpretable_model = SkLearnRidge(alpha=1.0)
            else:
                interpretable_model = SkLearnLasso(alpha=0.01)

        if perturb_func is None:
            perturb_func = fss_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            self.fss_similarity_func,
            perturb_func,
            True,
            self.fss_from_interp_rep_transform,
            None,
        )
        self.use_baselines = use_baselines
        self.slic_num_segments = slic_num_segments
        self.current_valid_masks = None
        self.kernel_width = kernel_width
        self.image_weight = image_weight

    def fss_from_interp_rep_transform(
        self, curr_sample, original_inputs, reduce_empty=True, **kwargs
    ):
        assert (
            "feature_mask" in kwargs
        ), "Must provide feature_mask to use default interpretable representation transform"
        assert (
            "baselines" in kwargs
        ), "Must provide baselines to use default interpretable representation transform"
        feature_mask = kwargs["feature_mask"]
        if isinstance(feature_mask, torch.Tensor):
            binary_mask = curr_sample[0][feature_mask].bool()
            return (
                binary_mask.to(original_inputs.dtype) * original_inputs
                + (~binary_mask).to(original_inputs.dtype) * kwargs["baselines"]
            )
        else:
            binary_mask = tuple(
                curr_sample[0][feature_mask[j]].bool() for j in range(len(feature_mask))
            )
            masks, images = tuple(
                binary_mask[j].to(original_inputs[j].dtype) * original_inputs[j]
                + (~binary_mask[j]).to(original_inputs[j].dtype)
                * kwargs["baselines"][j]
                for j in range(len(feature_mask))
            )
            # If any masks has all zeros, remove that mask
            valid_masks = masks.any(dim=(2, 3, 4)).any(dim=0)  # [M]
            self.current_valid_masks = valid_masks
            if not valid_masks.all() and reduce_empty:
                masks = masks[:, valid_masks]
                images = images[
                    :,
                    torch.cat(
                        [
                            torch.ones(1, dtype=torch.bool, device=images.device),
                            valid_masks,
                        ]
                    ),
                ]

            return masks, images

    def fss_similarity_func(
        self,
        original_input,
        perturbed_input,
        perturbed_interpretable_input=None,
        **kwargs,
    ):
        """
        Custom similarity function for LIME in Few-Shot Segmentation.
        Computes similarity based on both images and masks.

        Expected original_input: tuple(images, masks)
            - images: [B, 1+M, C, H, W]   (query + supports)
            - masks:  [B, M, Cmask, H, W]
        Returns:
            Similarity score [1, N] where N is number of perturbed samples
        """
        orig_masks, orig_images = original_input
        pert_masks, pert_images = perturbed_input

        support_orig_images = orig_images[:, 1:]  # [B, M, C, H, W]
        support_pert_images = pert_images[:, 1:]  # [B, M, C, H, W]

        alpha = kwargs.get("alpha", 0.2)  # weight for images vs masks

        # Flatten images and masks
        orig_images_flat = support_orig_images[:, self.current_valid_masks].flatten(
            start_dim=2
        )
        pert_images_flat = support_pert_images.flatten(start_dim=2)

        # Compute Cosine distances for images
        image_sims = F.cosine_similarity(
            orig_images_flat, pert_images_flat, dim=2
        )  # [N]
        zeros_sims = torch.zeros(
            self.current_valid_masks.logical_not().sum(), device=image_sims.device
        )  # [num_invalid]
        if zeros_sims.numel() > 0:
            # If any invalid masks, add zero similarities for them
            image_sims = torch.cat(
                [image_sims, zeros_sims.to(image_sims).unsqueeze(0)], dim=1
            )

        image_sims = image_sims.mean(dim=1)  # [N]
        image_distances = 1 - image_sims  # convert similarity to distance

        # Mask distances is how many ones are removed
        mask_distances = (orig_masks[:, self.current_valid_masks] - pert_masks).sum(
            dim=(2, 3, 4)
        ) / orig_masks[:, self.current_valid_masks].sum(dim=(2, 3, 4))
        if zeros_sims.numel() > 0:
            # If any invalid masks, add zero distances for them
            mask_distances = torch.cat(
                [
                    mask_distances,
                    torch.ones_like(zeros_sims, device=mask_distances.device).unsqueeze(
                        0
                    ),
                ],
                dim=1,
            )

        mask_distances = mask_distances.mean(dim=1)  # [N]

        total_distances = alpha * image_distances + (1 - alpha) * mask_distances

        # Convert distances to similarities using exponential kernel
        kernel_width = kwargs.get("kernel_width", 1.0)  # default kernel width
        similarities = torch.exp(-(total_distances**2) / (kernel_width**2))

        return similarities.squeeze(0)  # [N]

    # ---- Baseline generation ----
    def generate_baselines(self, images, masks):
        """
        Create baselines for both images and masks.
        - For images: blurred query + mean of supports.
        - For masks: deterministic zero tensor.
        """
        B, M_plus_1, C, H, W = images.shape
        _, M, _, _, _ = masks.shape

        # Query = first image, supports = rest
        query = images[:, 0]
        supports = images[:, 1:]

        # Blurred query as baseline
        blurred = torch.nn.functional.avg_pool2d(
            query, kernel_size=3, stride=1, padding=1
        )

        # Mean of support images as baseline
        mean_support = supports.mean(dim=1, keepdim=True)

        baseline_images = (
            0.5 * blurred.unsqueeze(1) + 0.5 * mean_support
        )  # soft interpolation
        baseline_masks = torch.zeros_like(masks)  # zeroed support masks

        return baseline_images, baseline_masks

    # ---- Override LIME attribution ----
    def attribute(self, *args, **kwargs):
        """
        Builds baselines from FSS batch dict, reorders according to `ordering`,
        and calls standard LIME attribution.
        """
        batch_dict = args[0]
        ordering = kwargs["additional_forward_args"][-1]

        images = batch_dict[ordering["images"]]  # [B, 1+M, C, H, W]
        masks = batch_dict[ordering["prompt_masks"]]  # [B, M, H, W]

        baselines_images, baselines_masks = self.generate_baselines(images, masks)

        if self.use_baselines:
            # Remove redundant singleton dimension if present
            baselines_images = baselines_images.squeeze(1)
            baselines_masks = baselines_masks.squeeze(1)

            # Pack tuple in correct order
            baselines_tuple = [None] * 2
            baselines_tuple[ordering["images"]] = baselines_images
            baselines_tuple[ordering["prompt_masks"]] = baselines_masks
            baselines_tuple = tuple(baselines_tuple)

            kwargs["baselines"] = baselines_tuple

        self.interpretable_model = self.interpretable_model.to(images.device)
        support_images = images[:, 1:]
        support_feature_mask = (
            torch.stack(
                [
                    torch.stack(
                        [
                            compute_feature_mask(
                                support_images[i, j],
                                num_segments=self.slic_num_segments,
                            )
                            for j in range(support_images.shape[1])
                        ],
                        dim=0,
                    )
                    for i in range(support_images.shape[0])
                ],
                dim=0,
            )
            .unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
        )  # [B, M, rgb, H, W]
        # query_feature_mask has special value that LIME ignores it (max value + 1)
        query_feature_mask = torch.ones_like(support_feature_mask[:, 0]) * (
            support_feature_mask.max() + 1
        )
        feature_mask = torch.cat(
            [query_feature_mask.unsqueeze(1), support_feature_mask], dim=1
        )

        # Support_mask feature_mask is the same as support_feature_mask
        support_masks_feature_mask = (
            support_feature_mask[:, :, 0]
            .unsqueeze(2)
            .repeat(1, 1, masks.shape[2], 1, 1)
        )
        feature_mask = (support_masks_feature_mask, feature_mask)

        return self._attribute_kwargs(*args, feature_mask=feature_mask, **kwargs)

    @log_usage()
    @torch.no_grad()
    def base_attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: object,
    ) -> Tensor:

        inp_tensor = cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
        device = inp_tensor.device

        interpretable_inps = []
        similarities = []
        outputs = []

        curr_model_inputs = []
        expanded_additional_args = None
        expanded_target = None
        gen_perturb_func = self._get_perturb_generator_func(inputs, **kwargs)

        if show_progress:
            attr_progress = progress(
                total=math.ceil(n_samples / perturbations_per_eval),
                desc=f"{self.get_name()} attribution",
            )
            attr_progress.update(0)

        batch_count = 0
        for _ in range(n_samples):
            try:
                interpretable_inp, curr_model_input = gen_perturb_func()
            except StopIteration:
                warnings.warn(
                    "Generator completed prior to given n_samples iterations!",
                    stacklevel=1,
                )
                break
            batch_count += 1
            interpretable_inps.append(interpretable_inp)
            curr_model_inputs.append(curr_model_input)

            curr_sim = self.similarity_func(
                inputs,
                curr_model_input,
                interpretable_inp,
                **{
                    **kwargs,
                    "kernel_width": self.kernel_width,
                    "alpha": self.image_weight,
                },
            )
            similarities.append(
                curr_sim.flatten()
                if isinstance(curr_sim, Tensor)
                else torch.tensor([curr_sim], device=device)
            )

            if len(curr_model_inputs) == perturbations_per_eval:
                if expanded_additional_args is None:
                    expanded_additional_args = _expand_additional_forward_args(
                        additional_forward_args, len(curr_model_inputs)
                    )
                if expanded_target is None:
                    expanded_target = _expand_target(target, len(curr_model_inputs))

                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )

                if show_progress:
                    attr_progress.update()

                outputs.append(model_out)

                curr_model_inputs = []

        if len(curr_model_inputs) > 0:
            expanded_additional_args = _expand_additional_forward_args(
                additional_forward_args, len(curr_model_inputs)
            )
            expanded_target = _expand_target(target, len(curr_model_inputs))
            model_out = self._evaluate_batch(
                curr_model_inputs,
                expanded_target,
                expanded_additional_args,
                device,
            )
            if show_progress:
                attr_progress.update()
            outputs.append(model_out)

        if show_progress:
            attr_progress.close()

        # Argument 1 to "cat" has incompatible type
        # "list[Tensor | tuple[Tensor, ...]]";
        # expected "tuple[Tensor, ...] | list[Tensor]"  [arg-type]
        combined_interp_inps = torch.cat(interpretable_inps).float()  # type: ignore
        combined_outputs = (
            torch.cat(outputs) if len(outputs[0].shape) > 0 else torch.stack(outputs)
        ).float()
        combined_sim = (
            torch.cat(similarities)
            if len(similarities[0].shape) > 0
            else torch.stack(similarities)
        ).float()
        dataset = CleanTensorDataset(
            combined_interp_inps, combined_outputs, combined_sim
        )
        if len(dataset) == 0:  # No valid samples
            warnings.warn(
                "No valid samples to fit LIME interpretable model!", stacklevel=1
            )
            return torch.zeros_like(inputs[1], dtype=torch.float)
        self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))
        return self.interpretable_model.representation()

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        return super().attribute_future()

    def _attribute_kwargs(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
        **kwargs: object,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. ",
                stacklevel=1,
            )

        coefs: Tensor
        if bsz > 1:
            test_output = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            if isinstance(test_output, Tensor) and torch.numel(test_output) > 1:
                if torch.numel(test_output) == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a "
                        "time.",
                        stacklevel=1,
                    )
                    output_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                    ):
                        coefs = super().attribute.__wrapped__(
                            self,
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        if return_input_shape:
                            output_list.append(
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    coefs,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                            )
                        else:
                            output_list.append(coefs.reshape(1, -1))  # type: ignore

                    return _reduce_list(output_list)
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        coefs = self.base_attribute.__wrapped__(
            self,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )
        if return_input_shape:
            # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got
            #  `Tuple[Tensor, ...]`.
            return self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                coefs,
                num_interp_features,
                is_inputs_tuple,
            )
        else:
            return coefs

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[True],
    ) -> Tuple[Tensor, ...]: ...

    @typing.overload
    def _convert_output_shape(  # type: ignore
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[False],
    ) -> Tensor: ...

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        return _format_output(is_inputs_tuple, tuple(attr))
