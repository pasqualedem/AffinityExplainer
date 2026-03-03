import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from affex.data.utils import BatchKeys
from affex.utils.torch import to_device
from affex.utils.utils import ResultDict


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype("float32"))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


def get_substrate_fn(substrate, kernel_size=3, sigma=1.0):
    r"""Returns a function that maps old pixels to new pixels.

    Args:
        substrate (str): 'blur', 'gaussian', 'random', 'zero'.
        image_size (tuple): size of the image (H, W).
        kernel_size (int): size of the kernel for blurring.
        sigma (float): standard deviation for gaussian blurring.

    Returns:
        function: a function that takes an image tensor and returns a new image tensor.
    """
    if substrate == "blur":
        kernel = gkern(kernel_size, sigma)
        return lambda x: F.conv2d(x, kernel, padding=kernel_size // 2)
    elif substrate == "gaussian":
        return lambda x: F.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    elif substrate == "random":
        return lambda x: torch.rand_like(x)
    elif substrate == "zero":
        return lambda x: torch.zeros_like(x)
    else:
        raise ValueError(f"Unknown substrate type: {substrate}")


class FSSCausalMetric(Metric):

    def __init__(
        self,
        model,
        mode,
        step=None,
        threshold_step=None,
        n_steps=None,
        substrate_fn="zero",
        percentage=None,
        loss=False,
        measure="logits",
        skip_empty=False,
        n_mid_statuses=30,
        mid_statuses_distribution="log",
    ):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            meausure (str): 'logits' or 'miou'. If 'logits', the metric will compute xAUC based on logits. If 'miou', it will compute mean IoU.
            threshold_step (float): percentage of pixels modified per one iteration, in (0, 1).
            n_steps (int): number of steps to take.
            percentage (float): percentage of pixels to modify, in (0, 1).
            loss (bool): if True, the metric will compute confidence/miou loss instead of xAUC.
            skip_empty (bool): if True, the metric will skip empty step at the beginning.
        """
        super().__init__()
        assert mode in ["del", "ins"]
        self.model = model
        self.mode = mode

        assert (
            step or threshold_step or n_steps
        ), "At least one of step, threshold_step or n_steps must be provided."
        # Only one of them should be provided
        assert not (
            step and threshold_step and n_steps
        ), "Only one of step, threshold_step or n_steps should be provided."
        assert measure in [
            "logits",
            "miou",
        ], "measure must be either 'logits' or 'miou'."

        self.step = step
        self.threshold_step = threshold_step
        self.n_steps = n_steps
        self.percentage = percentage
        self.loss = loss
        self.measure = measure
        self.skip_empty = skip_empty
        self.n_mid_statuses = n_mid_statuses
        self.mid_statuses_distribution = mid_statuses_distribution

        if self.percentage is not None:
            assert 0 < self.percentage < 1.0, "percentage must be in (0, 1)"

        self.substrate_fn = get_substrate_fn(substrate_fn)
        self.reduce = lambda x: torch.mean(x, dim=-1)

        self.mid_statuses = None
        self.mid_status_frequency = None
        self.xauc = None
        self.scores = None
        self.step_intervals = None
        self.computed_n_steps = None

        self.results = []

    def get_start_finish(self, input_dict):
        images = input_dict[BatchKeys.IMAGES]
        masks = input_dict[BatchKeys.PROMPT_MASKS]
        B, M, C, H, W = masks.shape

        query_image = images[:, 0:1]
        support_images = images[:, 1:]

        substrate_images = self.substrate_fn(support_images)
        substrate_masks = self.substrate_fn(masks)

        if self.mode == "del":
            caption = "Deleting  "
            start_si = support_images.clone()
            start_masks = masks.clone()
            finish_si = substrate_images
            finish_masks = substrate_masks
            start = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, start_si], dim=1),
                BatchKeys.PROMPT_MASKS: start_masks,
            }
            finish = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, finish_si], dim=1),
                BatchKeys.PROMPT_MASKS: finish_masks,
            }

        elif self.mode == "ins":
            caption = "Inserting "
            start_si = substrate_images
            start_masks = substrate_masks
            finish_si = support_images.clone()
            finish_masks = masks.clone()
            start = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, start_si], dim=1),
                BatchKeys.PROMPT_MASKS: start_masks,
            }
            finish = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, finish_si], dim=1),
                BatchKeys.PROMPT_MASKS: finish_masks,
            }

        return start, finish, caption

    def finish_to_start(self, start, finish, coords):
        start_masks = rearrange(
            start[BatchKeys.PROMPT_MASKS], "B M C H W -> C (B M H W)"
        )
        finish_masks = rearrange(
            finish[BatchKeys.PROMPT_MASKS], "B M C H W -> C (B M H W)"
        )
        start_images = start[BatchKeys.IMAGES]
        query_image = start_images[:, 0:1]
        start_support_images = rearrange(
            start_images[:, 1:], "B M C H W -> C (B M H W)"
        )
        finish_support_images = rearrange(
            finish[BatchKeys.IMAGES][:, 1:], "B M C H W -> C (B M H W)"
        )
        B, M, C, H, W = start[BatchKeys.PROMPT_MASKS].shape
        MHW = H * W * M

        # coords: [B, K] — indices into last dim (flattened)]
        coords = rearrange(coords, "B K -> (B K)")
        # gather values
        start_support_images[:, coords] = finish_support_images[:, coords]
        start_support_images = rearrange(
            start_support_images, "C (B M H W) -> B M C H W", M=M, H=H, W=W
        )

        start_masks[:, coords] = finish_masks[:, coords]
        start_masks = rearrange(start_masks, "C (B M H W) -> B M C H W", M=M, H=H, W=W)
        start_images = torch.cat([query_image, start_support_images], dim=1)
        start[BatchKeys.IMAGES] = start_images
        start[BatchKeys.PROMPT_MASKS] = start_masks
        return start

    def evaluate(self, input_dict, explanation, explanation_mask, gt=None):
        r"""Non-interactive evaluation: returns final xAUC and all scores."""
        for _ in self._evaluate_core(
            input_dict, explanation, explanation_mask, interactive=False, gt=gt
        ):
            pass
        return {"auc": self.xauc, "scores": self.scores}

    def evaluate_interactive(self, input_dict, explanation, explanation_mask, gt=None):
        r"""Interactive evaluation: yields intermediate states."""
        yield from self._evaluate_core(
            input_dict, explanation, explanation_mask, interactive=True, gt=gt
        )

    def _evaluate_core(
        self,
        input_dict,
        explanation,
        explanation_mask,
        interactive,
        verbose=False,
        gt=None,
    ):
        r"""Efficiently evaluate big batch of images.

        Args:
            input_dict (dict): dictionary containing image tensor, image masks
            exp_batch (np.ndarray): saliency map over the support iamges
            explanation_mask (torch.tensor): mask over the query image

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        images = input_dict[BatchKeys.IMAGES]
        masks = input_dict[BatchKeys.PROMPT_MASKS]
        B, M, C, H, W = masks.shape
        MHW = H * W * M

        device = images.device
        self.mid_statuses = []

        if self.measure == "miou":
            if gt is None:
                raise ValueError(
                    "Ground truth labels (gt) must be provided for miou computation."
                )
            miou = MulticlassJaccardIndex(
                num_classes=C, ignore_index=-100, average=None
            ).to(device)

        with torch.no_grad():
            result = self.model(to_device(input_dict, device), postprocess=False)
            preds = F.softmax(result[ResultDict.LOGITS], dim=1)
            preds = self.reduce(
                preds[:, :, explanation_mask]
            )  # Reduce over the selected pixels -> [B, C, S] (S number of selected pixels) -> [B, C]
        top = torch.argmax(preds, -1)
        ordered_saliency, salient_order = torch.sort(
            rearrange(explanation, "B M H W -> B (M H W)", M=M, H=H, W=W),
            dim=1,
            descending=True,
        )
        assert salient_order.shape == (
            B,
            MHW,
        ), f"Expected shape {(B, MHW)}, got {salient_order.shape}"

        self.set_steps(MHW, ordered_saliency)
        if self.mid_statuses_distribution == "linear":
            self.mid_status_frequency = list(
                range(0, self.computed_n_steps + 1, max(1, self.computed_n_steps // self.n_mid_statuses))
            )
        else:  # log distribution
            self.mid_status_frequency = np.unique(
                np.round(np.logspace(0, np.log10(self.computed_n_steps), num=self.n_mid_statuses)).astype(
                    int
                )
            ).tolist()

        scores = torch.empty((self.computed_n_steps + 1, B))
        start, finish, caption = self.get_start_finish(input_dict)

        # While not all pixels are changed
        for i in tqdm(
            range(self.computed_n_steps + 1),
            desc=caption + "pixels",
            disable=not verbose,
        ):

            if not (i == 0 and self.skip_empty):
                # Compute new scores
                with torch.no_grad():
                    result = self.model(to_device(start, device), postprocess=False)
                    preds = F.softmax(result[ResultDict.LOGITS], dim=1)
                reduced_preds = self.reduce(
                    preds[:, :, explanation_mask]
                )  # Reduce over the selected pixels -> [B, C, S] (S number of selected pixels) -> [B, C]
                if self.measure == "miou":
                    seg = preds.argmax(dim=1)  # Get the predicted segmentation
                    top_preds = repeat(
                        miou(seg, gt)[1:].mean(dim=0, keepdim=True), "c -> b c", b=B
                    )
                    miou.reset()
                else:
                    top_preds = reduced_preds[
                        :, top
                    ]  # Take the top classes for each batch
                scores[i] = top_preds

            if (
                i == self.computed_n_steps
            ):  # If we are at the last step, we don't need to change anything
                break
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[
                :, self.step_intervals[i] : self.step_intervals[i + 1]
            ]
            start = self.finish_to_start(start, finish, coords)

            if i in self.mid_status_frequency:
                self.mid_statuses.append(
                    (
                        i,
                        start[BatchKeys.IMAGES].clone().cpu(),
                        start[BatchKeys.PROMPT_MASKS].clone().cpu(),
                        preds,
                        top_preds.clone(),
                    )
                )
            if interactive:
                yield start, i, scores[: i + 1]

        if self.skip_empty:
            # If we skip the first empty step, we need to remove it from scores
            scores = scores[1:]

        if self.loss:
            # If confidence loss is used, the last step is the final state
            full_confidence = scores[-1].clone()
            reduced_scores = scores[:-1]  # Remove the last step from scores
            # Compute the final score as the mean of all scores
            if (
                reduced_scores.shape[1] == 1
            ):  # If there is only one batch element, xAUC it's just a value
                self.xauc = full_confidence - reduced_scores
            else:
                self.xauc = torch.abs(full_confidence - auc(reduced_scores.mean(1)))
        else:
            self.xauc = auc(scores.mean(1))

        self.scores = scores

    def set_steps(self, MHW, ordered_saliency):
        r"""Set the number of steps and step intervals based on the MHW and step size."""
        if self.n_steps is not None:
            assert self.n_steps > 0, "n_steps must be a positive integer"
            self.computed_n_steps = self.n_steps
            if self.percentage is not None and self.percentage < 1.0:
                reduced_MHW = int(MHW * self.percentage)
            else:
                reduced_MHW = MHW
            self.step_intervals = [
                reduced_MHW * i // self.n_steps for i in range(self.n_steps + 1)
            ]
        elif self.step is not None:
            self.computed_n_steps = MHW // self.step + 1
            self.step_intervals = [
                self.step * i for i in range(self.computed_n_steps)
            ] + [MHW]
            if MHW % self.step == 0:
                self.step_intervals.pop()  # Remove the last step if it is equal to MHW
                self.computed_n_steps -= 1
            self.computed_n_steps = int(self.computed_n_steps)
            if self.percentage is not None and self.percentage < 1.0:
                raise ValueError(
                    "percentage is not supported with step size, please use n_steps or threshold_step instead."
                )
        else:
            assert (
                self.threshold_step < 1.0 and self.threshold_step > 0.0
            ), "threshold_step must be in (0, 1)"
            self.computed_n_steps = int(1 / self.threshold_step)

            # Make ascending intervals
            ordered_saliency = ordered_saliency.flip(dims=[1])
            # Precompute all threshold edges
            edges = torch.arange(0, 1 + self.threshold_step, self.threshold_step)
            edges = (
                edges.unsqueeze(0)
                .expand(ordered_saliency.shape[0], -1)
                .to(ordered_saliency.device)
            )
            num_elems = ordered_saliency.shape[1]

            assert (
                ordered_saliency.shape[0] == 1
            ), "Only support batch size of 1 for now"

            # Find indices of the edges in the ordered saliency map and flip them
            self.step_intervals = (
                num_elems
                - torch.searchsorted(ordered_saliency, edges).flip(dims=[1])[0]
            )

        if self.loss and self.step_intervals[-1] != MHW:
            # If confidence loss is used, we need to add the last step
            self.step_intervals.append(MHW)
            self.computed_n_steps += 1

    def update(self, input_dict, explanation, explanation_mask, gt):
        r"""Update metric with new batch of images.

        Args:
            input_dict (dict): dictionary containing image tensor, image masks
            exp_batch (np.ndarray): saliency map over the support iamges
            explanation_mask (torch.tensor): mask over the query image
        """
        for _ in self._evaluate_core(
            input_dict, explanation, explanation_mask, interactive=False, gt=gt
        ):
            pass
        self.results.append((self.xauc, self.scores))

    def compute(self):
        r"""Compute final xAUC and scores."""
        if not self.results:
            raise ValueError("No results to compute.")
        xaucs, scores = zip(*self.results)
        # Remove nan values from xaucs
        notna_xaucs = [xauc for xauc in xaucs if not np.isnan(xauc)]
        self.xauc = np.mean(notna_xaucs)
        self.scores = np.mean(scores, axis=0)
        return {"auc": self.xauc, "scores": self.scores, "aucs": xaucs}

    def reset(self):
        r"""Reset the metric."""
        self.xauc = None
        self.scores = None
        self.results = []
        self.mid_statuses = []
        self.computed_n_steps = None
