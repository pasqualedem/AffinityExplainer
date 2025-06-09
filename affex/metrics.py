import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from torchmetrics import Metric

from affex.data.utils import BatchKeys
from affex.utils.utils import ResultDict, to_device


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

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
    if substrate == 'blur':
        kernel = gkern(kernel_size, sigma)
        return lambda x: F.conv2d(x, kernel, padding=kernel_size//2)
    elif substrate == 'gaussian':
        return lambda x: F.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    elif substrate == 'random':
        return lambda x: torch.rand_like(x)
    elif substrate == 'zero':
        return lambda x: torch.zeros_like(x)
    else:
        raise ValueError(f"Unknown substrate type: {substrate}")

class FSSCausalMetric(Metric):

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super().__init__()
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = get_substrate_fn(substrate_fn)
        self.reduce = lambda x : torch.mean(x, dim=-1)
        
        self.mid_statuses = None
        self.mid_status_frequency = None
        self.xauc = None
        self.scores = None
        self.n_steps = None
        
        self.results = []
        
    def get_start_finish(self, input_dict):
        images = input_dict[BatchKeys.IMAGES]
        masks = input_dict[BatchKeys.PROMPT_MASKS]
        B, M, C, H, W = masks.shape
        
        query_image = images[:, 0:1]
        support_images = images[:, 1:]
        
        substrate_images = self.substrate_fn(support_images)
        substrate_masks = self.substrate_fn(masks)
        
        if self.mode == 'del':
            caption = 'Deleting  '
            start_si = support_images.clone()
            start_masks = masks.clone()
            finish_si = substrate_images
            finish_masks = substrate_masks
            start = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, start_si], dim=1),
                BatchKeys.PROMPT_MASKS: start_masks
            }
            finish = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, finish_si], dim=1),
                BatchKeys.PROMPT_MASKS: finish_masks
            }
            
        elif self.mode == 'ins':
            caption = 'Inserting '
            start_si = substrate_images
            start_masks = substrate_masks
            finish_si = support_images.clone()
            finish_masks = masks.clone()
            start = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, start_si], dim=1),
                BatchKeys.PROMPT_MASKS: start_masks
            }
            finish = {
                **input_dict,
                BatchKeys.IMAGES: torch.cat([query_image, finish_si], dim=1),
                BatchKeys.PROMPT_MASKS: finish_masks
            }
            
        return start, finish, caption
    
    def finish_to_start(self, start, finish, coords):
        start_masks = rearrange(start[BatchKeys.PROMPT_MASKS], 'B M C H W -> C (B M H W)')
        finish_masks = rearrange(finish[BatchKeys.PROMPT_MASKS], 'B M C H W -> C (B M H W)')
        start_images = start[BatchKeys.IMAGES]
        query_image = start_images[:, 0:1]
        start_support_images = rearrange(start_images[:, 1:], 'B M C H W -> C (B M H W)')
        finish_support_images = rearrange(finish[BatchKeys.IMAGES][:, 1:], 'B M C H W -> C (B M H W)')
        B, M, C, H, W = start[BatchKeys.PROMPT_MASKS].shape
        MHW = H * W * M
        
        # coords: [B, K] — indices into last dim (flattened)]
        coords = rearrange(coords, 'B K -> (B K)')
        # gather values
        start_support_images[:, coords] = finish_support_images[:, coords]
        start_support_images = rearrange(start_support_images, 'C (B M H W) -> B M C H W', M=M, H=H, W=W)
        
        start_masks[:, coords] = finish_masks[:, coords]
        start_masks = rearrange(start_masks, 'C (B M H W) -> B M C H W', M=M, H=H, W=W)
        start_images = torch.cat([query_image, start_support_images], dim=1)
        start[BatchKeys.IMAGES] = start_images
        start[BatchKeys.PROMPT_MASKS] = start_masks
        return start
    
    def evaluate(self, input_dict, explanation, explanation_mask):
        r"""Non-interactive evaluation: returns final xAUC and all scores."""
        for _ in self._evaluate_core(input_dict, explanation, explanation_mask, interactive=False):
            pass
        return {"auc": self.xauc, "scores": self.scores}


    def evaluate_interactive(self, input_dict, explanation, explanation_mask):
        r"""Interactive evaluation: yields intermediate states."""
        yield from self._evaluate_core(input_dict, explanation, explanation_mask, interactive=True)

    def _evaluate_core(self, input_dict, explanation, explanation_mask, interactive):
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
        
        with torch.no_grad():
            result = self.model(to_device(input_dict, device), postprocess=False)
            preds = F.softmax(result[ResultDict.LOGITS], dim=1)
            preds = self.reduce(preds[:, :, explanation_mask]) # Reduce over the selected pixels -> [B, C, S] (S number of selected pixels) -> [B, C]
        top = torch.argmax(preds, -1)
        self.n_steps = (MHW + self.step - 1) // self.step
        self.mid_status_frequency = self.n_steps // 10 if self.n_steps > 10 else 1
        scores = torch.empty((self.n_steps + 1, B))
        salient_order = torch.sort(rearrange(explanation, "B M H W -> B (M H W)", M=M, H=H, W=W), dim=1, descending=True)[1]
        assert salient_order.shape == (B, MHW), f"Expected shape {(B, MHW)}, got {salient_order.shape}"

        start, finish, caption = self.get_start_finish(input_dict)

        # While not all pixels are changed
        for i in tqdm(range(self.n_steps+1), desc=caption + 'pixels'):
            # clear_output()
            # display(unnormalize(start[BatchKeys.IMAGES][0, 1:]).rgb)
            # display(unnormalize(finish[BatchKeys.IMAGES][0, 1:]).rgb)
            # Compute new scores
            with torch.no_grad():
                result = self.model(to_device(start, device), postprocess=False)
                preds = F.softmax(result[ResultDict.LOGITS], dim=1)
            preds = self.reduce(preds[:, :, explanation_mask]) # Reduce over the selected pixels -> [B, C, S] (S number of selected pixels) -> [B, C]
            top_preds = preds[:, top] # Take the top classes for each batch
            scores[i] = top_preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start = self.finish_to_start(start, finish, coords)
            
            if i % self.mid_status_frequency == 0:
                self.mid_statuses.append(
                    (i, start[BatchKeys.IMAGES].clone().cpu(), start[BatchKeys.PROMPT_MASKS].clone().cpu(), top_preds.clone())
                )
            if interactive:
                yield start, i, scores[:i+1]
        self.xauc = auc(scores.mean(1))
        self.scores = scores
        
    def update(self, input_dict, explanation, explanation_mask):
        r"""Update metric with new batch of images.

        Args:
            input_dict (dict): dictionary containing image tensor, image masks
            exp_batch (np.ndarray): saliency map over the support iamges
            explanation_mask (torch.tensor): mask over the query image
        """
        for _ in self._evaluate_core(input_dict, explanation, explanation_mask, interactive=False):
            pass
        self.results.append((self.xauc, self.scores))
        
    def compute(self):
        r"""Compute final xAUC and scores."""
        if not self.results:
            raise ValueError("No results to compute.")
        xaucs, scores = zip(*self.results)
        self.xauc = np.mean(xaucs)
        self.scores = np.mean(scores, axis=0)
        return {"auc": self.xauc, "scores": self.scores, "aucs": xaucs}
    
    def reset(self):
        r"""Reset the metric."""
        self.xauc = None
        self.scores = None
        self.n_steps = None
        self.results = []
    