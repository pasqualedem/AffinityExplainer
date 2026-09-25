"""Two geometry helpers from SANSA's util/commons.py.

The rest of that module is training scaffolding (logging, checkpoint resume) that
the inference path never touches, so it is not vendored.
"""
import torch
import torch.nn.functional as F


def resize_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Resize a mask to the model image size.

    Args:
        mask: Tensor [1, N, H, W].

    Returns:
        Boolean tensor of shape [1, N, IMG, IMG].
    """
    mask = F.interpolate(
        mask,
        (image_size, image_size),
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )
    return (mask.float() > 0)


def rescale_points(points: torch.Tensor, from_hw, to_hw):
    """
    points: (..., 2) tensor of (x, y) in pixels for an image of size from_hw=(H0,W0)
    returns: points rescaled to to_hw=(H1,W1)
    """
    H0, W0 = from_hw
    H1, W1 = to_hw
    scale = points.new_tensor([W1 / W0, H1 / H0])
    return points * scale
