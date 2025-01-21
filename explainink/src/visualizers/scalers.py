from functools import partial
from typing import Callable, Optional

import numpy as np

from ..types import AttributedSample


def higher_mean(
    x: np.ndarray, hard: bool = True, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Replaces those values lower than the mean by zero.
    Values larger than the mean can be keeped (`hard`=False)
    or replaced by 1 (`hard`=True).

    Args:
        x (np.ndarray): array to be scaled.
        hard (bool): whether to do hard scaling.
        mask (Optional[np.ndarray]): Mask indicating attributions used
            for scaling (elements not indicated in mask will have value 0).
            If None, masking not applied. Defaults to None.

    Returns:
        np.ndarray: scaled array.
    """
    mean = x[mask].mean() if mask is not None else x.mean()

    x[x < mean] = 0
    if hard:
        x[x > mean] = 1
    return x


def minmax(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Minmax scaling.

    Args:
        x (np.ndarray): array to be scaled.
        mask (Optional[np.ndarray]): Mask indicating attributions used
            for scaling (elements not indicated in mask will have value 0).
            If None, masking not applied. Defaults to None.

    Returns:
        np.ndarray: scaled array.
    """
    if mask is not None:
        x[~mask] = 0
        x_min = x[mask].min()
        x_max = x[mask].max()

        x[mask] = (x[mask] - x_min) / (x_max - x_min)
    else:
        x = (x - x.min()) / (x.max() - x.min())

    return x


def standard(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Standard scaling.

    Args:
        x (np.ndarray): array to be scaled.
        mask (Optional[np.ndarray]): Mask indicating attributions used
            for scaling (elements not indicated in mask will have value 0).
            If None, masking not applied. Defaults to None.

    Returns:
        np.ndarray: scaled array.
    """
    if mask is not None:
        return (x - x[mask].mean()) / (x[mask].std(ddof=1))

    return (x - x.mean()) / (x.std(ddof=1))


def identity(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Identity scaling (no scaling)

    Args:
        x (np.ndarray): array to be scaled.
        mask (Optional[np.ndarray]): Mask indicating attributions used
            for scaling (elements not indicated in mask will have value 0).
            If None, masking not applied. Defaults to None.

    Returns:
        np.ndarray: scaled array.
    """
    return x


def get_scaler(name: str) -> Callable:
    """
    Get an scaler by name.

    Args:
        name (str): name of an scaler.
        mask (Optional[np.ndarray]): Mask indicating attributions used
            for scaling (elements not indicated in mask will have value 0).
            If None, masking not applied. Defaults to None.

    Returns:
        Callable: scaler function.
    """
    if name == "minmax":
        return minmax
    elif name == "standard":
        return standard
    elif name == "hard_higher_mean":
        return partial(higher_mean, hard=True)
    elif name == "soft_higher_mean":
        return partial(higher_mean, hard=False)
    elif name == "none":
        return identity
    else:
        raise ValueError(f"Scaler {name} not implemented.")


def scale_scores(
    sample: AttributedSample, scaler_fn: Callable, mask_zeros: bool = True
) -> AttributedSample:
    """
    Scales the scores of a sample (both at token and char level).

    Args:
        sample (AttributedSample): attributed sample.
        scaler_fn (Callable): scaler function.
        mask_zeros (bool): Should use only those attributions for
            scaling which value is non-zero. Defaults to True.

    Returns:
        AttributedSample: sample with scaled scores.
    """
    token_mask = sample.scores != 0 if mask_zeros else None
    sample.scores = scaler_fn(sample.scores, mask=token_mask)

    char_mask = sample.char_scores != 0 if mask_zeros else None
    sample.char_scores = scaler_fn(sample.char_scores, mask=char_mask)

    return sample
