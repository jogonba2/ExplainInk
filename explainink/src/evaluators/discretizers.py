from math import ceil
from typing import List

import numpy as np

from ..types import AttributedSample


def thresholding(
    sample: AttributedSample, threshold: float, **kwargs
) -> List[str]:
    """
    Returns the tokens whose score surpasses a given threshold.

    Args:
        sample (AttributedSample): an attributed sample.
        threshold (float): a threshold.

    Returns:
        List[str]: list of tokens.
    """
    return np.array(sample.tokens)[sample.scores > threshold].tolist()


def gt_mean(sample: AttributedSample, **kwargs) -> List[str]:
    """
    Returns the tokens whose score surpasses a the mean score.

    Args:
        sample (AttributedSample): an attributed sample.

    Returns:
        List[str]: list of tokens.
    """
    return thresholding(sample, threshold=sample.scores.mean())


def gt_median(sample: AttributedSample, **kwargs) -> List[str]:
    """
    Returns the tokens whose score surpasses a the median score.

    Args:
        sample (AttributedSample): an attributed sample.

    Returns:
        List[str]: list of tokens.
    """
    return thresholding(sample, threshold=np.median(sample.scores))


def gt_25(sample: AttributedSample, **kwargs) -> List[str]:
    """
    Returns the tokens whose score surpasses a the first quartile.

    Args:
        sample (AttributedSample): an attributed sample.

    Returns:
        List[str]: list of tokens.
    """
    return thresholding(
        sample, threshold=sample.scores.mean() - sample.scores.std()
    )


def top_k(sample: AttributedSample, k: int, **kwargs) -> List[str]:
    """
    Returns the `k` tokens with highest score.

    Args:
        sample (AttributedSample): an attributed sample.

    Returns:
        List[str]: list of tokens.
    """
    sorted_idxs = sample.scores.argsort()[::-1][:k]
    return np.array(sample.tokens)[sorted_idxs].tolist()


def top_percentage(
    sample: AttributedSample, percentage: float, **kwargs
) -> List[str]:
    """
    Returns the `percentage`% tokens with highest score.

    Args:
        sample (AttributedSample): an attributed sample.

    Returns:
        List[str]: list of tokens.
    """
    k = ceil(len(sample.tokens) * percentage)
    return top_k(sample, k)
