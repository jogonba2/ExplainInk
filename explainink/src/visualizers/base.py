from abc import ABC, abstractmethod
from typing import List

from ..types import AttributedSample
from .scalers import get_scaler, scale_scores


class Visualizer(ABC):
    """
    Base class for visualizers. Custom visualizers
    must extend this class and implement the `_colorize`
    method, to return HTML with colored text according
    to token or char attribution scores.

    Attributes:
        scaler_fn_name (str): name of the scaler to be used.
        token_level (bool): whether to visualize token (True) or char scores (False).
        mask_zeros (bool):  Use only non-zero scores for scaling. Defaults to True.
    """

    def __init__(
        self,
        scaler_fn_name: str = "none",
        token_level: bool = False,
        mask_zeros: bool = True,
    ):
        self.scaler_fn_name = scaler_fn_name
        self.mask_zeros = mask_zeros
        self.token_level = token_level

    def preprocess(
        self,
        samples: List[AttributedSample],
    ) -> List[AttributedSample]:
        """
        Preprocesses samples before calling `_colorize`, e.g., applying scaler fns to scores.

        Args:
            samples (List[AttributedSamples]): list of attributed samples.

        Returns:
            List[AttributedSamples]: list of preprocessed attributed samples.
        """
        after_samples = []
        if self.scaler_fn_name:
            scaler_fn = get_scaler(self.scaler_fn_name)

        for sample in samples:
            if self.scaler_fn_name:
                sample = scale_scores(sample, scaler_fn, self.mask_zeros)
            after_samples.append(sample)
        return after_samples

    @abstractmethod
    def _colorize(self, samples: List[AttributedSample]) -> List[str]:
        """
        Colorizes attributed samples using HTML format.
        Each new visualizer must override this method.

        Args:
            samples (List[AttributedSample]): list of attributed samples.

        Returns:
            List[str]: list of HTMLs, each one corresponding to one attributed sample.
        """
        ...

    def colorize(self, samples: List[AttributedSample]) -> List[str]:
        """
        Preprocesses and colorize attributed samples.

        Args:
            samples (List[AttributedSample]): list of attributed samples.

        Returns:
            List[str]: list of HTMLs, each one corresponding to one attributed sample.
        """
        preprocessed_samples = self.preprocess(samples)
        return self._colorize(preprocessed_samples)
