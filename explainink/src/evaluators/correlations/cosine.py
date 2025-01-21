from typing import List

import numpy as np
from scipy.spatial.distance import cosine

from ...types import AttributedSample, CorrelationOutput
from .base import CorrelationEvaluator


class CosineEvaluator(CorrelationEvaluator):
    """
    Base class for the correlation evaluator
    based on Pearson correlation.
    """

    def __init__(self): ...

    def evaluate(
        self,
        attributed_samples_a: List[AttributedSample],
        attributed_samples_b: List[AttributedSample],
    ) -> CorrelationOutput:
        """
        Computes average cosine similarity between the samples of
        of `attributed_samples_a` and the samples of `attributed_samples_b`.

        Args:
            attributed_samples_a (List[AttributedSamples]): samples attributed by an explainer.
            attributed_samples_b (List[AttributedSamples]): samples attributed by another explainer.

        Returns:
            CorrelationOutput: cosine similarity at token and char levels.
        """
        token_sims = []
        char_sims = []
        for sample_a, sample_b in zip(
            attributed_samples_a, attributed_samples_b
        ):
            token_sims.append(1.0 - cosine(sample_a.scores, sample_b.scores))
            char_sims.append(
                1.0 - cosine(sample_a.char_scores, sample_b.char_scores)
            )

        return CorrelationOutput(
            token_output={
                "mean": np.mean(token_sims),
                "std": np.std(token_sims),
            },
            char_output={"mean": np.mean(char_sims), "std": np.std(char_sims)},
        )
