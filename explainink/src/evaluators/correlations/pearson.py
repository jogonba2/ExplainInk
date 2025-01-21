from typing import List

import numpy as np
from scipy.stats import pearsonr

from ...types import AttributedSample, CorrelationOutput
from .base import CorrelationEvaluator


class PearsonEvaluator(CorrelationEvaluator):
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
        Computes averaged pearson correlation between the sample scores
        of `attributed_samples_a` and the sample scores of `attributed_samples_b`.

        Args:
            attributed_samples_a (List[AttributedSamples]): samples attributed by an explainer.
            attributed_samples_b (List[AttributedSamples]): samples attributed by another explainer.

        Returns:
            CorrelationOutput: scores at token and char levels.
        """
        token_pearson = []
        char_pearson = []
        for sample_a, sample_b in zip(
            attributed_samples_a, attributed_samples_b
        ):
            token_pearson.append(
                pearsonr(sample_a.scores, sample_b.scores).statistic
            )
            char_pearson.append(
                pearsonr(sample_a.char_scores, sample_b.char_scores).statistic
            )

        return CorrelationOutput(
            token_output={
                "mean": np.mean(token_pearson),
                "std": np.std(token_pearson),
            },
            char_output={
                "mean": np.mean(char_pearson),
                "std": np.std(char_pearson),
            },
        )
