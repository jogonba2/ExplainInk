from abc import ABC, abstractmethod
from typing import List

from ...types import AttributedSample, CorrelationOutput


class CorrelationEvaluator(ABC):
    """
    Base class for attribution evaluators based on
    correlation/similarity between scores, e.g., attention explainer
    scores compared with integrated gradient scores.

    """

    def __init__(self): ...

    @abstractmethod
    def evaluate(
        self,
        attributed_samples_a: List[AttributedSample],
        attributed_samples_b: List[AttributedSample],
    ) -> CorrelationOutput:
        """
        Computes, correlation/similarity scores between the samples attributed
        using one explainer (`attributed_samples_a`) and the samples attributed
        by another explainer (`attributed_samples_b`).

        This method must be implemented for each correlation evaluator

        Args:
            attributed_samples_a (List[AttributedSamples]): samples attributed by an explainer.
            attributed_samples_b (List[AttributedSamples]): samples attributed by another explainer.

        Returns:
            CorrelationOutput: scores at token and char levels.
        """
        ...
