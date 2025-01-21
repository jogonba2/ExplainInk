# flake8: noqa

from typing import Dict, List

from ...types import AttributedSample, CorrelationOutput
from .base import CorrelationEvaluator
from .cosine import CosineEvaluator
from .pearson import PearsonEvaluator


def evaluate_attributions(
    attributed_samples_a: List[AttributedSample],
    attributed_samples_b: List[AttributedSample],
) -> Dict[str, CorrelationOutput]:
    """
    Evaluate attributions using all the correlation evaluators.

    Args:
        attributed_samples_a (List[AttributedSamples]): samples attributed by an explainer.
        attributed_samples_b (List[AttributedSamples]): samples attributed by another explainer.

    Return:
        Dict[str, CorrelationOutput]: dictionary mapping metrics to scores.
    """
    evaluators = {
        "Pearson": PearsonEvaluator(),
        "Cosine": CosineEvaluator(),
    }
    results = {}
    for evaluator_name, evaluator in evaluators.items():
        results[evaluator_name] = evaluator.evaluate(
            attributed_samples_a, attributed_samples_b
        )
    return results


__all__ = ["CorrelationEvaluator", "PearsonEvaluator", "CosineEvaluator"]
