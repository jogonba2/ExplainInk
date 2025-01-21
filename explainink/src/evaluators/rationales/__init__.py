# flake8: noqa

from typing import Dict, List

from transformers import PreTrainedTokenizerBase

from ...types import AttributedSample
from .aupr import AUPR
from .base import RationalesEvaluator
from .token_f1 import TokenF1
from .token_iou import TokenIOU


def evaluate_attributions(
    attributed_samples: List[AttributedSample],
    rationales: List[List[str]],
    tokenizer: str | PreTrainedTokenizerBase,
    discretizer_name: str,
    discretizer_args: Dict = {},
) -> Dict[str, float]:
    """
    Evaluate attributions using all the rationales evaluators.

    Args:
        attributed_samples (List[AttributedSamples]): list of attributed samples.
        rationales (List[List[str]]): list of rationales.
        tokenizer (str | PreTrainedTokenizerBase): tokenizer name or tokenizer of the model.
        discretizer_name (str): name of the discretization function.
                                The discretization consists on extracting the tokens that make
                                true some condition based on attribution scores, e.g., those
                                token with higher scores than the mean of the attributions.
        discretizer_args (str): args to be passed to the discretizer function.

    Return:
        Dict[str, float]: dictionary mapping metrics to its score.
    """
    evaluators = {
        "TokenIOU": TokenIOU(tokenizer, discretizer_name, discretizer_args),
        "TokenF1": TokenF1(tokenizer, discretizer_name, discretizer_args),
        "AUPR": AUPR(tokenizer, discretizer_name, discretizer_args),
    }
    results = {}
    for evaluator_name, evaluator in evaluators.items():
        results[evaluator_name] = evaluator.evaluate(
            attributed_samples, rationales
        )
    return results


__all__ = ["RationalesEvaluator", "AUPR", "TokenF1", "TokenIOU"]
