from typing import Dict, List

from ...common.utils import remove_prefix_symbols
from ...types import AttributedSample
from .base import RationalesEvaluator


class TokenF1(RationalesEvaluator):
    """
    Metric that computes the F1 score at token level.
    """

    def __init__(
        self, tokenizer: str, discretizer_name: str, discretizer_args: Dict = {}
    ):
        super().__init__(tokenizer, discretizer_name, discretizer_args)

    def evaluate_sample(
        self,
        sample: AttributedSample,
        rationale: List[str],
    ) -> float:
        ref = set(remove_prefix_symbols(rationale))
        pred = set(remove_prefix_symbols(self.discretizer(sample)))
        if len(pred) == 0 or len(ref) == 0:
            return 0.0
        precision = len(ref.intersection(pred)) / len(pred)
        recall = len(ref.intersection(pred)) / len(ref)
        num = precision * recall
        denom = precision + recall
        if denom == 0:
            return 0.0
        return 2 * (num / denom)
