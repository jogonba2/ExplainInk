from typing import Dict, List

from ...common.utils import remove_prefix_symbols
from ...types import AttributedSample
from .base import RationalesEvaluator


class TokenIOU(RationalesEvaluator):
    """
    Metric that computes the intersection over the union score at token level.
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
        num = len(ref.intersection(pred))
        denom = len(ref.union(pred))
        if denom == 0:
            return 0.0
        return num / denom
