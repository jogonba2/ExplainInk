from typing import Dict, List

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from ...common.utils import remove_prefix_symbols
from ...types import AttributedSample
from .base import RationalesEvaluator


class AUPR(RationalesEvaluator):
    """
    Metric that computes the area under the precision-recall curve at token-level.
    From:
    https://github.com/g8a9/ferret/blob/22935f8fd9033bae4dc385039a2852da0715d3f0/ferret/evaluators/plausibility_measures.py#L13
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
        rationale_set = set(remove_prefix_symbols(rationale))
        sample_tokens = remove_prefix_symbols(sample.tokens)
        rationale_one_hot = np.zeros(len(sample_tokens))
        for token in rationale_set:
            if token in sample_tokens:
                rationale_one_hot[sample_tokens.index(token)] = 1.0
        precision, recall, _ = precision_recall_curve(
            rationale_one_hot, sample.scores
        )
        auc_score = auc(recall, precision)
        return auc_score
