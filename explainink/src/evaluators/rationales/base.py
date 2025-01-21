from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ...types import AttributedSample
from .. import discretizers


class RationalesEvaluator(ABC):
    """
    Base class for attribution evaluators based on rationales.
    These metrics compare predicted token attributions
    with rationales from human-labeled datasets.

    Attributes:
        tokenizer (str | PreTrainedTokenizerBase): tokenizer name or tokenizer of the model.
        discretizer_name (str): name of the discretization function.
                                The discretization consists on extracting the tokens that make
                                true some condition based on attribution scores, e.g., those
                                token with higher scores than the mean of the attributions.
        discretizer_args (str): args to be passed to the discretizer function.
    """

    def __init__(
        self,
        tokenizer: str | PreTrainedTokenizerBase,
        discretizer_name: str,
        discretizer_args: Dict = {},
    ):
        self.discretizer = partial(
            getattr(discretizers, discretizer_name), **discretizer_args
        )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    @abstractmethod
    def evaluate_sample(
        self,
        sample: AttributedSample,
        rationale: List[str],
    ) -> float:
        """
        Computes the evaluation metric on an attributed sample.
        This method must be implemented in each new metric.

        Args:
            sample (AttributedSample): attributed sample.
            rationale (List[str]): tokenized reference rationales for the sample.

        Returns:
            float: metric value.
        """

        ...

    def evaluate(
        self,
        attributed_samples: List[AttributedSample],
        rationales: List[List[str]],
    ) -> float:
        """
        Computes the mean of the evaluation metric across all the attributed samples.
        This method tokenizes the rationales to match the token-level attributions.

        Args:
            attributed_samples (List[AttributedSamples]): list of attributed samples.
            rationales (List[List[str]]): list of rationales.

        Returns:
            float: averaged evaluation metric across all the attributed samples.
        """
        results = [
            self.evaluate_sample(
                sample=sample,
                rationale=sum(
                    [
                        self.tokenizer.convert_ids_to_tokens(
                            self.tokenizer(word, add_special_tokens=False)[
                                "input_ids"
                            ]
                        )
                        for word in rationale
                    ],
                    [],
                ),
            )
            for sample, rationale in zip(attributed_samples, rationales)
        ]
        return np.mean(results)
