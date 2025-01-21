import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ...types import AttributedSample


class DataExplainer(ABC):
    """
    Base class for data-based explainers.
    The explainers are in charge of creating attributed samples
    by using trained models.

    Attributes:
        language (str): language of the texts.
    """

    def __init__(
        self,
        language: str = "en",
        **kwargs,
    ):
        self.language = language
        self.kwargs = kwargs

    @abstractmethod
    def _get_span_scores(
        self, texts: List[str], labels: List[int], target_label: int
    ) -> List[Tuple[str, float]]:
        """
        Method to compute span scores for a given `target_label`, from a list of `texts`.
        This method must be implemented by each new explainer.

        Args:
            texts (List[str]): list of texts.
            labels (List[int]): list of labels.
            target_label (int): a target label.

        Returns:
            List[Tuple[str, float]]: list of tuples containing span and score.
        """
        ...

    def _attribute_sample(
        self, text: str, label: int, label_scores: List[Tuple[str, float]]
    ) -> AttributedSample:
        """
        Base method to build an AttributedSample from a `text`, its `label` and the
        `label_scores` of the label.

        In this base implementation, the resulting AttributedSample will not
        contain token-related info (tokens, score, word_ids, and token_to_chars),
        since it depends on the tokenization of each DataExplainer.
        `char_scores` are computed by following a best-effort strategy
        to deal with overlapping spans:

        Example:
            text = "I don't hate you"
            label_scores = [("don't hate", 0.3), ("hate", 0.4)]
            1) don't hate -> max(0.3, 0) -> 0.3
            2) hate -> max(0.3, 0.4) -> 0.4
            result=[0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0]

        This method should be overriden if your DataExplainer is not well
        suited for this merging strategy.

        Args:
            text (str): a text
            label (int): predicted label for the `text`
            label_scores (List[Tuple[str, float]]): list of span scores for the `label`

        Returns:
            AttributedSample: attributed sample with `text`, `pred_label`, and `char_scores`.
        """

        lower_text = text.lower()
        char_scores = np.zeros(len(text))
        for span, score in label_scores:
            matches = [
                (match.start(), match.end())
                for match in re.finditer(re.escape(span), lower_text)
            ]
            for start, end in matches:
                char_scores[start:end] = max(
                    char_scores[start:end].min(), score
                )

        return AttributedSample(
            text=text,
            tokens=[],
            scores=np.array([]),
            pred_label=label,
            pred_prob=-1.0,
            word_ids=[],
            token_to_chars=[],
            char_scores=char_scores,
        )

    def explain(
        self,
        texts: List[str],
        labels: List[int],
    ) -> List[AttributedSample]:
        """
        Method to compute attributions for a list of texts.
        This method calls to `_get_span_scores` and creates
        attributed sample by calling `_attribute_sample`.

        Args:
            texts (List[str]): list of texts to be explained.
            labels (List[int]): list of labels.

        Returns:
            List[AttributedSample]: a list of attributed samples, each one containing all
                                    the attribution information of each text.
        """
        # Compute scores per label and sort by length to start
        # attributing with the longest and refine with the shortest.
        scores = {
            target_label: sorted(
                self._get_span_scores(texts, labels, target_label),
                key=lambda x: len(x[0]),
                reverse=True,
            )
            for target_label in tqdm(set(labels))
        }

        # Attribute samples
        attributed_samples = [
            self._attribute_sample(text, label, scores[label])
            for text, label in tqdm(zip(texts, labels))
        ]

        return attributed_samples
