from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CharSpan
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class CorrelationOutput:
    """
    Wrapper for the outputs of correlation evaluators.
    """

    token_output: Dict[str, float]
    char_output: Dict[str, float]

    def serialize(self):
        return {
            "token_output": self.token_output,
            "char_output": self.char_output,
        }


@dataclass
class ModelExplainerOutput:
    """
    Wrapper for the outputs of model-based explainers.
    """

    scores: np.ndarray
    pred_labels: List[int]
    pred_probs: List[float]


@dataclass
class ClassifierOutput(SequenceClassifierOutput):
    """
    Wrapper for the outputs of the models' forward pass.
    """

    classifier_attention: Optional[torch.FloatTensor] = None
    label_embeddings: Optional[torch.FloatTensor] = None


@dataclass
class AttributedSample:
    """
    Wrapper to store all the information of an attributed sample.
    """

    text: str
    tokens: List[str]
    scores: np.ndarray
    pred_label: int
    pred_prob: float
    word_ids: List[int]
    token_to_chars: List[CharSpan]
    char_scores: np.ndarray
    true_label: int = -100

    def __post_init__(self):
        if len(self.tokens) != self.scores.shape[0]:
            raise ValueError(
                f"Invalid shapes between tokens and scores"
                f" {len(self.tokens)} and {self.scores.shape[0]}"
            )

    def serialize(self):
        return {
            "text": self.text,
            "tokens": self.tokens,
            "scores": self.scores.tolist(),
            "pred_label": self.pred_label,
            "pred_prob": self.pred_prob,
            "word_ids": self.word_ids,
            "token_to_chars": [span._asdict() for span in self.token_to_chars],
            "char_scores": self.char_scores.tolist(),
            "true_label": self.true_label,
        }

    @classmethod
    def load(
        cls,
        text: str,
        tokens: List[str],
        scores: Union[List, np.ndarray],
        pred_label: int,
        pred_prob: float,
        word_ids: List[int],
        token_to_chars: Union[List[Dict], List[CharSpan]],
        char_scores: Union[List, np.ndarray],
        true_label: int = -100,
    ):
        if isinstance(scores, list):
            scores = np.array(scores)
        if isinstance(char_scores, list):
            char_scores = np.array(char_scores)
        if isinstance(token_to_chars[0], dict):
            token_to_chars = [
                CharSpan(start=span["start"], end=span["end"])
                for span in token_to_chars
            ]

        return cls(
            text=text,
            tokens=tokens,
            scores=scores,
            pred_label=pred_label,
            pred_prob=pred_prob,
            word_ids=word_ids,
            token_to_chars=token_to_chars,
            char_scores=char_scores,
            true_label=true_label,
        )
