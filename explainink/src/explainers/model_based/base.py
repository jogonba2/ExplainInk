from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ...common import batch_to_device
from ...common.utils import spacy_pipeline
from ...models import ClassificationModel
from ...types import AttributedSample, ModelExplainerOutput
from .utils import (
    build_char_scores,
    postprocess_scores,
    propagate_scores_to_same_word,
    remove_special_tokens,
)


class ModelExplainer(ABC):
    """
    Base class for model-based explainers.
    The explainers are in charge of creating attributed samples
    by using trained models.

    Attributes:
        model (ClassificationModel): a classification model.
        sentence_level (bool): whether to compute attributions at sentence level (True) or at document-level (False).
        language (str): language of the texts.
        mask_scores (Dict): what masking postprocesses make.
        device (str): device where to put the tensors.
    """

    def __init__(
        self,
        model: ClassificationModel,
        sentence_level: bool = False,
        language: str = "en",
        mask_scores: Dict = {
            "stopwords": False,
            "punctuation": False,
            "shorter": -1,
        },
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.sentence_level = sentence_level
        self.language = language
        self.mask_scores = mask_scores
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.kwargs = kwargs

    @abstractmethod
    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        """
        Method to compute attributions for a batch of samples from its features.
        This method must be implemented by each new explainer.

        Args:
            features (Dict): features of a batch of samples (input_ids, attention_mask, etc.)
            targets (Optional[List[int]]): a list of target labels to compute attributions
                                           (e.g. in gradient-based explainers)

        Returns:
            ExplainerOutput: output containing token scores, pred labels, and
                             pred probs for the samples in the batch.
        """
        ...

    def explain(
        self,
        texts: List[str],
        targets: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> List[AttributedSample]:
        """
        Method to compute attributions for a list of texts.
        This method calls to `_explain` with batches of tokenized texts,
        and finally applies a list of postprocesses:

        1) Always removes special tokens ([CLS], [SEP], etc.)
        2) Make zero the scores of stopwords, punctuation, etc. if specified in `self.mask_scores`.
        3) Propagates scores of tokens within the same word.

        Args:
            texts (List[str]): list of texts to be explained.
            targets (Optional[List[int]]): a list of target labels to compute attributions
                                           (e.g. in gradient-based explainers)
            batch_size (int): batch size.

        Returns:
            List[AttributedSample]: a list of attributed samples, each one containing all
                                    the attribution information of each text.
        """
        # Split in sentences if specified to work at sentence-level.
        if self.sentence_level:
            docs = spacy_pipeline(texts, language=self.language)
            texts = sum([[sent.text for sent in doc.sents] for doc in docs], [])

        # Tokenize the dataset's text
        dataset = self.model.tokenize(
            Dataset.from_dict({"text": texts}), columns_to_remove=["text"]
        )

        # Get word ids and token_to_chars
        tokenizations = self.model.tokenize_texts(texts)
        word_ids = [tokenizations.word_ids(i) for i in range(len(texts))]
        tokens = [tokenizations.tokens(i) for i in range(len(texts))]
        token_to_chars = [
            [tokenizations.token_to_chars(i, j) for j in range(len(tokens[i]))]
            for i in range(len(texts))
        ]

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(self.model.tokenizer),
        )
        attributed_samples = []

        for batch_idx, batch in tqdm(
            enumerate(data_loader), desc="Explaining..."
        ):
            # Get batch texts and labels
            batch_texts = texts[
                batch_idx * batch_size : batch_idx * batch_size + batch_size
            ]
            batch_word_ids = word_ids[
                batch_idx * batch_size : batch_idx * batch_size + batch_size
            ]
            batch_token_to_chars = token_to_chars[
                batch_idx * batch_size : batch_idx * batch_size + batch_size
            ]

            batch_targets = None

            if targets is not None and not self.sentence_level:
                batch_targets = targets[
                    batch_idx * batch_size : batch_idx * batch_size + batch_size
                ]

            # Compute attributions
            batch = batch_to_device(batch, self.device)
            output = self._explain(batch, batch_targets, batch_size=batch_size)

            # Pack attributted samples
            for (
                text,
                sample_scores,
                sample_pred_label,
                sample_pred_prob,
                attention_mask,
                input_ids,
                sample_word_ids,
                sample_token_to_chars,
            ) in zip(
                batch_texts,
                output.scores,
                output.pred_labels,
                output.pred_probs,
                batch["attention_mask"],
                batch["input_ids"],
                batch_word_ids,
                batch_token_to_chars,
            ):
                att_mask = attention_mask.detach().cpu().numpy()

                sample_tokens = self.model.tokenizer.convert_ids_to_tokens(
                    input_ids
                )

                sample_tokens = [
                    token
                    for idx, token in enumerate(sample_tokens)
                    if att_mask[idx] == 1
                ]

                sample_word_ids = [
                    word_id
                    for idx, word_id in enumerate(sample_word_ids)
                    if idx < len(att_mask) and att_mask[idx] == 1
                ]

                sample_token_to_chars = [
                    span
                    for idx, span in enumerate(sample_token_to_chars)
                    if idx < len(att_mask) and att_mask[idx] == 1
                ]

                sample_scores = sample_scores[np.where(att_mask == 1)]

                attributed_samples.append(
                    AttributedSample(
                        text=text,
                        tokens=sample_tokens,
                        scores=sample_scores,
                        pred_label=sample_pred_label,
                        pred_prob=sample_pred_prob,
                        word_ids=sample_word_ids,
                        token_to_chars=sample_token_to_chars,
                        char_scores=None,
                    )
                )

        # Remove special tokens always
        attributed_samples = remove_special_tokens(
            attributed_samples, self.model.tokenizer.special_tokens_map.values()
        )

        # Apply score postprocessing if specified
        attributed_samples = postprocess_scores(
            attributed_samples, self.language, self.mask_scores
        )

        # Propagate scores of tokens within the same word
        attributed_samples = propagate_scores_to_same_word(attributed_samples)

        # Compute char scores
        attributed_samples = build_char_scores(attributed_samples)

        return attributed_samples
