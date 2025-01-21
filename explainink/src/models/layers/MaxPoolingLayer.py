import json
import os
from typing import Dict

import torch
from torch import Tensor as T
from torch import nn


class MaxPoolingLayer(nn.Module):
    """
    Class for interpretable MaxPooling. The attention scores
    of max pooling comes from counting and normalizing the
    number of times that a token contributes with a max
    value to the max pooling output.

    Attributes:
        input_dim (int): input dimension.
        mask_value (int): value for masking special tokens.
        mask_special_tokens (bool): whether to mask special tokens (CLS and SEP)
    """

    def __init__(
        self,
        input_dim: int,
        mask_value: int = 0,
        mask_special_tokens: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask_value = mask_value
        self.mask_special_tokens = mask_special_tokens
        self.kwargs = kwargs

        self.config_keys = [
            "input_dim",
            "mask_value",
            "mask_special_tokens",
            "kwargs",
        ]

    def forward(self, features: Dict[str, T]) -> Dict[str, T]:
        x = features["token_embeddings"]
        attention_mask = features["attention_mask"].clone()

        # Mask special tokens [CLS] and [SEP] if specified (remove from pooling)
        if self.mask_special_tokens:
            sep_cols = (torch.sum(attention_mask, axis=1) - 1).unsqueeze(-1)
            rows = torch.arange(len(sep_cols)).unsqueeze(1)
            # Mask [SEP]
            attention_mask[rows, sep_cols] = 0
            # Mask [CLS]
            attention_mask[:, 0] = 0

        # Pooling
        pooled = x.max(axis=1)
        pooled_vals = pooled.values
        pooled_inds = pooled.indices

        # Compute bin counts (B, L)
        N = x.shape[1]
        ids = (
            pooled_inds
            + (N * torch.arange(pooled_inds.shape[0], device=x.device))[:, None]
        )

        unnorm_scores = torch.bincount(
            ids.ravel(), minlength=N * pooled_inds.shape[0]
        ).reshape(-1, N)

        # Mask to 0 the counts of [PAD], and [CLS], [SEP] if specified
        masked_unnorm_scores = unnorm_scores.masked_fill(
            (1.0 - attention_mask).bool(), 0
        )

        # Normalize scores scores / sum(scores)
        scores = masked_unnorm_scores / masked_unnorm_scores.sum(
            dim=-1
        ).unsqueeze(-1)

        return {
            "scores": scores,
            "sentence_embedding": pooled_vals,
        }

    def get_config_dict(self) -> Dict:
        """
        Gets the config keys to be stored by SentenceTransformer models.

        Returns:
            Dict: config keys.
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str) -> None:
        """
        Saves the config dict to be stored by SentenceTransformer models.

        Args:
            output_path (str): path where to store the config.
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str) -> "MaxPoolingLayer":
        """
        Instantiates the attention layer using the config keys.
        This method is implemented for compatibility with SentenceTransformers.

        Args:
            input_path (str): path where the config is stored.

        Returns:
            MaxPoolingLayer: an instance of MaxPoolingLayer.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return MaxPoolingLayer(**config)
