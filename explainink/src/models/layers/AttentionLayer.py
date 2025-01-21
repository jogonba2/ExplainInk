import json
import os
from typing import Dict

import torch
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F


class AttentionLayer(nn.Module):
    """
    Class for simple, incontextual attention, defined as:

    ```
    x_i = h_i if embeddings is None otherwise h_i + embedding_i
    e_i = tanh(W_e @ x_i + b_e)
    a_i = exp(e_i) / sum(exp(e_j))
    h = sum(x_i * a_i)
    ```

    Note that it is incontextual since each h_i is not related with the
    others, instead it is just projected to an scalar that multiplies h_i.
    This simplification assumes that each h_i has been already contextualized
    in previous layers of the models.

    Attributes:
        input_dim (int): input dimension.
        mask_value (int): value for masking special tokens.
        mask_special_tokens (bool): whether to mask special tokens (CLS and SEP)
    """

    def __init__(
        self,
        input_dim: int,
        mask_value: int = -100,
        mask_special_tokens: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.w_e = nn.Linear(self.input_dim, 1)
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
        embeddings = features.get("embeddings")

        if embeddings is not None:
            x = x + embeddings
        e = F.tanh(self.w_e(x))

        # Mask special tokens [CLS] and [SEP] if specified.
        if self.mask_special_tokens:
            sep_cols = (torch.sum(attention_mask, axis=1) - 1).unsqueeze(-1)
            rows = torch.arange(len(sep_cols)).unsqueeze(1)
            # Mask [SEP]
            attention_mask[rows, sep_cols] = 0
            # Mask [CLS]
            attention_mask[:, 0] = 0

        masked_e = e.masked_fill(
            (1.0 - attention_mask.unsqueeze(-1)).bool(), self.mask_value
        )

        a = F.softmax(masked_e, dim=1)

        return {
            "scores": a.squeeze(-1),
            "sentence_embedding": torch.sum(a * x, dim=1),
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
    def load(input_path: str) -> "AttentionLayer":
        """
        Instantiates the attention layer using the config keys.
        This method is implemented for compatibility with SentenceTransformers.

        Args:
            input_path (str): path where the config is stored.

        Returns:
            AttentionLayer: an instance of AttentionLayer.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return AttentionLayer(**config)
