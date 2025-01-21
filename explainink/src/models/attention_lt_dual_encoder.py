from typing import Dict, List, Optional

import torch
from torch import Tensor as T
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..common import get_signature_args
from ..types import ClassifierOutput
from . import layers
from .base import ClassificationModel
from .losses import attention_kl_loss, conicity_loss


class ClassificationHead(nn.Module):
    """
    Classification head.

    Attributes:
        input_dim (int): input dimension.
        num_labels (int): num labels (output dimension).
        dropout (float): classifier dropout.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, num_labels, bias=False)
        self.kwargs = kwargs

    def forward(self, hidden_state: T):
        x = self.dropout(hidden_state)
        logits = self.out_proj(x)
        return logits


class TopAttentionLTDualEncoderClassifier(ClassificationModel):
    """
    Class for Dual Encoder classifiers, just for label tuning, with an attention layer on top of the encoder.

    This class simulates label tuning from `symanto-dec` using HuggingFace models, by preparing a model:
    `encoder` (B, L, D) -> `top attention` (B, D) -> `weights initialized with label embeddings` (B, C)

    where the `encoder` is frozen, and only the `top attention` and the `label embeddings` are trained.

    This simulation is useful to add top attention, new losses for training, etc, since this is not
    possible (or easily) to modify in the `symanto-dec` package.

    These models can be later explained by any explainer except `TopAttentionExplainer`.
    The arguments in the config dictionaries must resemble those expected by HuggingFace.

    Attributes:
        encoder_params (Dict): parameters for the encoder.
        tokenizer_params (Dict): parameters for the tokenizer.
        classifier_params (Dict): parameters for the classifier.
        attention_params (Dict): parameters for the top attention layer.
        training_params (Dict): parameters to be used by the `Trainer` during training.
        inference_params (Dict): parameters to be used by the `Trainer` during inference.
    """

    def __init__(
        self,
        encoder_params: Dict,
        tokenizer_params: Dict,
        classifier_params: Dict,
        attention_params: Dict,
        training_params: Dict,
        inference_params: Dict,
        **kwargs,
    ):
        super().__init__(
            encoder_params,
            tokenizer_params,
            classifier_params,
            training_params,
            inference_params,
            **kwargs,
        )

        self._cast_label_dict()
        self.attention_params = attention_params

        self.encoder = AutoModel.from_pretrained(**self.encoder_params)
        self.tokenizer = AutoTokenizer.from_pretrained(**self.encoder_params)

        # Encoder frozen by design
        self.freeze_encoder()

        attention_class = getattr(layers, self.attention_params["class_name"])
        self.attention = attention_class(
            self.encoder.config.hidden_size, **self.attention_params
        )
        self.classifier = ClassificationHead(
            self.encoder.config.hidden_size,
            **self.classifier_params,
        )
        # Initialize the weights of the classifer to meaningful units
        self.classifier.out_proj.weight = self._get_label_embeddings(
            list(self.classifier_params["label2text"].values())
        )

    def _get_label_embeddings(self, labels: List[str]) -> nn.Parameter:
        """
        Get the label embeddings from the encoder to be used as
        initialization of the last projection layer.

        Args:
            labels (List[str]): list of label verbalizations.

        Returns:
            Parameter: torch trainable parameters.
        """
        tok_labels = self.tokenize_texts(
            labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            pooled_output = self.encoder(**tok_labels).pooler_output
        return nn.Parameter(pooled_output)

    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Implements the `forward` pass of a `TopAttentionLTDualEncoderClassifier`.
        This `forward` method resembles the one used by HuggingFace to be used
        by the `Trainer` both for training and inference. Thus, computes the loss
        when labels are available (training), and stores useful information
        for the explainers as `hidden_states` or `attentions`.

        Args:
            input_ids (Tensor): input ids of the tokenized samples.
            attention_mask (Optional[Tensor]): attention mask of the tokenized samples.
            labels (Optional[Tensor]): reference labels to be used during training.
            kwargs: any other parameter to be passed to the encoder forward.

        Returns:
            ClassifierOutput: output information of the forward pass.
        """

        model_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }

        # Embedder forward
        embeddings = None
        if self.attention_params.get("add_embeddings", None):
            embedder_args = get_signature_args(
                self.encoder.embeddings.forward, model_args
            )
            embeddings = self.encoder.embeddings(**embedder_args)

        # Encoder forward
        encoder_args = get_signature_args(self.encoder.forward, model_args)
        encoder_output = self.encoder(**encoder_args)

        # Attention forward
        attention_output = self.attention(
            {
                "token_embeddings": encoder_output.last_hidden_state,
                "attention_mask": attention_mask,
                "embeddings": embeddings,
            }
        )

        # Classifier forward
        logits = self.classifier(attention_output["sentence_embedding"])

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Maximize KL divergence between uniform and attention scores
        if self.attention_params.get("attention_kl_loss") and loss:
            loss += attention_kl_loss(attention_output["scores"], loss)

        # Minimize the conicity
        if self.attention_params.get("conicity_loss") and loss:
            loss += conicity_loss(encoder_output.last_hidden_state)

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=(
                encoder_output.attentions
                if "output_attentions" in encoder_args
                else None
            ),
            hidden_states=(
                encoder_output.hidden_states
                if "output_hidden_states" in encoder_args
                else None
            ),
            classifier_attention=attention_output["scores"],
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
