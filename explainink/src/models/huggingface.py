from typing import Dict, Optional

from torch import Tensor as T
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..common import get_signature_args
from ..types import ClassifierOutput
from .base import ClassificationModel


class HuggingFaceClassifier(ClassificationModel):
    """
    Class for vanilla HuggingFace classifiers (encoder + classification head).
    These models can be later explained by any explainer except `TopAttentionExplainer`.
    The arguments in the config dictionaries must resemble those expected by HuggingFace.

    Attributes:
        encoder_params (Dict): parameters for the encoder.
        tokenizer_params (Dict): parameters for the tokenizer.
        classifier_params (Dict): parameters for the classifier.
        training_params (Dict): parameters to be used by the `Trainer` during training.
        inference_params (Dict): parameters to be used by the `Trainer` during inference.
    """

    def __init__(
        self,
        encoder_params: Dict,
        tokenizer_params: Dict,
        classifier_params: Dict,
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

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            **{**self.encoder_params, **self.classifier_params}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_params["pretrained_model_name_or_path"]
        )

    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Implements the `forward` pass of a `HuggingFaceClassifier`.
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

        # Encoder forward
        model_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        encoder_args = get_signature_args(self.encoder.forward, model_args)
        output = self.encoder(**encoder_args)

        # Compute loss
        loss = self.compute_loss(output.logits, labels)

        return ClassifierOutput(
            loss=loss,
            logits=output.logits,
            attentions=(
                output.attentions
                if "output_attentions" in encoder_args
                else None
            ),
            hidden_states=(
                output.hidden_states
                if "output_hidden_states" in encoder_args
                else None
            ),
            classifier_attention=None,
        )
