import re
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers is not installed, install it if required.")

try:
    from symanto_dec import DeClassifier
except ImportError:
    print("SymantoDec is not installed, install it if required.")

from torch import Tensor as T

from ..common import get_signature_args
from ..types import ClassifierOutput
from . import layers
from .base import ClassificationModel


class TopAttentionFFTDualEncoderClassifier(ClassificationModel):
    """
    Class for Dual Encoder classifiers, just for fully finetuning, with an attention layer on top of the encoder.
    Gradient-based explainers can't be used since there is no classification head.
    The arguments in the config dictionaries must resemble those expected by the `symanto-dec` package.

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

        self.classifier = DeClassifier(
            **get_signature_args(
                DeClassifier.__init__,
                {
                    **self.encoder_params,
                    **self.classifier_params,
                    **self.training_params,
                    **self.inference_params,
                },
            )
        )

        self.classifier.encoder = SentenceTransformer(
            self.encoder_params["pretrained_model_name_or_path"]
        )
        self.tokenizer = self.classifier.encoder.tokenizer

        if self.encoder_params.get("freeze_encoder"):
            self.freeze_encoder()

        # Replace pooling by random initialized attention if using a pretrained model.
        if re.match(
            r".*pool.*", self.classifier.encoder[1]._get_name().lower()
        ):
            attention_class = getattr(
                layers, self.attention_params["class_name"]
            )
            attention = attention_class(
                self.classifier.encoder[0].get_word_embedding_dimension(),
                **self.attention_params,
            )
            self.attention = attention.to(self.classifier.encoder.device)

        # Load attention if using a finetuned TopAttentionDec model
        else:
            self.load(self.encoder_params["pretrained_model_name_or_path"])

        self.classifier.encoder[1] = self.attention
        self.token_encoder = self.classifier.encoder[0]

        # Initialize label embeddings of the classifier
        self.classifier.fit()

    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Implements the `forward` pass of a `TopAttentionFFTDualEncoderClassifier`.
        Differently from HuggingFace models, this `forward` will be not
        used by any trainer and it is implemented just to be used by the
        explainers. Thus, it does not compute the loss and just stores
        stores useful information for the explainers as `hidden_states`.

        Args:
            input_ids (Tensor): input ids of the tokenized samples.
            attention_mask (Optional[Tensor]): attention mask of the tokenized samples.
            labels (Optional[Tensor]): reference labels to be used during training.
            kwargs: any other parameter to be passed to the encoder forward.

        Returns:
            ClassifierOutput: output information of the forward pass.
        """

        # Encoder forward
        encoder_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }

        encoder_output = self.token_encoder.auto_model(**encoder_args)

        # Attention forward
        attention_output = self.attention(
            {
                "token_embeddings": encoder_output.last_hidden_state,
                "attention_mask": attention_mask,
            }
        )

        sentence_embedding = attention_output["sentence_embedding"]

        # Compute the logits using the label embeddings of dec classifier
        scores = None
        hypothesis_encodings = np.vstack(
            list(self.classifier.label2embedding.values())
        )
        if self.classifier.sim_fn == "cos":
            hypothesis_encodings = np.linalg.norm(
                hypothesis_encodings, axis=-1, keepdims=True
            )
        elif self.classifier.sim_fn != "dot":
            raise NotImplementedError(self.classifier.sim_fn)

        hypothesis_encodings = torch.from_numpy(hypothesis_encodings).to(
            sentence_embedding.device
        )

        scores = sentence_embedding @ hypothesis_encodings.T

        return ClassifierOutput(
            loss=None,
            logits=scores,
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

    def fit(self, dataset: Dataset) -> ClassificationModel:
        """
        Override `fit` to use the `fit` from `symanto-dec`.
        This `fit` method harcodes the strategy to be always
        fully-finetuning (`strategy=default`)

        Args:
            dataset (Dataset): a HuggingFace dataset.

        Returns:
            ClassificationModel: trained instance (self).
        """
        self.classifier = self.classifier.fit(
            X=dataset["text"],
            y=dataset["label"],
            strategy="default",
        )
        return self

    def predict_proba(self, dataset: Dataset) -> List[List[float]]:
        """
        Override `predict_proba` to use the `predict_proba` from `symanto-dec`.

        Args:
            dataset (Dataset): a HuggingFace dataset.

        Returns:
            List[List[float]]: predicted probs for each sample in the dataset.
        """
        return self.classifier.predict_proba(dataset["text"]).tolist()

    def predict(self, dataset: Dataset) -> List[int]:
        """
        Override `predict` to use the `predict` from `symanto-dec`.

        Args:
            dataset (Dataset): a HuggingFace dataset.

        Returns:
            List[int]: predicted label for each sample in the dataset.
        """
        return self.classifier.predict(dataset["text"])

    def save(self) -> None:
        """
        Override `save` to save the SentenceTransformer model
        and the attention head for dual encoder classifiers.
        """
        output_dir = self.training_params["output_dir"]
        self.classifier.encoder.save(output_dir)
        torch.save(self.classifier.encoder[1], f"{output_dir}/top_attention.pt")

    def load(self, checkpoint_path: str) -> None:
        """
        Overriden load method to load the top attention layer.

        Args:
            checkpoint_path (str): path where the checkpoint is stored.
        """
        self.attention = torch.load(f"{checkpoint_path}/top_attention.pt")

    def freeze_encoder(self) -> None:
        for param in self.classifier.encoder.parameters():
            param.requires_grad = False
