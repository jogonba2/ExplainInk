from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset

try:
    from symanto_dec import DeClassifier
except ImportError:
    print("SymantoDec is not installed, install it if required.")

from sentence_transformers.models import Dense, Normalize
from torch import Tensor as T

from ..common import get_signature_args
from ..types import ClassifierOutput
from .base import ClassificationModel


class DualEncoderClassifier(ClassificationModel):
    """
    Class for vanilla Dual Encoder models from `symanto-dec` (both fully finetune and label tuning)
    The arguments in the config dictionaries must resemble those expected by the `symanto-dec` package.

    Attributes:
        encoder_params (Dict): parameters for the encoder.
        tokenizer_params (Dict): parameters for the tokenizer.
        classifier_params (Dict): parameters for the classifier.
        training_params (Dict): parameters to be used during training.
        inference_params (Dict): parameters to be used by during inference.
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
        self._cast_label_dict()

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

        self.classifier.fit()

        # Load sentence transformer components
        self.tokenizer = self.classifier.encoder.tokenizer
        self.token_encoder = self.classifier.encoder[0]
        self.pooling = self.classifier.encoder[1]
        self.dense, self.normalizer = None, None
        for module in self.classifier.encoder._modules.values():
            if isinstance(module, Dense):
                self.dense = module
            if isinstance(module, Normalize):
                self.normalizer = module

    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Implements the `forward` pass of a `DualEncoderClassifier`.
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

        # Pool token embeddings
        sentence_embedding = self.pooling(
            {
                "token_embeddings": encoder_output.last_hidden_state,
                "attention_mask": attention_mask,
            }
        )["sentence_embedding"]

        # Apply dense if required
        if self.dense:
            sentence_embedding = self.dense(
                {"sentence_embedding": sentence_embedding}
            )["sentence_embedding"]

        # Normalize if required
        if self.normalizer:
            sentence_embedding = self.normalizer(
                {"sentence_embedding": sentence_embedding}
            )["sentence_embedding"]

        # Compute the logits using the label embeddings of dec classifier
        scores = None
        hypothesis_encodings = np.vstack(
            list(self.classifier.label2embedding.values())
        )
        if self.classifier.sim_fn == "cos":
            hypothesis_encodings /= np.linalg.norm(
                hypothesis_encodings, axis=-1, keepdims=True
            )
        elif self.classifier.sim_fn != "dot":
            raise NotImplementedError(self.classifier.sim_fn)

        hypothesis_encodings = (
            torch.from_numpy(hypothesis_encodings)
            .to(sentence_embedding.device)
            .float()
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
            classifier_attention=None,
            label_embeddings=hypothesis_encodings,
        )

    def fit(self, dataset: Dataset) -> ClassificationModel:
        """
        Override `fit` to use the `fit` from `symanto-dec`.

        Args:
            dataset (Dataset): a HuggingFace dataset.

        Returns:
            ClassificationModel: trained instance (self).
        """
        self.classifier = self.classifier.fit(
            X=dataset["text"],
            y=dataset["label"],
            strategy=self.training_params["strategy"],
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
        Override `save` to save the SentenceTransformer
        model for dual encoder classifiers.
        """
        self.classifier.encoder.save(self.training_params["output_dir"])

    def load(self, checkpoint_path: str) -> None:
        """
        Overriden load method to avoid loading from checkpoint, since
        (i) the whole checkpoint can be loaded directly in the constructor
        from `encoder_params["model_name_or_path"]`, and (ii) SentenceTransformers
        do not allow to load weights from safetensors as implemented
        in the base `load` method.
        """
        ...
