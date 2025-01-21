from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import safetensors
from datasets import Dataset
from torch import Tensor as T
from torch import nn
from transformers import (
    BatchEncoding,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..types import ClassifierOutput


class ClassificationModel(nn.Module, ABC):
    """
    Base class for classification models.

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
        super().__init__()
        self.encoder_params = encoder_params
        self.tokenizer_params = tokenizer_params
        self.classifier_params = classifier_params
        self.training_params = training_params
        self.inference_params = inference_params
        self.kwargs = kwargs

    @abstractmethod
    def forward(
        self,
        input_ids: T,
        attention_mask: Optional[T] = None,
        labels: Optional[T] = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Forward method of the model.
        This method must be overriden for each new model.

        If this method will be used by the HuggingFace Trainer,
        it must replicate the `forward` structure of HuggingFace models,
        including the loss computation to be used by the Trainer.

        If not, the loss computation is not required, but it is recommended
        to store as many information as possible in the `ClassifierOutput`
        result, to be used by the explainers.

        In any case, this `forward` method will be used by the explainers
        to get the `ClassifierOutput` from which compute token attributions.

        Args:
            input_ids (Tensor): input ids of the tokenized samples.
            attention_mask (Optional[Tensor]): attention mask of the tokenized samples.
            labels (Optional[Tensor]): reference labels to be used during training.
            kwargs: any other parameter to be passed to the encoder forward.

        Returns:
            ClassifierOutput: output information of the forward pass.
        """
        ...

    def compute_loss(
        self, logits: T, labels: Optional[T] = None
    ) -> Optional[T]:
        """
        Computes the `CrossEntropy` loss if `labels` are passed.
        Args:
            logits (Tensor): logits of the model (last model output)
            labels (torch.Tensor): labels
        Returns:
            Tensor: the `CrossEntropy` loss.
        """
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.classifier_params["num_labels"]),
                labels.view(-1),
            )
        return loss

    def predict_proba(self, dataset: Dataset) -> List[List[float]]:
        """
        Predicts class probabilities for all the samples in `dataset`.
        Override this method if a new model does not complies the HuggingFace interface.

        Args:
            dataset (Dataset): a HuggingFace dataset.

        Returns:
            List[List[float]]: predicted probs for each sample in the dataset.
        """
        # Tokenize dataset
        dataset = self.tokenize(dataset)

        inference_args = TrainingArguments(
            do_predict=True, **self.inference_params
        )

        trainer = Trainer(
            model=self,
            tokenizer=self.tokenizer,
            args=inference_args,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        preds = trainer.predict(test_dataset=dataset).predictions

        if isinstance(preds, tuple):
            preds = preds[0]

        return preds.tolist()

    def predict(self, dataset: Dataset) -> List[int]:
        """
        Predicts class labels for all the samples in `dataset`.
        Override this method if a new model does not complies the HuggingFace interface.

        Args:
            dataset (Dataset): a HuggingFace dataset.
        Returns:
            List[int]: list of predicted labels.
        """
        preds = np.array(self.predict_proba(dataset))
        pred_labels = preds.argmax(axis=-1)
        return pred_labels

    def fit(self, dataset: Dataset) -> "ClassificationModel":
        """
        Fits the classifier on the `dataset`.
        Override this method if a new model does not complies the HuggingFace interface.

        Args:
            dataset (Dataset): dataset from HF datasets.

        Returns:
            ClassificationModel: trained instance (self).
        """
        # Show a model summary
        print(self.model_summary())

        # Prepare the dataset
        dataset = self.tokenize(dataset)

        dataset = dataset.rename_column("label", "labels")

        training_args = TrainingArguments(do_train=True, **self.training_params)
        trainer = Trainer(
            model=self,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        trainer.train()
        return self

    def model_summary(self) -> Dict:
        """
        Gets a summary of the model: number of parameters and model size.

        Returns:
            Dict: dictionary with the model information.
        """
        stats = {}
        param_size, num_params = 0, 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            num_params += param.nelement()
        stats["num_params"] = float(num_params)
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        stats["model_size (MB)"] = (param_size + buffer_size) / 1024**2
        return stats

    def tokenize(
        self,
        dataset: Dataset,
        columns_to_remove: Optional[List[str]] = None,
        **kwargs,
    ) -> Dataset:
        """
        Tokenizes the texts in a `dataset`.

        Args:
            dataset (Dataset): a HuggingFace dataset.
            columns_to_remove (Optional[List[str]]): columns to remove after tokenizing.
            kwargs (Dict): any other parameter to the tokenizer.

        Returns:
            Dataset: a tokenized dataset.
        """
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.tokenizer_params.get("max_seq_length", None),
                **kwargs,
            ),
            batched=True,
        )
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        return dataset

    def tokenize_texts(self, texts: List[str], **kwargs) -> BatchEncoding:
        """
        Tokenizes a list of texts.

        Args:
            texts (List[str]): list of texts.
            kwargs (Dict): any other parameter to the tokenizer.

        Returns:
            BatchEncoding: tokenizer output.
        """
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.tokenizer_params.get("max_seq_length", None),
            **kwargs,
        )

    def load(self, checkpoint_path: str) -> None:
        """
        Loads the weights of all the classifier layers.
        This base method, as it is, can be used only with HuggingFace-based models.
        For SentenceTransformer-based models, this method must be overriden.

        Args:
            checkpoint_path (str): path where the checkpoint is stored.
        """
        safetensors.torch.load_model(self, checkpoint_path)

    def save(self) -> None:
        """
        Saves the model weights.
        This method must be overriden when some layers of the models
        are not stored by default like in Dual Encoders with top attention layers.
        """
        ...

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder weights.
        This method must be overriden for each new model if freezing is allowed.
        """
        ...

    def _cast_label_dict(self) -> None:
        """
        Casts the labels in the verbalization `label2text` dict for dual encoder models.
        ExplainInk always expects the labels to be integers, if not,
        `_cast_labels` will cast the labels (not the verbalizations).
        """
        self.classifier_params["label2text"] = {
            int(c): v for c, v in self.classifier_params["label2text"].items()
        }
