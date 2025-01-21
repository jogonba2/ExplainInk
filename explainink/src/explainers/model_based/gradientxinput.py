from typing import Dict, List, Optional

import torch

try:
    from captum.attr import InputXGradient, Saliency
except ImportError:
    print("Captum is not installed, install it if required.")

from ...models import (
    ClassificationModel,
    DualEncoderClassifier,
    TopAttentionFFTDualEncoderClassifier,
)
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class GradientXInputExplainer(ModelExplainer):
    """
    Gradient times input explainer.
    Computes the gradient of a given label w.r.t the input
    and multiplies the gradient by the input if specified.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `multiply_by_inputs`.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.multiply_by_inputs = self.kwargs.get("multiply_by_inputs", True)

    def _get_input_embeds(self, features: dict):
        # Get the input embeddings according to the model
        if isinstance(self.model, DualEncoderClassifier) or isinstance(
            self.model, TopAttentionFFTDualEncoderClassifier
        ):
            return self.model.token_encoder.auto_model.get_input_embeddings()(
                features
            )
        return self.model.encoder.get_input_embeddings()(features)

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:

        with torch.inference_mode():
            logits = self.model.forward(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            ).logits

        pred_probs = logits.softmax(-1).max(-1).values.tolist()
        pred_labels = logits.argmax(-1).tolist()

        # If reference targets are not passed, use the predictions
        if targets is None:
            targets = pred_labels

        def func(inputs_embeds):
            return self.model.forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=features["attention_mask"],
            ).logits

        dl = InputXGradient(func) if self.multiply_by_inputs else Saliency(func)

        inputs_embeds = self._get_input_embeds(features["input_ids"])
        scores = dl.attribute(inputs_embeds, target=targets)
        scores = scores.norm(p=2, dim=-1).detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
