from typing import Dict, List, Optional

import torch

try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("Captum is not installed, install it if required.")

from ...models import (
    ClassificationModel,
    DualEncoderClassifier,
    TopAttentionFFTDualEncoderClassifier,
)
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class IntegratedGradientsExplainer(ModelExplainer):
    """
    Integrated gradients explainer.
    Computes an importance score to each input feature by approximating
    the integral of gradients of the model's output w.r.t the inputs
    along the path from given baselines / references to inputs.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `multiply_by_inputs`.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.multiply_by_inputs = self.kwargs.get("multiply_by_inputs", True)
        self.internal_batch_size = self.kwargs.get("internal_batch_size", 8)
        self.n_steps = self.kwargs.get("n_steps", 5)

    def _get_input_embeds(self, features: dict):
        # Get the input embeddings according to the model
        if isinstance(self.model, DualEncoderClassifier) or isinstance(
            self.model, TopAttentionFFTDualEncoderClassifier
        ):
            return self.model.token_encoder.auto_model.get_input_embeddings()(
                features
            )
        return self.model.encoder.get_input_embeddings()(features)

    def _generate_baselines(self, input_len):
        ids = (
            [self.model.tokenizer.cls_token_id]
            + [self.model.tokenizer.pad_token_id] * (input_len - 2)
            + [self.model.tokenizer.sep_token_id]
        )
        embeddings = self.__get_input_embeds(
            torch.tensor(ids, device=self.device)
        )
        return embeddings.unsqueeze(0)

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
            attention_mask = torch.ones(
                *inputs_embeds.shape[:2], dtype=torch.uint8, device=self.device
            )
            return self.model.forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            ).logits

        dl = IntegratedGradients(
            func, multiply_by_inputs=self.multiply_by_inputs
        )

        # Get the input embeddings according to the model
        inputs_embeds = self._get_input_embeds(features["input_ids"])

        input_len = inputs_embeds.shape[1]
        baselines = self._generate_baselines(input_len)
        baselines = baselines.repeat(inputs_embeds.shape[0], 1, 1)
        scores = dl.attribute(
            inputs_embeds,
            baselines=baselines,
            target=targets,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

        # Pool over hidden size
        scores = scores.norm(p=2, dim=-1).detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
