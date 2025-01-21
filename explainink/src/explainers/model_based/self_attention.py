from typing import Dict, List, Optional

import torch

from ...common import InvalidModelForExplainer, get_torch_aggregation
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class SelfAttentionExplainer(ModelExplainer):
    """
    Self attention explainer.

    Averages the multi-head self-attention matrices (softmax) across all
    the specified layers (all by default).

    Can only be used with models whose `forward` method returns `attentions`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `layers`.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.aggregation = get_torch_aggregation(
            self.kwargs.get("aggregation", "mean")
        )
        self.layers = self.kwargs.get("layers", None)

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        with torch.inference_mode():
            output = self.model.forward(**features, output_attentions=True)

        attentions = output.attentions

        # Check whether the explainer can be applied
        if attentions is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        # Compute predictions
        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # From l tensors of (b, h, t, t) to (b, l, h, t, t)
        attentions = torch.stack(attentions, dim=0)
        attentions = torch.einsum("lbhij->blhij", attentions)

        # Select layers if specified
        if self.layers is not None:
            attentions = attentions[:, self.layers, ...]

        # Average across heads, across layers, and per token
        scores = self.aggregation(
            self.aggregation(self.aggregation(attentions, dim=2), dim=1), dim=1
        )

        # To numpy array
        scores = scores.detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
