from typing import Dict, List, Optional

import torch

from ...common import InvalidModelForExplainer, get_torch_aggregation
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class AttentionRolloutExplainer(ModelExplainer):
    """
    Attention Rollout from https://arxiv.org/pdf/2005.00928

    Look at https://jacobgil.github.io/deeplearning/vision-transformer-explainability#q--k--v-and-attention
    for a better intuition of Attention Rollout

    Can only be used with models whose `forward` method returns `attentions`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters.

    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.aggregation = get_torch_aggregation(
            self.kwargs.get("aggregation", "mean")
        )

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        with torch.inference_mode():
            output = self.model.forward(**features, output_attentions=True)

        attentions = output.attentions

        if attentions is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # From l tensors of (b, h, t, t) to (b, l, h, t, t)
        attentions = torch.stack(attentions, dim=0)
        attentions = torch.einsum("lbhij->blhij", attentions)

        # Fuse attention heads in each layer (see Section 3 of the paper)
        attentions = self.aggregation(attentions, dim=2)

        # Add identity matrix to the raw attentions to include residual connections in the flow
        attentions = (
            attentions
            + torch.eye(attentions.shape[-1])
            .expand(*attentions.shape)
            .to(attentions.device)
        ) / 2

        # Normalize the attentions so each row sums 1
        attentions = attentions / attentions.sum(-1, keepdim=True)

        # Compute Eq 1, that is basically the cumulative product of the attentions
        rollout = (
            torch.eye(attentions.shape[-1])
            .expand(attentions.shape[0], *attentions.shape[2:])
            .to(attentions.device)
        )

        for layer in range(attentions.shape[1]):
            rollout = attentions[:, layer, ...] @ rollout

        # Compute the mean per token, which are the final scores (B, T)
        scores = rollout.mean(dim=1).detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
