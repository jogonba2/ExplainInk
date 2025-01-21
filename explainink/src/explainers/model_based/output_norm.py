from typing import Dict, List, Optional

import torch

from ...common import InvalidModelForExplainer, get_torch_aggregation
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class OutputNormExplainer(ModelExplainer):
    """
    Output norm explainer.
    Averages the L2 norm of the hidden states of the specified layers (all by default).

    Can only be used with models whose `forward` method returns `hidden_states`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `layers`.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.aggregation = get_torch_aggregation(
            self.kwargs.get("aggregation", "mean")
        )
        self.norm_type = self.kwargs.get("norm_type", "fro")
        self.layers = self.kwargs.get("layers", None)

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        with torch.inference_mode():
            output = self.model.forward(
                **features,
                output_hidden_states=True,
            )

        hidden_states = output.hidden_states

        # Check whether the explainer can be applied
        if hidden_states is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        # Compute predictions
        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # Reshape hidden states from l tensors of (b, t, d) to (b, l, t, d)
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = torch.einsum("lbtd->bltd", hidden_states)

        # Select layers if specified
        if self.layers is not None:
            hidden_states = hidden_states[:, self.layers, ...]

        # Compute the average of the hidden states' norm
        scores = self.aggregation(
            hidden_states.norm(dim=-1, p=self.norm_type), dim=1
        )

        # To numpy array
        scores = scores.detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
