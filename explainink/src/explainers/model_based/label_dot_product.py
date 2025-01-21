from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ...common import InvalidModelForExplainer, get_torch_aggregation
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class AggregatedLabelDotProductExplainer(ModelExplainer):
    """
    Label dot product explainer.

    Averages token embedding across the layers and computes the dot
    product between each token and the predicted/target label embedding.

    Specially dedicated to explain FSL (Label tuning) models.

    Can only be used with models whose `forward` method
    returns `label_embeddings` and `hidden_states`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `layers`.
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
            output = self.model.forward(
                **features,
                output_hidden_states=True,
            )

        # Check if the explainer can be computed
        if output.hidden_states is None or output.label_embeddings is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        hidden_states = output.hidden_states
        label_embeddings = output.label_embeddings.clone()

        # Compute predictions
        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # Reshape hidden states from l tensors of (b, t, d) to (b, l, t, d)
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = torch.einsum("lbtd->bltd", hidden_states)

        # Compute aggregation of hidden states
        token_embeddings = self.aggregation(hidden_states, dim=1)

        # Normalize token and label embeddings
        token_embeddings /= torch.linalg.norm(
            token_embeddings, dim=-1, keepdims=True
        )
        label_embeddings /= torch.linalg.norm(
            label_embeddings, dim=-1, keepdims=True
        )

        # If targets are provided, pick the embeddings of the targets
        # Otherwise, pick the predicted labels
        if targets:
            target_embeddings = label_embeddings[targets]
        else:
            target_embeddings = label_embeddings[pred_labels]

        # Dirty trick for models with a dense layer on top of the sentence embedding
        # that projects the sentence embedding to a different dimensionality. In this
        # case, products between token embeddings and label embeddings can not be computed
        # since the shape of label embeddings match the shape of sentence embeddings
        # (after applying the dense layer), but do not match the shape of the token
        # embeddings (before applying the dense layer). Trick: add zero padding.
        padding_needed = (
            0,
            target_embeddings.shape[-1] - token_embeddings.shape[-1],
        )
        token_embeddings = F.pad(
            token_embeddings, padding_needed, "constant", 0
        )

        # Compute the scores of each token by multiplying by the target embeddings
        scores = torch.einsum("btd,bd->bt", token_embeddings, target_embeddings)

        # To numpy
        scores = scores.detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )


class LayerWiseLabelDotProductExplainer(ModelExplainer):
    """
    Label dot product explainer.

    Computes the dot product between the embedding of the predicted/target label
    and each token embedding across layers. The final scores are the aggregation
    of all these scores.

    Specially dedicated to explain FSL (Label tuning) models.

    Can only be used with models whose `forward` method
    returns `label_embeddings` and `hidden_states`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters, e.g. `layers`.
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
            output = self.model.forward(
                **features,
                output_hidden_states=True,
            )

        # Check if the explainer can be computed
        if output.hidden_states is None or output.label_embeddings is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        hidden_states = output.hidden_states
        label_embeddings = output.label_embeddings.clone()

        # Compute predictions
        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # Reshape hidden states from l tensors of (b, t, d) to (b, l, t, d)
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = torch.einsum("lbtd->bltd", hidden_states)

        # Normalize hidden states and label embeddings
        hidden_states /= torch.linalg.norm(hidden_states, dim=-1, keepdims=True)
        label_embeddings /= torch.linalg.norm(
            label_embeddings, dim=-1, keepdims=True
        )

        # If targets are provided, pick the embeddings of the targets
        # Otherwise, pick the predicted labels
        if targets:
            target_embeddings = label_embeddings[targets]
        else:
            target_embeddings = label_embeddings[pred_labels]

        # Dirty trick for models with a dense layer on top of the sentence embedding
        # that projects the sentence embedding to a different dimensionality. In this
        # case, products between token embeddings and label embeddings can not be computed
        # since the shape of label embeddings match the shape of sentence embeddings
        # (after applying the dense layer), but do not match the shape of the token
        # embeddings (before applying the dense layer). Trick: add zero padding.
        padding_needed = (
            0,
            target_embeddings.shape[-1] - hidden_states.shape[-1],
        )
        hidden_states = F.pad(hidden_states, padding_needed, "constant", 0)

        # Reshape target embeddings to avoid broadcastings
        target_embeddings = target_embeddings[:, None, None, :]

        # Compute the scores at each token position across layers for the target embeddings
        scores = torch.einsum(
            "bltd,byyd->blt", hidden_states, target_embeddings
        )

        # Aggregate scores per layer
        scores = self.aggregation(scores, dim=1)

        # To numpy
        scores = scores.detach().cpu().numpy()

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
