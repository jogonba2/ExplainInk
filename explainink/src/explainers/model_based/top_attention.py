from typing import Dict, List, Optional

import torch

from ...common import InvalidModelForExplainer
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class TopAttentionExplainer(ModelExplainer):
    """
    Top attention explainer.
    Extracts the scores from the top attention layer.

    Can only be used with models that include an attention layer in top of the encoder
    and return the scores of the attention layer in `classifier_attention`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        with torch.inference_mode():
            output = self.model.forward(**features)

        classifier_attention = output.classifier_attention

        if classifier_attention is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        scores = classifier_attention.detach().cpu().numpy()
        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        return ModelExplainerOutput(
            scores=scores, pred_labels=pred_labels, pred_probs=pred_probs
        )
