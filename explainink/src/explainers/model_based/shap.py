from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from shap import KernelExplainer
except ImportError:
    print("SHAP is not installed, install it if required.")

from tqdm import tqdm

from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer


class SHAPExplainer(ModelExplainer):
    """
    SHAP explainer.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters.
    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids_np = features["input_ids"].cpu().numpy()
        batch_size = kwargs["batch_size"]
        subsample = 2

        def predict(inputs):
            input_ids, attention_mask = inputs
            with torch.inference_mode():
                output = self.model.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            return output.logits

        def wrapped_predict(input_ids_):
            # Force attention masks to be always one, since internally
            # SHAP modifies the batch dimension of the input ids.
            attention_mask = np.ones_like(input_ids_)
            res = []
            for batch_idx in tqdm(
                range(0, len(input_ids_), batch_size), desc="Batching SHAP"
            ):
                batch_input_ids = input_ids_[batch_idx : batch_idx + batch_size]
                batch_attention_mask = attention_mask[
                    batch_idx : batch_idx + batch_size
                ]
                logits = predict(
                    (
                        torch.LongTensor(batch_input_ids).to(device),
                        torch.LongTensor(batch_attention_mask).to(device),
                    )
                )
                res.append(logits.detach().cpu().numpy())
            return np.vstack(res)

        # Compute probabilities and predictions
        logits = predict((features["input_ids"], features["attention_mask"]))
        pred_probs = logits.softmax(-1).max(-1).values.tolist()
        pred_labels = logits.argmax(-1).tolist()

        # Instantiate explainer, using only `subsample` samples for efficiency
        explainer = KernelExplainer(wrapped_predict, input_ids_np[:subsample])
        explanations = explainer.shap_values(input_ids_np)

        # Get attribution scores
        if targets is None:
            targets = pred_labels
        scores = explanations[np.arange(len(targets)), :, targets]

        return ModelExplainerOutput(
            scores=scores, pred_labels=pred_labels, pred_probs=pred_probs
        )
