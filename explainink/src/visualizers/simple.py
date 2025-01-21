from typing import List

import numpy as np

from ..types import AttributedSample
from .base import Visualizer


class SimpleVisualizer(Visualizer):
    """
    Simple visualizer that assign color to attribution scores
    using background green color with varying opacity.
    """

    def __init__(
        self,
        scaler_fn_name: str = "none",
        token_level: bool = False,
        mask_zeros: bool = True,
    ):
        super().__init__(scaler_fn_name, token_level, mask_zeros)

    def _colorize(self, samples: List[AttributedSample]) -> List[str]:
        htmls = []
        for idx, sample in enumerate(samples):
            importances = (
                sample.scores if self.token_level else sample.char_scores
            )
            tokens = sample.tokens if self.token_level else list(sample.text)

            # SimpleHTMLVisualizer needs to scale the scores between 0 and 1.
            # Normalize importances to values between 0 and 1 (picking the values > 0)
            max_importance = max(importances)

            # Better subjective perception when normalized by importances > 0
            # even when we loss some words. It seems to increase P and reduce R
            if (importances <= 0).all():
                normalized_importances = np.zeros_like(importances)
            else:
                min_importance = min(importances[importances > 0])

                # Check if there is any variation in importances
                # e.g. for hard_higher_mean values are only {0, 1} causing
                # denominator to be equal to 0
                if ((importances > 0) == (importances == min_importance)).all():
                    normalized_importances = importances
                else:
                    normalized_importances = [
                        (imp - min_importance)
                        / (max_importance - min_importance)
                        for imp in importances
                    ]
            # Generate HTML with inline styles
            html = f'<h3 align="center"> Sample: {idx}, Pred: {sample.pred_label} (prob={round(sample.pred_prob, 2)})</h3><hr>'
            html += '<p style="font-size: 16px; line-height: 1.5;">'
            for token, importance in zip(tokens, normalized_importances):
                # Calculate background color based on importance
                # Green with varying opacity
                background_color = f"rgba(0, 255, 0, {importance})"
                # Add token with inline style
                if self.token_level:
                    html += f'<span style="background-color: {background_color}; padding: 2px; border-radius: 3px;">{token}</span> '
                else:
                    html += f'<span style="background-color: {background_color};">{token}</span>'
            html += "</p><hr>"
            htmls.append(html)
        return htmls
