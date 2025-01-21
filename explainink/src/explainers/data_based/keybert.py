from typing import Dict, List, Tuple

import pandas as pd

try:
    from keybert import KeyBERT
except ImportError:
    print("KeyBERT is not installed, install it if required.")

from sklearn.feature_extraction.text import CountVectorizer

from ...common import get_spacy_model
from .base import DataExplainer


class KeyBERTExplainer(DataExplainer):
    """
    Class for KeyBERT explainer.
    Extracts keyphrases using the KeyBERT pipeline.

    Attributes:
        keybert_params (Dict): params to instantiate the KeyBERT model.
        inference_params (Dict): params to be passed to KeyBERT's `extract_keywords` method.
    """

    def __init__(
        self,
        keybert_params: Dict = {},
        inference_params: Dict = {
            "keyphrase_ngram_range": (1, 3),
            "top_n": 1000,
        },
        **kwargs
    ):
        super().__init__(**kwargs)

        self.keybert = KeyBERT(**keybert_params)
        self.inference_params = inference_params

        if "keyphrase_ngram_range" in self.inference_params:
            self.inference_params["keyphrase_ngram_range"] = tuple(
                self.inference_params["keyphrase_ngram_range"]
            )

    def _get_span_scores(
        self, texts: List[str], labels: List[int], target_label: int
    ) -> List[Tuple[str, float]]:
        # Prepare the data (select the texts predicted as `target_label`).
        df = pd.DataFrame({"text": texts, "label": labels})
        texts_from_target_label = df[df["label"] == target_label][
            "text"
        ].tolist()

        # Join the documents of the class
        class_document = " ".join(texts_from_target_label)

        # Pass the vectorizer to KeyBERT to avoid preprocessing and tokenize
        # by white spaces to not miss the link between spans and the original text.
        stopwords = list(get_spacy_model(self.language).Defaults.stop_words)
        vectorizer = CountVectorizer(
            ngram_range=self.inference_params["keyphrase_ngram_range"],
            tokenizer=lambda x: x.split(),
            stop_words=stopwords,
            lowercase=True,
        )

        # Extract keyphrases
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            class_document, **self.inference_params, vectorizer=vectorizer
        )

        return keywords
