from typing import Dict, List, Tuple

import pandas as pd

try:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
except ImportError:
    print("BERTopic is not installed, install it if required.")

from sklearn.feature_extraction.text import CountVectorizer

from ...common import get_spacy_model
from .base import DataExplainer


class CTFIDFExplainer(DataExplainer):
    """
    Class for CTFIDF based on:
    https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html

    Attributes:
        vectorizer_params (Dict): params to be passed to the BERTopic vectorizer.
    """

    def __init__(
        self, vectorizer_params: Dict = {"ngram_range": (1, 3)}, **kwargs
    ):
        super().__init__(**kwargs)

        self.vectorizer_params = vectorizer_params

        if "ngram_range" in self.vectorizer_params:
            self.vectorizer_params["ngram_range"] = tuple(
                self.vectorizer_params["ngram_range"]
            )

    def _get_span_scores(
        self, texts: List[str], labels: List[int], target_label: int
    ) -> List[Tuple[str, float]]:
        # Prepare data for BERTopic's `_c_tf_idf` method
        df = pd.DataFrame({"text": texts, "label": labels})
        df = df.sort_values("label").groupby(["label"], as_index=False)
        df = df.agg({"text": " ".join})
        df = df.rename(columns={"label": "Topic", "text": "Document"})

        # Pass the vectorizer to BERTopic to avoid preprocessing and tokenize
        # by whitespaces to not miss the link between spans and the original text.
        stopwords = list(get_spacy_model(self.language).Defaults.stop_words)
        vectorizer = CountVectorizer(
            **self.vectorizer_params,
            tokenizer=lambda x: x.split(),
            stop_words=stopwords,
            lowercase=True,
            preprocessor=None
        )

        ctfidf_transformer = ClassTfidfTransformer(
            reduce_frequent_words=True, bm25_weighting=True
        )
        topic_model = BERTopic(
            ctfidf_model=ctfidf_transformer, vectorizer_model=vectorizer
        )
        topic_model._preprocess_text = lambda x: x

        # Compute the CTFIDF scores
        c_tf_idf_scores, words = topic_model._c_tf_idf(df)
        c_tf_idf_scores = c_tf_idf_scores.toarray()[target_label]

        # Prepare the output
        scores = [
            (ngram, score) for ngram, score in zip(words, c_tf_idf_scores)
        ]

        return scores
