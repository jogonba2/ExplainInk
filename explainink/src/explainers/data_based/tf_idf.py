from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ...common import get_spacy_model
from .base import DataExplainer


class TFIDFExplainer(DataExplainer):
    """
    Class for TFIDF explainer.
    Similar to CTFIDF, but computing the IDF term just from the texts of a class.

    Attributes:
        vectorizer_params (Dict): params to be passed to the sklearn vectorizer.
        use_idf (bool): whether to use the IDF term or not.
    """

    def __init__(
        self,
        vectorizer_params: Dict = {"ngram_range": (1, 3)},
        use_idf: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vectorizer_params = vectorizer_params
        self.use_idf = use_idf

        if "ngram_range" in self.vectorizer_params:
            self.vectorizer_params["ngram_range"] = tuple(
                self.vectorizer_params["ngram_range"]
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

        # Instantiate the vectorizer, avoid preprocessing and tokenize
        # by whitespaces to not miss the link between spans and the original text.
        stopwords = list(get_spacy_model(self.language).Defaults.stop_words)
        vectorizer = TfidfVectorizer(
            **self.vectorizer_params,
            use_idf=False,
            stop_words=stopwords,
            tokenizer=lambda x: x.split(),
            lowercase=True,
        )

        # Compute TF scores.
        tf_scores = vectorizer.fit_transform([class_document]).toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        scores = [
            (ngram, score) for ngram, score in zip(feature_names, tf_scores)
        ]

        # Multiply by the IDF term, using 0 for the new ngrams created by joining texts
        if self.use_idf:
            vectorizer = TfidfVectorizer(
                **self.vectorizer_params,
                stop_words=stopwords,
                tokenizer=lambda x: x.split(),
            )
            vectorizer.fit(texts)
            idf = vectorizer.idf_
            ngram_ids = {
                ngram: idx
                for idx, ngram in enumerate(vectorizer.get_feature_names_out())
            }
            scores = [
                (
                    ngram,
                    score
                    * (idf[ngram_ids[ngram]] if ngram in ngram_ids else 0),
                )
                for ngram, score in scores
            ]

        return scores
