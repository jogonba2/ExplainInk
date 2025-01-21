from collections import defaultdict
from string import punctuation
from typing import Dict, List

import numpy as np

from ...common.utils import get_spacy_model
from ...types import AttributedSample


def remove_special_tokens(
    samples: List[AttributedSample], special_tokens: List[str]
) -> List[AttributedSample]:
    """
    Removes the special tokens from the attributed samples:
    token scores, tokens, word_ids, and token_to_chars.

    Args:
        samples (List[AttributedSample]): list of attributed samples.
        special_tokens (List[str]): list of special tokens of the model.

    Returns:
        List[AttributedSample]: list of attributed samples without special token related information.
    """
    new_samples = []
    for sample in samples:
        after_tokens = []
        after_scores = []
        after_word_ids = []
        after_token_to_chars = []
        for token, score, word_id, token_to_char in zip(
            sample.tokens, sample.scores, sample.word_ids, sample.token_to_chars
        ):
            if token not in special_tokens:
                after_tokens.append(token)
                after_scores.append(score)
                after_word_ids.append(word_id)
                after_token_to_chars.append(token_to_char)

        new_samples.append(
            AttributedSample(
                text=sample.text,
                tokens=after_tokens,
                scores=np.array(after_scores),
                pred_label=sample.pred_label,
                pred_prob=sample.pred_prob,
                word_ids=after_word_ids,
                token_to_chars=after_token_to_chars,
                char_scores=sample.char_scores,
            )
        )
    return new_samples


def postprocess_scores(
    samples: List[AttributedSample], language: str, mask_scores: Dict
) -> List[AttributedSample]:
    """
    Attributes minimum scores to: stopwords, punctuation tokens,
    or short tokens, if specified in `mask_scores`.

    Args:
        samples (List[AttributedSample]): list of attributed samples.
        language (str): language of the texts.
        mask_scores (Dict): what masking postprocesses make.

    Returns:
        List[AttributedSample]: list of postprocessed attributed samples.
    """
    stopwords = set(get_spacy_model(language).Defaults.stop_words)
    punctuation_symbols = set(punctuation)
    new_samples = []
    for sample in samples:
        min_score = min(sample.scores)
        for i in range(len(sample.tokens)):
            token = sample.tokens[i]
            if (
                (mask_scores["stopwords"] and token in stopwords)
                or (mask_scores["punctuation"] and token in punctuation_symbols)
                or (len(token) < mask_scores["shorter"])
            ):
                sample.scores[i] = min_score
        new_samples.append(sample)
    return new_samples


def propagate_scores_to_same_word(
    samples: List[AttributedSample],
) -> List[AttributedSample]:
    """
    Propagates scores from tokens within the same word, using the max score,
    to avoid having tokens within the same word with very different scores.

    Example:
        - tokens = cl ##eti ##s
        - scores = 0  0.2   0.3
        - result = 0.3 0.3 0.3

    Args:
        samples (List[AttributedSample]): list of attributed samples.

    Returns:
        List[AttributedSample]: list of attributed samples with propagated scores.
    """
    new_samples = []
    for sample in samples:
        word_mapping = defaultdict(list)
        for i, word_id in enumerate(sample.word_ids):
            word_mapping[word_id].append(i)
        for token_group in word_mapping.values():
            max_score = np.max([sample.scores[i] for i in token_group])
            for i in token_group:
                sample.scores[i] = max_score
        new_samples.append(sample)
    return new_samples


def build_char_scores(
    samples: List[AttributedSample],
) -> List[AttributedSample]:
    """
    Assigns scores to chars according to the token scores.
    This way we get score of each char in the text.

    Example:
        tokens = cletis
        token_scores = 0.3
        chars = c l e t i s
        scores = 0.3 0.3 0.3 0.3 0.3 0.3

    Args:
        samples (List[AttributedSample]): list of attributed samples.

    Returns:
        List[AttributedSample]: list of attributed samples with char scores.
    """
    new_samples = []
    for sample in samples:
        token_pos = 0
        char_scores = np.zeros(len(sample.text))
        for i in range(len(sample.text)):
            # If the text has been truncated, then we need to stop
            # at the maximum token.
            if token_pos >= len(sample.token_to_chars):
                break

            # Avoid scores to whitespaces.
            if sample.text[i] != " ":
                char_scores[i] = sample.scores[token_pos]

            # Postprocessings:
            # If English negation ("'t") or possessive ("'s"), assign the scores of the previous chars
            if (
                i > 0
                and i < len(sample.text) - 1
                and sample.text[i] == "'"
                and (sample.text[i + 1] == "t" or sample.text[i + 1] == "s")
            ):
                char_scores[i] = char_scores[i - 1]
            if (
                i > 0
                and (sample.text[i] == "t" or sample.text[i] == "s")
                and sample.text[i - 1] == "'"
            ):
                char_scores[i] = char_scores[i - 1]

            # Increase the token position pointer for this character
            # if it i is out of the current token span
            if i + 1 >= sample.token_to_chars[token_pos].end:
                token_pos += 1

        sample.char_scores = char_scores
        new_samples.append(sample)
    return new_samples
