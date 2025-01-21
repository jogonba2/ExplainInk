import inspect
from typing import Callable, Dict, Final, List, Literal, Optional

import spacy
import torch
from tqdm import tqdm

SPACY_MODEL_MAPPING: Final[Dict[str, str]] = {
    "ca": "ca_core_news_sm",
    "zh": "zh_core_web_sm",
    "hr": "hr_core_news_sm",
    "da": "da_core_news_sm",
    "nl": "nl_core_news_sm",
    "en": "en_core_web_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
    "multilingual": "xx_sent_ud_sm",
}


def get_spacy_model(language: str) -> spacy.lang:
    """
    Gets or download a Spacy model.

    Args:
        language (str): language.

    Returns:
        spacy.lang: a Spacy model.
    """
    spacy_model = SPACY_MODEL_MAPPING.get(language, None)
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        spacy.cli.download(spacy_model)
        nlp = spacy.load(spacy_model)
    return nlp


def spacy_pipeline(
    texts: List[str],
    language: str,
    disable_pipes: List[str] = [],
    n_process: int = 4,
) -> List[spacy.tokens.Doc]:
    """
    Processes texts with spacy pipeline for entity extraction.

    Args:
        texts (List[str]): list of texts.
        language (str): language of the text.
        disable_pipes (List[str]): Spacy pipes to be disabled.
        n_process (int): number of processes.

    Returns:
        List[spacy.tokens.Doc]: list of Spacy docs.
    """
    nlp = get_spacy_model(language)

    processed_texts = list(
        tqdm(
            nlp.pipe(
                texts,
                n_process=n_process,
                disable=disable_pipes,
            ),
            total=len(texts),
            desc="Spacy processing",
        )
    )

    return processed_texts


def get_signature_args(
    fn: Callable, args: Dict, accepted_params: Optional[List[str]] = None
) -> Dict:
    """
    Extracts the arguments from `args` that match the
    parameters of the `fn` function.

    Args:
        fn (Callable): a function.
        args (Dict): a dictionary of arguments.
        accepted_params (Optional[List[str]]): if known, the list
            of params accepted by fn. If `accepted_params` is
            provided, the `fn` argument will not be used to infer
            the accepted parameters.
    Returns:
        Dict: arguments in `args` that match the `fn` parameters.
    """
    if accepted_params is None:
        accepted_params = list(inspect.signature(fn).parameters.keys())
    return {arg: value for arg, value in args.items() if arg in accepted_params}


def batch_to_device(batch: Dict, device: str) -> Dict:
    """
    Moves a batch of features to the device.

    Args:
        batch (Dict): batch of features.
        device (str): device where to move the batch, e.g., cuda.

    Returns:
        Dict: batch of features in `device`.
    """
    return {k: v.to(device) for k, v in batch.items()}


def remove_prefix_symbols(
    tokens: list[str], prefix_symbols: list[str] = ["Ġ", "_"]
) -> list[str]:
    """
    Removes prefixes and artifacts that can cause inconsistency in attribution
    evaluation with rationales. That is, the tokenization of an attributed sample
    tokens could not match the tokens of the rationales.

    Specifially useful for some BPE tokenizers like RoBERTa or GPT-2. See here
    https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475/2
    for more information.

    For instance, with the RoBERTa tokenizer:

    - Text: "This is a negative text" -> ['ĠThis', 'Ġis', 'Ġa', 'Ġnegative', 'Ġtext']
    - Tokens with highest attribution scores -> ['Ġthis', 'Ġnegative']
    - Rationale: ["This", "negative"] -> ["This", "negative"]

    So 'Ġthis' != "this" and "negative" != 'Ġnegative' -> metrics = 0, but the match is perfect.

    Args:
        tokens (list[str]) -> list of tokens

    Returns:
        list[str] -> tokens without inconsistent symbols
    """
    replace_symbols = "".join(prefix_symbols)
    return [token.lstrip(replace_symbols).lower().strip() for token in tokens]


def get_torch_aggregation(name: Literal["max", "mean", "min"]) -> Callable:
    """
    Get an aggregation statistic function from torch in an way that all
    of them work within the same interface: pass a tensor and return values
    without indices.

    Args:
        name (Literal): one of "max", "mean", or "min"
    Returns:
        Callable: a torch function that just returns values
    """
    aggregation_map = {
        "mean": torch.mean,
        "max": lambda input, dim: torch.max(input, dim=dim).values,
        "min": lambda input, dim: torch.min(input, dim=dim).values,
    }

    try:
        return aggregation_map[name]
    except KeyError:
        raise ValueError(f"The aggregation '{name}' is not supported.")
