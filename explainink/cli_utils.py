import json
from pathlib import Path
from typing import Dict, List, Union

from datasets import Dataset, load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score


def read_json(path: Path) -> Dict:
    """
    Reads a JSON file from disk.

    Args:
        path (Path): path to the JSON file.

    Returns:
        Dict: content of the JSON file as dictionary.
    """
    with path.open("r") as fr:
        return json.loads(fr.read())


def save_json(content: Dict, path: Path) -> None:
    """
    Saves a dictionary as a JSON in disk.

    Args:
        content (Dict): dictionary to be stored.
        path (Path): path where to store the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fw:
        json.dump(content, fw, indent=4)


def save_text(text: Union[str, List[str]], path: Path) -> None:
    """
    Saves a text or list of texts in disk.

    Args:
        text (Union[str, List[str]]): text or list of texts to be stored.
        path (Path): path where to store the text file.
    """
    if isinstance(text, str):
        text = [text]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fw:
        for t in text:
            fw.write(t)


def load_hf_dataset(name: str, args: Dict) -> Dataset:
    """
    Loads a HuggingFace dataset from the hub or from disk.

    Args:
        name (str): name of the dataset
        args (Dict): additional arguments when loading from the hub.

    Returns:
        Dataset: a HuggingFace dataset.
    """

    try:
        dataset = load_from_disk(name)
    except FileNotFoundError:
        dataset = load_dataset(name, **args)
    return dataset


def evaluate_classification(
    refs: List[int], preds: List[int]
) -> Dict[str, float]:
    """
    Computes accuracy, macro-f1 and micro-f1 using `refs` and `preds`.

    Args:
        refs (List[int]): reference labels.
        preds (List[int]): predictions of the model.

    Returns:
        Dict[str, float]: dictionary with the metric values.
    """
    return {
        "accuracy": accuracy_score(refs, preds),
        "macro-f1": f1_score(refs, preds, average="macro"),
        "micro-f1": f1_score(refs, preds, average="micro"),
    }
