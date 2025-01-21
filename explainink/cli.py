from pathlib import Path
from time import time

import typer
from typing_extensions import Annotated

from .cli_utils import (
    evaluate_classification,
    load_hf_dataset,
    read_json,
    save_json,
    save_text,
)
from .src import explainers, models, visualizers
from .src.evaluators.correlations import (
    evaluate_attributions as evaluate_attributions_correlations,
)
from .src.evaluators.rationales import (
    evaluate_attributions as evaluate_attributions_rationales,
)
from .src.types import AttributedSample

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to train a model on a training set and evaluate it
    at the end of the training on a test set.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the evaluation results in JSON format.
    """
    config = read_json(config_path)
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )

    model = getattr(models, config["model"]["class"])(
        **config["model"]["params"]
    )
    model.fit(dataset["train"])
    preds = model.predict(dataset["test"])
    model.save()
    results = evaluate_classification(dataset["test"]["label"], preds)
    save_json(results, output_path)


@app.command()
def eval_classification(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to evaluate a model, loaded from disk,
    on a test set of a classification task.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the evaluation results in JSON format.
    """
    config = read_json(config_path)
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )
    model = getattr(models, config["model"]["class"])(
        **config["model"]["params"]
    )

    model.load(config["model"]["checkpoint"])

    preds = model.predict(dataset["test"])
    results = evaluate_classification(dataset["test"]["label"], preds)
    save_json(results, Path(output_path))


@app.command()
def explain_model_based(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to compute model-based attributions of the test samples.
    Stores all the information of the attributed samples in disk.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the attributions in JSON format.
    """
    config = read_json(config_path)
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )
    model = getattr(models, config["model"]["class"])(
        **config["model"]["params"]
    )

    model.load(config["model"]["checkpoint"])

    explainer = getattr(explainers, config["explainer"]["class"])(
        model, **config["explainer"]["instantiation_params"]
    )
    ts = time()

    attributed_samples = explainer.explain(
        dataset["test"][config["dataset"]["text_column"]],
        batch_size=config["explainer"]["batch_size"],
        targets=(
            dataset["test"][config["dataset"]["label_column"]]
            if config["explainer"].get("pass_reference_targets")
            else None
        ),
    )

    te = time()
    total_time = te - ts
    output = {
        "total_time": total_time,
        "samples": [sample.serialize() for sample in attributed_samples],
    }
    save_json(output, Path(output_path))


@app.command()
def explain_data_based(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to compute data-based attributions of the test samples.
    Stores all the information of the attributed samples in disk.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the attributions in JSON format.
    """
    config = read_json(config_path)
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )

    explainer = getattr(explainers, config["explainer"]["class"])(
        **config["explainer"]["instantiation_params"]
    )
    ts = time()

    attributed_samples = explainer.explain(
        texts=dataset["test"][config["dataset"]["text_column"]],
        labels=dataset["test"][config["dataset"]["label_column"]],
    )

    te = time()
    total_time = te - ts
    output = {
        "total_time": total_time,
        "samples": [sample.serialize() for sample in attributed_samples],
    }
    save_json(output, Path(output_path))


@app.command()
def visualize_from_path(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    visualizer_class: Annotated[
        str,
        typer.Option(
            help="Visualizer class.",
        ),
    ] = "SimpleVisualizer",
):
    """
    Endpoint to visualize attributed samples already stored in disk.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the HTML visualization.
        visualizer_class (str): class from `visualizers` to compute the HTML.
    """
    config = read_json(config_path)
    visualizer = getattr(visualizers, visualizer_class)()
    json_attributions = read_json(Path(config["attributions_path"]))
    attributed_samples = [
        AttributedSample.load(**sample)
        for sample in json_attributions["samples"]
    ]
    htmls = visualizer.colorize(attributed_samples)
    output_path = Path(output_path).parents[0] / f"{visualizer_class}.html"
    save_text(htmls, output_path)


@app.command()
def visualize_model_based(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to perform visualization of model-based approaches in an end-to-end manner.
    First computes model-based explanations and then generates the HTMLs for visualization.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the HTML visualization.
    """
    # Explain
    config = read_json(config_path)

    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )

    model = getattr(models, config["model"]["class"])(
        **config["model"]["params"]
    )

    model.load(config["model"]["checkpoint"])

    explainer = getattr(explainers, config["explainer"]["class"])(
        model, **config["explainer"]["instantiation_params"]
    )

    attributed_samples = explainer.explain(
        dataset["test"][config["dataset"]["text_column"]],
        batch_size=config["explainer"]["batch_size"],
        targets=(
            dataset["test"][config["dataset"]["label_column"]]
            if config["explainer"].get("pass_reference_targets")
            else None
        ),
    )

    # Visualize
    visualizer_class = config["visualizer"]["visualizer_class"]
    visualizer_params = config["visualizer"]["instantiation_params"]
    visualizer = getattr(visualizers, visualizer_class)(**visualizer_params)
    htmls = visualizer.colorize(attributed_samples)
    save_text(htmls, output_path)


@app.command()
def visualize_data_based(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to perform visualization of data-based approaches in an end-to-end manner.
    First computes data-based explanations and then generates the HTMLs for visualization.

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the HTML visualization.
    """
    # Explain
    config = read_json(config_path)
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )

    explainer = getattr(explainers, config["explainer"]["class"])(
        **config["explainer"]["instantiation_params"]
    )

    attributed_samples = explainer.explain(
        texts=dataset["test"][config["dataset"]["text_column"]],
        labels=dataset["test"][config["dataset"]["label_column"]],
    )

    # Visualize
    visualizer_class = config["visualizer"]["visualizer_class"]
    visualizer_params = config["visualizer"]["instantiation_params"]
    visualizer = getattr(visualizers, visualizer_class)(**visualizer_params)
    htmls = visualizer.colorize(attributed_samples)
    save_text(htmls, output_path)


@app.command()
def eval_attributions_with_rationales(
    config_path: Annotated[
        Path,
        typer.Option(
            help="Path to the config file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to evaluate the token attributions of a model and explainer, already stored
    in disk, using datasets with rationales (list of important words selected by humans).

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the evaluation results.
    """
    config = read_json(config_path)
    json_attributions = read_json(Path(config["attributions_file"]))
    attributed_samples = [
        AttributedSample.load(**sample)
        for sample in json_attributions["samples"]
    ]
    dataset = load_hf_dataset(
        config["dataset"]["name"], config["dataset"]["args"]
    )
    rationales = dataset["test"]["rationales"]
    results = evaluate_attributions_rationales(
        attributed_samples,
        rationales,
        config["evaluation_params"]["tokenizer"],
        config["evaluation_params"]["discretizer_name"],
        config["evaluation_params"]["discretizer_params"],
    )
    save_json(results, output_path)


@app.command()
def eval_attributions_with_correlations(
    attributions_file_a: Annotated[
        Path,
        typer.Option(
            help="Path to attributions.",
            exists=True,
            file_okay=True,
            resolve_path=True,
        ),
    ],
    attributions_file_b: Annotated[
        Path,
        typer.Option(
            help="Path to another attributions.",
            exists=True,
            file_okay=True,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            help="Output path",
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """
    Endpoint to evaluate the correlation between the scores computed
    by one explainer and the scores computed by another explainer
    (potentially human-crafted scores).

    Args:
        config_path (Path): path to the config file.
        output_path (Path): path where to store the evaluation results.
    """
    json_attributions_a = read_json(Path(attributions_file_a))
    json_attributions_b = read_json(Path(attributions_file_b))

    attributed_samples_a = [
        AttributedSample.load(**sample)
        for sample in json_attributions_a["samples"]
    ]
    attributed_samples_b = [
        AttributedSample.load(**sample)
        for sample in json_attributions_b["samples"]
    ]
    results = evaluate_attributions_correlations(
        attributed_samples_a, attributed_samples_b
    )
    results = {k: v.serialize() for k, v in results.items()}
    save_json(results, output_path)


if __name__ == "__main__":
    app()
