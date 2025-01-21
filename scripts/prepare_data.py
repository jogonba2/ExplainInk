from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from scipy import stats

RANDOM_SEED = 13


def prepare_movie_rationales():
    """
    https://huggingface.co/datasets/movie_rationales
    """
    dataset = load_dataset("movie_rationales")
    dataset = DatasetDict(
        train=dataset["train"],
        test=concatenate_datasets([dataset["validation"], dataset["test"]]),
    )
    dataset = dataset.rename_column("review", "text")
    dataset = dataset.rename_column("evidences", "rationales")
    dataset = dataset.map(
        lambda x: {"rationales": list(set(" ".join(x["rationales"]).split()))}
    )
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    print(dataset["test"]["text"][0])
    print(dataset["test"]["rationales"][0])
    print(dataset["test"]["label"][0])
    dataset.save_to_disk("az://snlp-data/explainink/movie_rationales")


def prepare_fair_rationales(name: str):
    """
    https://huggingface.co/datasets/coastalcph/fair-rationales
    """
    dataset = load_dataset("coastalcph/fair-rationales", name)["train"]
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("rationale", "rationales")
    dataset = dataset.remove_columns(["label"])
    dataset = dataset.rename_column("original_label", "label")

    # Group annotations
    df = dataset.to_pandas()
    result_df = (
        df.groupby("text_id")["rationales"]
        .agg(lambda x: ",".join(x))
        .reset_index()
    )
    result_df["text"] = result_df.apply(
        lambda row: df.loc[df["text_id"] == row["text_id"], "text"].iloc[0],
        axis=1,
    )
    result_df["label"] = result_df.apply(
        lambda row: df.loc[df["text_id"] == row["text_id"], "label"].iloc[0],
        axis=1,
    )
    result_df["rationales"] = result_df["rationales"].apply(
        lambda x: list(set(x.split(",")))
    )
    result_df["rationales"] = result_df["rationales"].apply(
        lambda x: [t for t in x if t]
    )
    dataset = Dataset.from_pandas(result_df)
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset)
    print(dataset["test"]["text"][0])
    print(dataset["test"]["rationales"][0])
    print(dataset["test"]["label"][0])
    exit()
    dataset.save_to_disk(f"az://snlp-data/explainink/fair_rationales_{name}")


def prepare_hateexplain():
    """
    https://huggingface.co/datasets/hatexplain
    """
    dataset = load_dataset("hatexplain")
    datasets = {}
    for split in ["train", "validation", "test"]:
        samples = []
        tokens = dataset[split]["post_tokens"]
        rationales = dataset[split]["rationales"]
        annotators = dataset[split]["annotators"]
        for token, rationale, annotator in zip(tokens, rationales, annotators):
            text = " ".join(token)
            label = stats.mode(annotator["label"])[0][0]
            bool_rationale = [
                int(any(elements)) for elements in zip(*rationale)
            ]
            tokens_rationale = [
                token for token, isin in zip(token, bool_rationale) if isin
            ]
            samples.append(
                {"text": text, "label": label, "rationales": tokens_rationale}
            )
        datasets[split] = Dataset.from_list(samples)
    dataset = DatasetDict(
        train=datasets["train"],
        test=concatenate_datasets([datasets["validation"], datasets["test"]]),
    )
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    print(dataset["test"]["text"][0])
    print(dataset["test"]["rationales"][0])
    print(dataset["test"]["label"][0])
    dataset.save_to_disk("az://snlp-data/explainink/hatexplain")


if __name__ == "__main__":
    # prepare_movie_rationales()
    # prepare_fair_rationales("cose")
    # prepare_fair_rationales("sst2")
    # prepare_fair_rationales("dynasent")
    # prepare_hateexplain()
    ...
