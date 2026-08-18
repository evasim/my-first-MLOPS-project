import numpy as np
import pandas as pd
import pytest
import ray

from src.data import clean_text, stratify_split, tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", "hello world"),
        ("Stocks   rally    after   the   Fed   meeting", "stocks rally fed meeting"),
        ("Check https://example.com for details", "check details"),
        ("", ""),
    ],
)
def test_clean_text(text, expected):
    assert clean_text(text) == expected


def test_clean_text_is_lowercase():
    assert clean_text("UPPER CASE TEXT") == clean_text("UPPER CASE TEXT").lower()


def test_stratify_split_preserves_class_proportions():
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        }
    )
    ds = ray.data.from_pandas(df)
    train_ds, test_ds = stratify_split(ds, stratify="label", test_size=0.2)

    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}
    assert train_df["label"].value_counts()[0] == train_df["label"].value_counts()[1]
    assert test_df["label"].value_counts()[0] == test_df["label"].value_counts()[1]


def test_tokenize_shapes_match_batch_size():
    batch = pd.DataFrame({"text": ["hello world", "a longer sentence here"], "label": [0, 1]})
    result = tokenize(batch)

    assert result["ids"].shape[0] == len(batch)
    assert result["masks"].shape == result["ids"].shape
    assert np.array_equal(result["targets"], np.array([0, 1]))
