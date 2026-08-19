import os

import pytest

from src.predict import LABEL_DECODER, predict

pytestmark = pytest.mark.skipif(
    "HF_REPO_ID" not in os.environ, reason="requires HF_REPO_ID (downloads the real checkpoint)"
)


def test_predict_returns_known_label():
    label = predict("Wall Street stocks rallied after the Federal Reserve meeting.")
    assert label in LABEL_DECODER.values()
