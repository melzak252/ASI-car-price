import numpy as np
import pandas as pd

from features.selection import apply_selection, select_features


def _frame(n=200, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # target zależny głównie od f0 i f1 - powinny trafić do najważniejszych cech
    y = 3 * X["f0"] - 2 * X["f1"] + rng.normal(scale=0.1, size=n)
    return X, pd.Series(y, name="Log_Price")


def test_select_features_respects_top_k():
    X, y = _frame()
    selected = select_features(X, y, top_k=4)
    assert len(selected) == 4
    assert all(col in X.columns for col in selected)


def test_select_features_finds_informative_columns():
    X, y = _frame()
    selected = select_features(X, y, top_k=3)
    assert "f0" in selected and "f1" in selected


def test_select_features_caps_top_k_to_column_count():
    X, y = _frame(n_features=5)
    selected = select_features(X, y, top_k=99)
    assert len(selected) == 5


def test_apply_selection_subsets_columns():
    X, _ = _frame(n_features=6)
    result = apply_selection(X, ["f0", "f2"])
    assert list(result.columns) == ["f0", "f2"]
    assert len(result) == len(X)
