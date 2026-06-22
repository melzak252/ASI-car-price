import numpy as np
import pandas as pd

from preprocessing.cleaning import (
    clean_features,
    filter_doors,
    remove_eur_offers,
    remove_log_price_iqr_outliers,
)


def test_clean_features_parses_list_string():
    assert clean_features("['ABS', 'Drivers airbag']") == ["ABS", "Drivers airbag"]


def test_clean_features_handles_nan_and_empty():
    assert clean_features(np.nan) == []
    assert clean_features("[]") == []


def test_clean_features_fallback_on_malformed():
    assert clean_features("ABS, Airbag") == ["ABS", "Airbag"]


def test_remove_eur_offers():
    df = pd.DataFrame({"Currency": ["PLN", "EUR", "PLN"]})
    result = remove_eur_offers(df)
    assert list(result["Currency"]) == ["PLN", "PLN"]


def test_filter_doors_keeps_nan_and_drops_outliers():
    df = pd.DataFrame({"Doors_number": [3.0, 5.0, 99.0, np.nan]})
    result = filter_doors(df, max_doors=7)
    assert len(result) == 3
    assert 99.0 not in result["Doors_number"].dropna().values


def test_remove_log_price_iqr_outliers_drops_extremes():
    df = pd.DataFrame({"Log_Price": list(range(10)) + [1000]})
    result = remove_log_price_iqr_outliers(df, "Log_Price")
    assert 1000 not in result["Log_Price"].values
