"""Węzły pipeline'u preprocessingu i inżynierii cech."""

import pandas as pd
from sklearn.model_selection import train_test_split

from features.build_features import add_brand_model, add_log_price, build_feature_matrices
from preprocessing.cleaning import (
    clean_features,
    filter_doors,
    remove_eur_offers,
    remove_log_price_iqr_outliers,
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Czyszczenie danych i podstawowa inżynieria cech (przed splitem)."""
    df = add_log_price(df)
    df = remove_log_price_iqr_outliers(df, "Log_Price")
    df = remove_eur_offers(df)
    df = filter_doors(df, max_doors=7)
    df = add_brand_model(df)
    df["feat_list"] = df["Features"].apply(clean_features)
    return df


def split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Podział na zbiór treningowy i testowy."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    drop_columns: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Budowa macierzy cech (target encoding liczony tylko na train)."""
    return build_feature_matrices(
        train_df,
        test_df,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        drop_columns=drop_columns,
        target_column=target,
    )
