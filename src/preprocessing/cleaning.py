import ast

import pandas as pd


def remove_eur_offers(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Currency"] != "EUR"].copy()


def remove_log_price_iqr_outliers(df: pd.DataFrame, column: str = "Log_Price") -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()


def filter_doors(df: pd.DataFrame, max_doors: int = 7) -> pd.DataFrame:
    """Usuwa oferty z nietypową liczbą drzwi (zachowuje też braki danych)."""
    return df[df["Doors_number"].isna() | (df["Doors_number"] <= max_doors)].copy()


def clean_features(text) -> list[str]:
    """Parsuje kolumnę `Features` (string z listą) do listy nazw wyposażenia."""
    if pd.isna(text):
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    cleaned = str(text).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [item.strip() for item in cleaned.split(",") if item.strip()]
