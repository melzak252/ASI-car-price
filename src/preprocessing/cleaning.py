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
