import numpy as np
import pandas as pd


def add_log_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Log_Price"] = np.log(df["Price"])
    return df
