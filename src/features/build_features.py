import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def add_log_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Log_Price"] = np.log(df["Price"])
    return df


def add_brand_model(df: pd.DataFrame) -> pd.DataFrame:
    """Łączy markę i model w jedną cechę `Brand_Model`."""
    df = df.copy()
    df["Brand_Model"] = df["Vehicle_brand"] + " " + df["Vehicle_model"].fillna("")
    return df


def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    target_column: str = "Log_Price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target encoding liczony wyłącznie na zbiorze treningowym (bez leakage)."""
    global_mean = train[target_column].mean()
    mapping = train.groupby(column)[target_column].mean()
    train[f"{column}_Encoded"] = train[column].map(mapping).fillna(global_mean)
    test[f"{column}_Encoded"] = test[column].map(mapping).fillna(global_mean)
    return train, test


def build_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    drop_columns: list[str],
    target_column: str = "Log_Price",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Buduje macierze cech zgodnie z notebookiem baseline.

    Kroki: target encoding marki/modelu, uzupełnienie medianą cech numerycznych,
    one-hot encoding cech kategorycznych, binaryzacja listy wyposażenia,
    standaryzacja cech numerycznych. Statystyki (mediany, mapy, średnie, std)
    liczone są na zbiorze treningowym i stosowane do testowego.
    """
    train = train.copy()
    test = test.copy()

    train, test = add_target_encoding(train, test, "Brand_Model", target_column)
    train, test = add_target_encoding(train, test, "Vehicle_brand", target_column)

    medians = train[numeric_columns].median(numeric_only=True)
    train[numeric_columns] = train[numeric_columns].fillna(medians)
    test[numeric_columns] = test[numeric_columns].fillna(medians)

    train_dummies = pd.get_dummies(train[categorical_columns], columns=categorical_columns, dummy_na=True)
    test_dummies = pd.get_dummies(test[categorical_columns], columns=categorical_columns, dummy_na=True)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    mlb = MultiLabelBinarizer()
    train_features = pd.DataFrame(
        mlb.fit_transform(train["feat_list"]),
        columns=[f"feature_{name}" for name in mlb.classes_],
        index=train.index,
    )
    test_features = pd.DataFrame(
        mlb.transform(test["feat_list"]),
        columns=train_features.columns,
        index=test.index,
    )

    y_train = train[target_column].copy()
    y_test = test[target_column].copy()

    train_base = train.drop(columns=drop_columns + categorical_columns + [target_column], errors="ignore")
    test_base = test.drop(columns=drop_columns + categorical_columns + [target_column], errors="ignore")

    X_train = pd.concat([train_base, train_dummies, train_features], axis=1)
    X_test = pd.concat([test_base, test_dummies, test_features], axis=1)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    for column in numeric_columns:
        mean = X_train[column].mean()
        std = X_train[column].std()
        if std and std > 0:
            X_train[column] = (X_train[column] - mean) / std
            X_test[column] = (X_test[column] - mean) / std

    X_train = X_train.fillna(0).astype(float)
    X_test = X_test.fillna(0).astype(float)

    return X_train, X_test, y_train, y_test
