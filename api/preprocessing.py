from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class InferencePreprocessor:
    def __init__(self) -> None:
        self.target_enc_columns: list[str] = []
        self.target_maps: dict[str, pd.Series] = {}
        self.global_target_mean: float = 0.0

        self.categorical_columns: list[str] = []
        self.dummy_columns: list[str] = []

        self.numeric_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}

        self.standard_columns: list[str] = []
        self.standard_means: dict[str, float] = {}
        self.standard_stds: dict[str, float] = {}

        self.mlb: MultiLabelBinarizer | None = None
        self.feature_names: list[str] = []

        self.drop_columns: list[str] = []
        self.all_feature_columns: list[str] = []

        self.target_column: str = "Log_Price"
        self.fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: pd.Series | None = None,
        *,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        target_column: str = "Log_Price",
        target_enc_columns: list[str] | None = None,
        standard_columns: list[str] | None = None,
    ) -> InferencePreprocessor:
        cat = categorical_columns or [
            "Condition", "Fuel_type", "Drive", "Type", "Doors_number"
        ]
        num = numeric_columns or [
            "Production_year", "Mileage_km", "Power_HP", "Displacement_cm3"
        ]
        drop = drop_columns or [
            "Index", "Vehicle_brand", "Vehicle_model", "Vehicle_version",
            "Vehicle_generation", "Brand_Model", "Currency", "Price",
            "CO2_emissions", "Transmission", "Colour", "Origin_country",
            "First_owner", "First_registration_date", "Offer_publication_date",
            "Offer_location", "Features", "feat_list",
        ]
        target_enc = target_enc_columns or ["Brand_Model", "Vehicle_brand"]
        std_cols = standard_columns or num

        self.target_column = target_column
        self.categorical_columns = list(cat)
        self.numeric_columns = list(num)
        self.drop_columns = list(drop)
        self.target_enc_columns = list(target_enc)
        self.standard_columns = [c for c in std_cols if c in num]

        if y_train is None:
            y_train = train_df[target_column]

        self.global_target_mean = float(y_train.mean())
        self.target_maps = {}
        for col in self.target_enc_columns:
            target_col = target_column if target_column in train_df.columns else "Price"
            self.target_maps[col] = train_df.groupby(col)[target_col].mean()

        self.numeric_medians = {c: float(train_df[c].median()) for c in self.numeric_columns}

        train_dummies = pd.get_dummies(train_df[self.categorical_columns], columns=self.categorical_columns, dummy_na=True)
        self.dummy_columns = list(train_dummies.columns)

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(train_df["feat_list"])
        self.feature_names = [f"feature_{name}" for name in self.mlb.classes_]

        for c in self.standard_columns:
            self.standard_means[c] = float(train_df[c].mean())
            self.standard_stds[c] = float(train_df[c].std())

        base_cols = [
            c for c in train_df.columns
            if c not in self.drop_columns
            and c not in self.categorical_columns
            and c != self.target_column
        ]
        encoded_cols = [f"{col}_Encoded" for col in self.target_enc_columns]
        self.all_feature_columns = []
        seen: set[str] = set()
        for col in base_cols + encoded_cols + self.dummy_columns + self.feature_names:
            if col not in seen:
                self.all_feature_columns.append(col)
                seen.add(col)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")

        df = df.copy()

        for col in self.target_enc_columns:
            if col in df.columns:
                enc_col = f"{col}_Encoded"
                df[enc_col] = df[col].map(self.target_maps.get(col, pd.Series(dtype=float))).fillna(self.global_target_mean)

        for c in self.numeric_columns:
            if c in df.columns:
                df[c] = df[c].fillna(self.numeric_medians.get(c, 0.0))

        present_cats = [c for c in self.categorical_columns if c in df.columns]
        dummies = pd.get_dummies(df[present_cats], columns=present_cats, dummy_na=True)
        for col in self.dummy_columns:
            if col not in dummies.columns:
                dummies[col] = 0.0

        if self.mlb is not None and "feat_list" in df.columns:
            feat_bin = self.mlb.transform(df["feat_list"])
            feat_df = pd.DataFrame(feat_bin, columns=self.feature_names, index=df.index)
        else:
            feat_df = pd.DataFrame(0, index=df.index, columns=self.feature_names, dtype=float)

        drop_set = set(self.drop_columns)
        base = df.drop(columns=[c for c in df.columns if c in drop_set or c in present_cats], errors="ignore")
        if self.target_column in base.columns:
            base = base.drop(columns=[self.target_column])

        X = pd.concat([base, dummies[self.dummy_columns], feat_df], axis=1)

        for c in self.standard_columns:
            if c in X.columns:
                mean = self.standard_means.get(c, 0.0)
                std = self.standard_stds.get(c, 1.0)
                if std > 0:
                    X[c] = (X[c] - mean) / std
                else:
                    X[c] = X[c] - mean

        for col in self.all_feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=self.all_feature_columns, fill_value=0.0).fillna(0.0).astype(float)

        return X

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> InferencePreprocessor:
        return joblib.load(path)
