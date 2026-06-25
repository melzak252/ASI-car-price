import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from api.preprocessing import InferencePreprocessor
from features.build_features import add_brand_model, add_log_price
from preprocessing.cleaning import clean_features, filter_doors, remove_eur_offers, remove_log_price_iqr_outliers


def main():
    with open(PROJECT_ROOT / "conf" / "base" / "parameters.yml") as f:
        params = yaml.safe_load(f)

    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Car_sale_ads.csv")
    df = add_log_price(raw)
    df = remove_log_price_iqr_outliers(df, params["target"])
    df = remove_eur_offers(df)
    df = filter_doors(df, max_doors=7)
    df = add_brand_model(df)
    df["feat_list"] = df["Features"].apply(clean_features)

    train_df, _ = train_test_split(df, test_size=params["test_size"], random_state=params["random_state"])
    y_train = train_df[params["target"]]

    preproc = InferencePreprocessor()
    preproc.fit(train_df, y_train, categorical_columns=params["categorical_columns"],
                numeric_columns=params["numeric_columns"], drop_columns=params["drop_columns"],
                target_column=params["target"])
    preproc.save(str(PROJECT_ROOT / "models" / "preprocessor.pkl"))

    sel_path = PROJECT_ROOT / "reports" / "selected_features.json"
    if sel_path.exists():
        with open(sel_path) as f:
            selected = json.load(f)
        joblib.dump(selected, PROJECT_ROOT / "models" / "selected_feature_columns.joblib")

    print(f"Preprocessor fitted and saved ({len(preproc.all_feature_columns)} features)")


if __name__ == "__main__":
    main()
