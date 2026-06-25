import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path
from src.data.load_data import load_car_ads, save_processed_data
from src.evaluation.metrics import car_price_metrics
from src.features.build_features import add_log_price, build_train_test_features
from src.models.predict_model import predict
from src.models.train_model import train_model
from src.preprocessing.cleaning import clean_car_ads


def run_pipeline(config_path: str = "config/config.yaml") -> dict:
    config = load_config(config_path)
    raw_path = resolve_path(config["data"]["raw_path"])
    processed_path = resolve_path(config["data"]["processed_path"])
    model_path = resolve_path(config["model"]["path"])
    metrics_path = resolve_path(config["model"]["metrics_path"])
    target = config["model"].get("target", "Log_Price")
    random_state = config["model"].get("random_state", 42)
    test_size = config["data"].get("test_size", 0.2)
    model_params = config["model"].get("params", {})

    raw_df = load_car_ads(raw_path)
    df = add_log_price(raw_df)
    df = clean_car_ads(df, target)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    x_train, x_test, y_train, y_test, feature_pipeline = build_train_test_features(train_df, test_df, target)
    model = train_model(x_train, y_train, random_state=random_state, params=model_params)
    y_pred = predict(model, x_test)
    metrics = car_price_metrics(y_test, y_pred)

    processed = pd.concat(
        [
            pd.concat([x_train, y_train.rename(target)], axis=1).assign(split="train"),
            pd.concat([x_test, y_test.rename(target)], axis=1).assign(split="test"),
        ],
        axis=0,
    )
    save_processed_data(processed, processed_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_pipeline": feature_pipeline,
            "target": target,
            "metrics": metrics,
        },
        model_path,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows_raw": int(len(raw_df)),
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features_count": int(x_train.shape[1]),
        "model_path": str(model_path),
        "processed_path": str(processed_path),
        "metrics": metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    result = run_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
