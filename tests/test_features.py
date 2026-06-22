import numpy as np
import pandas as pd

from features.build_features import add_brand_model, add_log_price, build_feature_matrices


def test_add_log_price():
    df = pd.DataFrame({"Price": [np.e, np.e**2]})
    result = add_log_price(df)
    np.testing.assert_allclose(result["Log_Price"], [1.0, 2.0])


def test_add_brand_model_combines_columns():
    df = pd.DataFrame({"Vehicle_brand": ["Audi"], "Vehicle_model": ["A4"]})
    result = add_brand_model(df)
    assert result["Brand_Model"].iloc[0] == "Audi A4"


def _sample_frame(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Log_Price": rng.normal(10, 1, n),
            "Vehicle_brand": rng.choice(["Audi", "BMW"], n),
            "Brand_Model": rng.choice(["Audi A4", "BMW X5"], n),
            "Condition": rng.choice(["Used", "New"], n),
            "Fuel_type": rng.choice(["Gasoline", "Diesel"], n),
            "Drive": rng.choice(["Front wheels", "4x4"], n),
            "Type": rng.choice(["sedan", "suv"], n),
            "Doors_number": rng.choice([3.0, 5.0], n),
            "Production_year": rng.integers(2000, 2020, n).astype(float),
            "Mileage_km": rng.integers(0, 200000, n).astype(float),
            "Power_HP": rng.integers(60, 300, n).astype(float),
            "Displacement_cm3": rng.integers(1000, 3000, n).astype(float),
            "feat_list": [["ABS"] for _ in range(n)],
        }
    )


def test_build_feature_matrices_shapes_and_no_leakage_columns():
    train = _sample_frame(30)
    test = _sample_frame(10)
    categorical = ["Condition", "Fuel_type", "Drive", "Type", "Doors_number"]
    numeric = ["Production_year", "Mileage_km", "Power_HP", "Displacement_cm3"]
    drop = ["Vehicle_brand", "Brand_Model", "feat_list"]

    X_train, X_test, y_train, y_test = build_feature_matrices(
        train, test, categorical, numeric, drop, target_column="Log_Price"
    )


    assert list(X_train.columns) == list(X_test.columns)
    assert len(X_train) == 30 and len(X_test) == 10

    assert "Brand_Model_Encoded" in X_train.columns
    assert "Log_Price" not in X_train.columns

    assert not X_train.isna().any().any()
