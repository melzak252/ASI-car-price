# ASI Car Price

Predykcja ceny samochodu na podstawie ogłoszeń. Target: Log_Price.

## Architektura

```mermaid
flowchart LR
    CSV[Car_sale_ads.csv] --> DP[data_processing<br/>Kedro]
    DP --> PT[train.parquet<br/>test.parquet]
    PT --> DS[data_science<br/>Kedro]
    DS --> LGBM[LightGBM]
    DS --> METRICS[metrics.json]
    PT --> MO[model_optimization<br/>Kedro]
    MO --> FS[Feature Selection]
    MO --> OPT[Optuna tuning]
    MO --> AG[AutoGluon]
    MO --> CMP[Model Comparison]
    MO --> BEST[best_model.pkl]
    BEST --> PP[fit_preprocessor.py]
    PP --> PREP[preprocessor.pkl]
    PREP --> API[FastAPI /predict]
    CSV --> DVC1[dvc: data/raw]
    MODELS[models/*] --> DVC2[dvc: models/]
    API --> LOG[predictions_log.csv]
    API --> DRIFT[z-score drift]
```

## Struktura

```
asi-car-price/
├── api/                    # FastAPI
│   ├── main.py             # /health, /predict, /drift, /predictions
│   ├── preprocessing.py    # InferencePreprocessor
│   ├── schemas.py          # Pydantic modele
│   └── monitoring.py       # logowanie + drift
├── conf/base/              # Kedro config
│   ├── catalog.yml         # dataset registry
│   └── parameters.yml      # parametry pipeline'u
├── data/
│   ├── raw/                # oryginalny CSV (DVC)
│   ├── processed/          # train/test parquet (DVC)
│   └── 05_model_input/     # X/y pickle (DVC)
├── models/                 # .pkl, .joblib (DVC)
├── scripts/
│   └── fit_preprocessor.py # dopasowanie preprocessora
├── src/
│   ├── car_price/          # Kedro pipelines
│   │   └── pipelines/
│   │       ├── data_processing/   # preprocessing
│   │       ├── data_science/      # trenowanie
│   │       └── model_optimization/# Optuna, AutoGluon
│   ├── features/           # feature engineering
│   ├── models/             # tuning, automl
│   ├── preprocessing/      # czyszczenie
│   ├── evaluation/         # metryki
│   └── tracking/           # MLflow utils
├── tests/
├── reports/                # metrics, params, comparison
├── Dockerfile
├── .github/workflows/ci.yml
└── wymagania.pdf
```

## Szybki start

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Dane: `data/raw/Car_sale_ads.csv`

### Pipeline bazowy
```bash
kedro run                          # data_processing + data_science
```

`data_processing`: czyszczenie (outliers, EUR, doors), dodanie log_price, Brand_Model, lista cech, podział train/test, target encoding.  
`data_science`: trenowanie LightGBM, ewaluacja (R²_log, MAE_PLN, RMSE_PLN), MLflow log.

### Pipeline optymalizacji
```bash
kedro run --pipeline model_optimization  # ~8 minut
```

1. AutoGluon (ensemble modeli)
2. Feature selection (LightGBM importance, top 50)
3. Optuna hyperparameter tuning (30 triali, 3-fold CV)
4. Porównanie Ridge vs HistGBR vs LightGBM default vs LightGBM tuned
5. Wybór najlepszego modelu → `models/best_model.pkl`

### API
```bash
python scripts/fit_preprocessor.py       # dopasowanie preprocessora
uvicorn api.main:app --reload            # http://localhost:8000
```

## API

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| GET | /health | Status |
| POST | /predict | Predykcja ceny |
| GET | /predictions | Historia |
| GET | /drift | Dryf danych |

### Przykład

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"vehicle_brand":"BMW","vehicle_model":"X5","production_year":2020,"mileage_km":50000,"power_hp":300,"fuel_type":"Diesel","type":"SUV","features":["ABS"]}'
```

## Docker

```bash
docker build -t asi-car-price-api .
docker run -p 8000:8000 asi-car-price-api
```

## DVC

Wersjonowanie danych i modeli:

```bash
dvc pull                    # pobranie danych
dvc checkout                # przywrócenie wersji
dvc push                    # wysłanie do storage
dvc status                  # sprawdzenie zmian
```

DVC remote (lokalny): `/tmp/dvc-storage`. Dla produkcji zmień na S3/GDrive/SSH.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) uruchamia testy przy każdym pushu/PR na main.

## Wyniki

| Model | R²_log | MAE_PLN | RMSE_PLN |
|-------|-------:|--------:|---------:|
| AutoGluon | 0.956 | 6 940 | 15 791 |
| LightGBM tuned | 0.953 | 7 342 | 16 698 |
| LightGBM default | 0.950 | 7 991 | 17 886 |
| HistGBR | 0.941 | 9 351 | 20 635 |
| Ridge | 0.811 | 16 792 | 34 238 |
