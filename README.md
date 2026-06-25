# ASI Car Price

Predykcja ceny samochodu na podstawie ogłoszeń. Target: Log_Price.

## Status

- [x] Punkt 1 - organizacja
- [x] Punkt 2 - baseline notebook
- [x] Punkt 3 - Kedro pipeline
- [x] Punkt 4 - Optuna, AutoGluon, MLflow, selekcja cech
- [x] Punkt 5 - FastAPI, monitoring, Docker
- [x] Punkt 6 - DVC, CI/CD (GitHub Actions)

## Szybki start

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Dane: `data/raw/Car_sale_ads.csv`

```bash
kedro run                          # pipeline bazowy
kedro run --pipeline model_optimization  # punkt 4
python scripts/fit_preprocessor.py       # preprocessor dla API
uvicorn api.main:app --reload            # API na http://localhost:8000
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
