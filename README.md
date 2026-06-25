# ASI Car Price - predykcja ceny samochodu

Projekt ML przewidujący cenę samochodu na podstawie danych z ogłoszeń (regresja).
Target modelu: `Log_Price` (logarytm ceny). Realizacja kolejnych punktów wymagań
projektu zaliczeniowego.

## Status realizacji wymagań

- **Punkt 1 - organizacja:** repozytorium z historią zmian, środowisko Python.
- **Punkt 2 - baseline:** [notebooks/02_baseline_model.ipynb](notebooks/02_baseline_model.ipynb)
  (EDA, preprocessing, trening, ewaluacja).
- **Punkt 3 - struktura i pipeline:** moduły w `src/` + pipeline ML w **Kedro**.
- **Punkt 4 - udoskonalanie modelu:** pipeline `model_optimization` z MLflow,
  Optuną, AutoGluon, selekcją cech i porównaniem modeli.
- Punkty 5–8: w trakcie / do realizacji.

## Struktura

```
conf/                  # konfiguracja Kedro (catalog, parametry)
data/                  # raw / processed / model input (poza repo)
models/                # zapisany model (poza repo)
notebooks/             # notebooki (baseline)
reports/               # metryki, wykresy
src/
  car_price/           # pakiet projektu Kedro (pipeline_registry, settings)
    pipelines/
      data_processing/ # preprocessing + inżynieria cech
      data_science/    # trening + ewaluacja
  preprocessing/       # czyste funkcje czyszczenia danych
  features/            # czyste funkcje inżynierii cech
  models/              # trening / predykcja
  evaluation/          # metryki
tests/                 # testy jednostkowe
```

## Pipeline ML (Kedro)

Pipeline `__default__` = `data_processing` + `data_science`:

1. **preprocess** - log ceny, usunięcie outlierów (IQR), ofert EUR, nietypowej liczby
   drzwi, budowa `Brand_Model` i listy wyposażenia.
2. **split** - podział train/test (seed z parametrów).
3. **build_features** - target encoding (liczony tylko na train), one-hot, binaryzacja
   wyposażenia, standaryzacja cech numerycznych.
4. **train** - model LightGBM.
5. **evaluate** - metryki (skala log i PLN) do `reports/metrics.json`.

### Potok `model_optimization` (udoskonalanie modelu, punkt 4)

Uruchamiany osobno (kilka minut), nie wchodzi w `__default__`:

1. **feature_selection** - wybór najważniejszych cech wg ważności z LightGBM.
2. **tune_hyperparameters** - strojenie LightGBM przez Optunę (sampler TPE = optymalizacja bayesowska).
3. **compare_models** - trening i porównanie kilku modeli (Ridge, HistGBR, LightGBM domyślny/strojony).
4. **automl** - AutoGluon (`TabularPredictor`) z leaderboardem modeli.
5. **select_best** - tabela porównawcza (`reports/model_comparison.csv`) i zapis wdrażalnego modelu (`models/best_model.pkl`).

Każdy oceniany model jest logowany jako przebieg **MLflow** (lokalne `./mlruns`).

Wyniki ostatniego uruchomienia punktu 4:

| Model | R²_log | MAE_PLN | RMSE_PLN |
|-------|-------:|--------:|---------:|
| AutoGluon / WeightedEnsemble_L2 | 0.956 | 6 940 | 15 791 |
| LightGBM strojony Optuną | 0.953 | 7 342 | 16 698 |
| LightGBM domyślny | 0.950 | 7 991 | 17 886 |
| HistGradientBoostingRegressor | 0.941 | 9 351 | 20 635 |
| Ridge | 0.811 | 16 792 | 34 238 |

Artefakty punktu 4:

- `reports/selected_features.json` - wybrane najważniejsze cechy,
- `reports/best_params.json` - najlepsze parametry Optuny,
- `reports/automl_metrics.json` - metryki najlepszego modelu AutoGluon,
- `reports/automl_leaderboard.csv` - ranking modeli AutoGluon,
- `reports/model_comparison.csv` - porównanie wszystkich kandydatów,
- `models/best_model.pkl` - wdrażalny model LightGBM po strojeniu,
- `mlflow.db`, `mlruns/`, `mlartifacts/` - lokalne śledzenie eksperymentów MLflow.

## Uruchomienie

> **Wymagany Python 3.10–3.13.**

```bash
# 1. Środowisko (zalecany wirtualny venv, Python 3.13)
py -3.13 -m venv .venv          # lub: python3.13 -m venv .venv
.venv/Scripts/activate          # Windows; na Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Dane - umieść Car_sale_ads.csv w data/raw/
#    (ścieżka konfigurowana w conf/base/catalog.yml)

# 3. Uruchomienie pipeline'u
kedro run                          # pełny przepływ
kedro run --pipeline data_processing
kedro run --pipeline data_science

# 4. Udoskonalanie modelu (Optuna + AutoGluon)
kedro run --pipeline model_optimization
mlflow ui --backend-store-uri sqlite:///mlflow.db   # podgląd eksperymentów na http://127.0.0.1:5000

# 5. Wizualizacja grafu pipeline'u
kedro viz

# 6. Testy
pytest
```

## Wyniki (LightGBM, baseline)

| Metryka  | Wartość |
|----------|---------|
| R²_log   | ~0.951  |
| MAE_PLN  | ~7 914  |
| RMSE_PLN | ~17 751 |
