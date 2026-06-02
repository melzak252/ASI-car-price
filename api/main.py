from fastapi import FastAPI

app = FastAPI(title="ASI Car Price Prediction API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
