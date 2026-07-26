from pydantic import BaseModel, field_validator


class PredictByTickerRequest(BaseModel):
    ticker: str | None = None  # se None, usa o ticker do modelo carregado


class PredictRequest(BaseModel):
    closes: list[float]

    @field_validator("closes")
    @classmethod
    def validate_closes(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("closes não pode ser vazio")
        if any(p <= 0 for p in v):
            raise ValueError("todos os valores de closes devem ser > 0")
        return v


class PredictResponse(BaseModel):
    ticker: str
    predicted_close: float
    prediction_for_date: str
    model_version: int
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: int
    ticker: str


class ModelInfoResponse(BaseModel):
    version: int
    ticker: str
    created_at: str
    lookback_window: int
    evaluation_metrics: dict
