from fastapi import APIRouter, Request

from api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.loaded_model
    return HealthResponse(status="ok", model_version=loaded.version, ticker=loaded.ticker)


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    loaded = request.app.state.loaded_model
    return ModelInfoResponse(
        version=loaded.version,
        ticker=loaded.ticker,
        created_at=loaded.created_at,
        lookback_window=loaded.lookback_window,
        evaluation_metrics=loaded.evaluation_metrics,
    )
