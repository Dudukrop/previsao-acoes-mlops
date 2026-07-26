import time
import uuid
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Request

from api.date_utils import next_trading_day
from api.inference import predict_next_close
from api.logging_middleware import log_prediction
from api.schemas import PredictByTickerRequest, PredictRequest, PredictResponse
from src.data_collection.fetch_data import fetch_historical_data

router = APIRouter()

FETCH_MARGIN_CALENDAR_DAYS = 15  # cobre feriados + fins de semana sem 2ª tentativa na prática


@router.post("/predict", response_model=PredictResponse)
def predict_manual(payload: PredictRequest, request: Request) -> PredictResponse:
    loaded = request.app.state.loaded_model
    expected = loaded.lookback_window + 1  # N+1 preços -> N retornos (ver api/inference.py)
    if len(payload.closes) != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Esperado exatamente {expected} valores em 'closes', "
                f"recebido {len(payload.closes)}."
            ),
        )

    start = time.perf_counter()
    try:
        predicted = predict_next_close(loaded, payload.closes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao gerar predição.")
    latency_ms = (time.perf_counter() - start) * 1000

    prediction_date = next_trading_day(date.today())
    request_id = str(uuid.uuid4())

    log_prediction(
        request.app.state.prediction_log_path, loaded.ticker, loaded.version,
        payload.closes, predicted, str(prediction_date), latency_ms, request_id,
    )

    return PredictResponse(
        ticker=loaded.ticker, predicted_close=predicted,
        prediction_for_date=str(prediction_date),
        model_version=loaded.version, request_id=request_id,
    )


@router.post("/predict/by-ticker", response_model=PredictResponse)
def predict_by_ticker(payload: PredictByTickerRequest, request: Request) -> PredictResponse:
    loaded = request.app.state.loaded_model
    ticker = payload.ticker or loaded.ticker
    if ticker != loaded.ticker:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Este serviço só prevê {loaded.ticker}; ticker '{ticker}' não corresponde "
                "ao modelo carregado."
            ),
        )

    fetch_start = date.today() - timedelta(
        days=loaded.lookback_window * 2 + FETCH_MARGIN_CALENDAR_DAYS
    )
    try:
        df = fetch_historical_data(ticker, fetch_start.isoformat(), None)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Não foi possível obter dados de mercado no momento. Tente novamente.",
        )

    required_closes = loaded.lookback_window + 1
    if len(df) < required_closes:
        raise HTTPException(
            status_code=502,
            detail="Histórico insuficiente retornado pela fonte de dados para montar a janela de previsão.",
        )

    closes = df["Close"].iloc[-required_closes:].tolist()

    fetch_end = time.perf_counter()
    try:
        predicted = predict_next_close(loaded, closes)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao gerar predição.")
    latency_ms = (time.perf_counter() - fetch_end) * 1000

    last_date = df.index[-1].date()
    prediction_date = next_trading_day(last_date)
    request_id = str(uuid.uuid4())

    log_prediction(
        request.app.state.prediction_log_path, loaded.ticker, loaded.version,
        closes, predicted, str(prediction_date), latency_ms, request_id,
    )

    return PredictResponse(
        ticker=loaded.ticker, predicted_close=predicted,
        prediction_for_date=str(prediction_date),
        model_version=loaded.version, request_id=request_id,
    )
