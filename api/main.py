import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import load_active_model
from api.routers import health, predict

load_dotenv()

MODEL_METADATA_DIR = os.getenv("MODEL_METADATA_DIR", "models/metadata")
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "monitoring/logs/predictions.jsonl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_model = load_active_model(MODEL_METADATA_DIR)
    app.state.prediction_log_path = PREDICTION_LOG_PATH
    yield


app = FastAPI(title="Previsão de Cotação de Ações - API", lifespan=lifespan)

# CORS amplo: API pública de demonstração acadêmica, sem autenticação/dados sensíveis (doc 08 §8).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
