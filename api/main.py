import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Sem isso, quem abrir o domínio raiz (sem saber os endpoints de cor) recebe 404 — redireciona
    para a documentação Swagger, que é o ponto de entrada natural para quem está explorando a API."""
    return RedirectResponse(url="/docs")
