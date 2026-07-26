# 09 — Deploy e Pipeline de MLOps

## 1. Plataforma de deploy

Recomendação: **Render** (free tier com Docker nativo, HTTPS automático, sem cartão de crédito).
Alternativas equivalentes documentadas para o caso de indisponibilidade do free tier: **Railway**
ou **Hugging Face Spaces (Docker Space)**.

## 2. Porta dinâmica

Render/Railway injetam a variável de ambiente `$PORT` em runtime. O `Dockerfile` (doc 07) já é
escrito desde o início com `CMD uvicorn ... --port ${PORT:-8000}` — não há ajuste a fazer aqui;
a mesma imagem funciona local (`docker run -p 8000:8000 ...`, sem `$PORT` definida, cai no
fallback `8000`) e em produção (Render/Railway definem `$PORT` automaticamente).

## 3. Passo a passo — Render

1. Criar conta em render.com, conectar a conta GitHub.
2. "New +" → "Web Service" → selecionar o repositório do projeto.
3. **Environment:** `Docker` (Render detecta o `Dockerfile` automaticamente).
4. **Region:** qualquer uma próxima (ex.: Ohio).
5. **Instance type:** Free.
6. **Environment Variables:** adicionar `TICKER`, `MODEL_ARTIFACT_DIR`, `MODEL_METADATA_DIR`,
   `PREDICTION_LOG_PATH` (mesmos nomes do `.env`, ver doc 12) — **nunca** commitar o `.env` real,
   configurar direto no painel do Render.
7. Deploy automático dispara a cada push na branch principal (padrão do Render).
8. Após o build, testar `https://{seu-servico}.onrender.com/health`.

**Limitação conhecida do free tier:** o serviço "dorme" após ~15 min de inatividade e a primeira
requisição subsequente pode levar 30-60s (cold start). Documentar isso no README para quem for
testar a API não achar que travou.

## 4. Passo a passo — Railway (alternativa)

1. railway.app → "New Project" → "Deploy from GitHub repo".
2. Railway detecta o `Dockerfile` automaticamente.
3. Aba "Variables" → adicionar as mesmas variáveis de ambiente do passo Render acima.
4. Railway expõe a porta via `$PORT` automaticamente (mesmo ajuste de Dockerfile da seção 2).
5. Gerar domínio público em "Settings" → "Networking" → "Generate Domain".

## 5. CI — GitHub Actions (validação antes do deploy)

Local: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-api.txt

      - name: Run tests
        run: pytest tests/ -v

      - name: Build Docker image
        run: docker build -t previsao-acoes-api:ci .
```

Este workflow garante que todo push para `main` (o que o Render/Railway vão usar para deploy) já
passou nos testes automatizados e builda a imagem Docker sem erros **antes** do deploy real
acontecer — equivalente a um pipeline de build/test do Azure DevOps antes de um release.

> **Pré-requisito importante:** `tests/test_api.py` instancia `TestClient(app)`, o que dispara o
> evento de startup da API e tenta carregar `models/artifacts/` + `models/metadata/` (doc 08,
> seção 4). Isso só funciona em CI se esses artefatos já estiverem commitados no repositório (ver
> checklist da seção 7 abaixo). Em um clone/fork limpo **antes** da primeira execução do pipeline
> de treino (doc 04-06), este workflow falha na etapa de testes — não é um bug do CI, é uma
> dependência de ordem esperada: rode o pipeline de treino e commite os artefatos gerados antes de
> confiar neste workflow.

## 6. Testes automatizados mínimos (`tests/`)

| Arquivo | O que valida |
|---|---|
| `tests/test_metrics.py` | Fórmulas de MAE/RMSE/MAPE contra valores calculados manualmente |
| `tests/test_preprocessing.py` | `chronological_split` não sobrepõe períodos; `make_windows` gera shapes corretos |
| `tests/test_api.py` | `TestClient(app)` do FastAPI: `/health` retorna 200; `/predict` com payload inválido retorna 422; com tamanho errado de `closes` retorna 400 |

```python
# tests/test_api.py (esqueleto)
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_rejects_empty_closes():
    resp = client.post("/predict", json={"closes": []})
    assert resp.status_code == 422
```

## 7. Checklist de entrega (mapeado 1:1 para o enunciado do desafio)

- [ ] Repositório GitHub público com todo o código, `docs/`, `Dockerfile`, `requirements*.txt`.
- [ ] `models/artifacts/` e `models/metadata/` commitados (ou disponibilizados via GitHub
      Releases se o tamanho do `.keras` exceder o limite confortável do git — documentar a escolha).
- [ ] Link da API pública funcionando (`/health` e `/docs` acessíveis).
- [ ] Vídeo de 5+ minutos cobrindo: contexto do problema → EDA → escolha do modelo → métricas de
      avaliação → arquitetura de deploy → demo da API rodando (`/docs` ou `curl`) → monitoramento.

## 8. Rollback

Se uma nova versão do modelo (nova promoção, doc 06) piorar o desempenho em produção, o rollback é:
1. Editar `models/metadata/current_version.json` para apontar para a versão anterior.
2. Commit + push → novo deploy automático carrega a versão anterior no próximo startup.

Não há rollback automático baseado em métricas nesta versão do projeto — decisão de escopo,
documentada como possível evolução futura em vez de feature implementada.
