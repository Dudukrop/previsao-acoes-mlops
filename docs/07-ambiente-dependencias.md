# 07 — Ambiente Virtualizado e Dependências

## 1. Estratégia

Duas camadas, como em qualquer projeto .NET com `global.json` + Docker:

1. **Ambiente local de desenvolvimento**: `venv` + `requirements.txt` — para treinar o modelo,
   rodar notebooks, testes.
2. **Ambiente de produção (API)**: imagem Docker, construída a partir de um `requirements-api.txt`
   enxuto (não inclui `pmdarima`, `jupyter`, `matplotlib` — só o necessário para servir a API).

Separar as duas listas evita subir para produção dependências pesadas usadas só em treino
(reduz tamanho de imagem e superfície de ataque).

> **Nota:** as versões abaixo são um ponto de partida coerente (compatíveis entre si na data de
> escrita deste documento), mas não foram validadas por uma resolução real de dependências neste
> ambiente. Ao implementar, rode `pip install -r requirements.txt` de fato, resolva qualquer
> conflito que o `pip` reportar, e trave (`pip freeze`) as versões que realmente funcionaram juntas
> antes de considerar o ambiente "pronto" — não assuma que os números abaixo são finais.

## 2. `requirements.txt` (ambiente completo de desenvolvimento/treino)

```txt
# requirements.txt — ambiente de treino/desenvolvimento
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.40
scikit-learn==1.5.0
tensorflow==2.16.1
statsmodels==0.14.2
pmdarima==2.0.4
joblib==1.4.2
pyyaml==6.0.1
pydantic==2.7.4
python-dotenv==1.0.1
matplotlib==3.9.0
seaborn==0.13.2
jupyter==1.0.0
pytest==8.2.2
```

## 3. `requirements-api.txt` (ambiente enxuto de produção)

```txt
# requirements-api.txt — apenas o necessário para servir predições
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
pyyaml==6.0.1
python-dotenv==1.0.1
numpy==1.26.4
scikit-learn==1.5.0
tensorflow-cpu==2.16.1
joblib==1.4.2
```

`tensorflow-cpu` (não `tensorflow` completo) — a API só faz inferência, não precisa de suporte a
GPU/treino, e a imagem fica significativamente menor.

## 4. Setup do ambiente local

```powershell
# PowerShell (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Bash (Linux/Mac/WSL)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Dockerfile da API

Local: `Dockerfile` (raiz do projeto).

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY src/config.py ./src/config.py
COPY config.yaml .
COPY models/artifacts/ ./models/artifacts/
COPY models/metadata/ ./models/metadata/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Forma shell (sem colchetes), não exec form: necessário para a variável ${PORT} ser expandida
# pelo shell do container. Plataformas de deploy (Render/Railway) injetam $PORT em runtime; ${PORT:-8000}
# garante que `docker run -p 8000:8000` local (sem $PORT definida) continue funcionando igual.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Pontos deliberados:
- `python:3.13-slim` — versão usada para travar as dependências deste projeto (ver nota acima);
  `tensorflow-cpu` não publica wheel para toda versão de Python, então a imagem base precisa
  acompanhar a versão testada, não a mais nova disponível por padrão.
- Copia **apenas** o necessário para servir (não copia `data/`, `notebooks/`, `tests/`) — imagem
  final enxuta e sem dados potencialmente sensíveis/desnecessários.
- `${PORT:-8000}` já vem pronta para deploy desde este Dockerfile (ver doc 09) — não há uma versão
  "de desenvolvimento" com porta fixa e outra "de produção" com porta dinâmica; é o mesmo
  Dockerfile nos dois ambientes, como determina a ADR-03 (doc 01).

## 6. `docker-compose.yml` (execução local, opcional mas recomendado)

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./monitoring/logs:/app/monitoring/logs
```

O volume garante que os logs de predição persistam fora do container, mesmo se ele for recriado.

## 7. Comandos de verificação (rodar antes de qualquer commit)

```bash
docker build -t previsao-acoes-api .
docker run --rm -p 8000:8000 --env-file .env previsao-acoes-api
curl http://localhost:8000/health
```

## 8. `.gitignore` (itens obrigatórios)

```gitignore
.venv/
__pycache__/
*.pyc
.env
data/raw/*.csv
monitoring/logs/*.jsonl
.ipynb_checkpoints/
```

`data/raw/*.csv` fica de fora do git por ser dado regenerável via `run_collection.py` — evita
poluir o repositório com CSVs grandes (decisão documentada, com instrução clara no README de como
regenerar).
