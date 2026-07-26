# 11 — Estrutura do Projeto e Convenções de Código

## 1. Árvore de diretórios (contrato final)

```
projeto_fiap/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── inference.py
│   ├── logging_middleware.py
│   └── routers/
│       ├── predict.py
│       └── health.py
├── src/
│   ├── config.py
│   ├── data_collection/
│   │   ├── fetch_data.py
│   │   └── run_collection.py
│   ├── preprocessing/
│   │   ├── prepare_dataset.py
│   │   └── run_preprocessing.py
│   ├── training/
│   │   ├── baseline_naive.py
│   │   ├── train_arima.py
│   │   ├── train_lstm.py
│   │   ├── serialize_model.py
│   │   └── run_training.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── gate.py
│   └── monitoring/
│       └── run_monitoring.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_avaliacao.ipynb
├── data/
│   ├── raw/              # gitignored (regenerável via run_collection.py)
│   └── processed/        # gitignored (regenerável via run_preprocessing.py)
├── models/
│   ├── artifacts/        # .keras, .joblib versionados
│   └── metadata/         # .json versionados + current_version.json
├── monitoring/
│   ├── logs/             # predictions.jsonl (gitignored)
│   └── reports/          # monitoring_report_*.json
├── tests/
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── docs/                 # este pacote de documentação
├── config.yaml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
└── README.md
```

## 2. Convenções de código (aplicam-se a todo `src/` e `api/`)

- **Type hints obrigatórios** em toda assinatura de função pública (parâmetros e retorno) — é o
  equivalente Python de nunca usar `var`/`object` sem necessidade em C#. `mypy` pode ser rodado
  opcionalmente, mas as assinaturas já devem ser tipadas mesmo sem o linter.
- **Docstring obrigatória** em toda função pública que não seja trivialmente óbvia pelo nome —
  documentar contrato (pré-condições, pós-condições, exceções levantadas), não o "o quê" óbvio.
- **Nenhuma função de pipeline lê `config.yaml` diretamente** — sempre recebe `AppConfig` (ou
  campos específicos dele) como parâmetro. Isso torna toda função testável isoladamente com um
  config de teste, sem precisar de arquivo em disco.
- **Fail-fast com exceções nomeadas**, não `return None`/`return False` silencioso para condições
  de erro de negócio (ver `ModelNotAcceptedError` na doc 05).
- **Sem estado global mutável** fora do necessário (a única exceção aceita é `app.state` do
  FastAPI, que é o padrão idiomático do framework para guardar o modelo carregado).
- **Nomenclatura:** `snake_case` para funções/variáveis, `PascalCase` para classes (incluindo
  todos os modelos Pydantic), `UPPER_SNAKE_CASE` para constantes de módulo.

## 3. Testes

- Todo módulo com lógica de negócio não-trivial (`metrics.py`, `prepare_dataset.py`, `gate.py`,
  a API) tem teste correspondente em `tests/`.
- Testes de pipeline de dados usam DataFrames pequenos e sintéticos (não o dataset real baixado),
  para rodar em milissegundos e não depender de rede.
- Rodar com `pytest tests/ -v` — deve ser o único comando necessário para validar a lógica de
  negócio antes de qualquer commit.

## 4. README.md — estrutura mínima obrigatória para a entrega final

1. **Contexto e objetivo** (1 parágrafo).
2. **Empresa/ticker escolhido e justificativa** (2-3 frases).
3. **Arquitetura** (reaproveitar o diagrama de [01-arquitetura-geral.md](01-arquitetura-geral.md)).
4. **Como rodar localmente** (venv, `.env`, comandos exatos dos scripts de coleta/preprocessamento/
   treino/serialização, depois `docker-compose up`).
5. **Resultados do modelo** (tabela comparativa da doc 05 + gráfico real vs. previsto).
6. **Documentação da API** (link para `/docs`, exemplo de `curl`).
7. **Link da API em produção**.
8. **Estratégia de monitoramento** (resumo da doc 10 + captura de um `monitoring_report`).
9. **Limitações conhecidas e próximos passos**.
10. **Link do vídeo de apresentação**.

## 5. Rastreabilidade entre requisito do desafio e documento

| Requisito do enunciado | Documento(s) |
|---|---|
| Escolher empresa e coletar dados históricos | [02](02-coleta-dados.md) |
| Escolher algoritmo de ML para séries temporais | [04](04-modelagem-treinamento.md) |
| Avaliar desempenho com métricas relevantes | [05](05-avaliacao-metricas.md) |
| Serializar o modelo | [06](06-serializacao-versionamento.md) |
| Ambiente virtualizado com dependências | [07](07-ambiente-dependencias.md) |
| API que recebe requisição, extrai dados, envia ao modelo | [08](08-api-especificacao.md) |
| Monitorar desempenho em produção | [10](10-monitoramento-observabilidade.md) |
| Documentação do projeto | Este documento (seção 4) + [00](00-indice.md) |
| Repositório GitHub + link da API + vídeo | [09](09-deploy-mlops.md), seção 7 |
