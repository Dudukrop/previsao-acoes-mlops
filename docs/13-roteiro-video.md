# 13 — Roteiro do Vídeo de Apresentação

Roteiro para o vídeo de 5+ minutos exigido na entrega, explicando a estratégia de MLOps
empregada. Não é um texto para decorar palavra por palavra — são os pontos-chave de cada bloco,
com tempo sugerido e o que mostrar na tela. Tempo total sugerido: **~6-7 minutos** (dá margem
confortável acima do mínimo de 5).

Grave com a tela dividida entre você falando (webcam, opcional) e a tela compartilhada mostrando
o que está descrito em "Mostrar na tela" de cada bloco.

---

## 1. Abertura (0:00 – 0:20)

**Falar:**
- Seu nome, o curso/fase ("Machine Learning Engineering — Fase 5"), e o objetivo do vídeo:
  apresentar o projeto de previsão de cotação de ações e a estratégia de MLOps usada para
  colocá-lo em produção.

**Mostrar na tela:** README.md do repositório no GitHub (topo da página).

---

## 2. Contexto e objetivo (0:20 – 0:50)

**Falar:**
- O desafio: prever o fechamento do próximo pregão de uma ação e servir isso via API, com
  monitoramento em produção.
- Por que isso é interessante do ponto de vista de MLOps: não é só treinar um modelo, é a
  pipeline inteira — coleta, treino, avaliação com critério de aceite, versionamento,
  containerização, deploy e observabilidade.

**Mostrar na tela:** o diagrama de arquitetura do README (seção 3, o mermaid com as 3 fases:
Offline, Online, Observabilidade).

---

## 3. Empresa escolhida e dados (0:50 – 1:30)

**Falar:**
- Ticker escolhido: **PETR4.SA** (Petrobras) — alta liquidez, histórico longo (2015 até hoje),
  sem lacunas relevantes de pregão.
- Fonte de dados: `yfinance`, sem custo, sem necessidade de API key.
- O ticker é 100% configurável (`config.yaml`/`.env`) — trocar de empresa não exige mudar código.

**Mostrar na tela:** `config.yaml` (bloco `data:`) e rapidamente `src/data_collection/fetch_data.py`.

---

## 4. Algoritmo escolhido (1:30 – 2:15)

**Falar:**
- Três modelos treinados e comparados no mesmo protocolo: **Naive** (baseline ingênuo), **ARIMA**
  (baseline estatístico, via `auto_arima`) e **LSTM** (modelo de produção, rede recorrente).
- Por que os três: ter um baseline estatístico e um ingênuo é o que permite *provar* que o LSTM
  agrega valor (ou não) — não adianta ter um MAPE bonito sem comparação.
- A avaliação é sempre **one-step-ahead**: cada previsão usa só dados até o dia anterior,
  simulando exatamente como a API se comporta em produção.

**Mostrar na tela:** `src/training/train_lstm.py` (a função `build_lstm_model`) e a tabela
comparativa do README (seção 5).

---

## 5. A parte mais importante: avaliação, métricas e um bug real encontrado (2:15 – 3:30)

Este é o bloco que mais demonstra profundidade de engenharia — não pule.

**Falar:**
- Métricas usadas: MAE, RMSE e MAPE. MAPE é o critério de aceite (hard gate: só serializa se
  MAPE de teste ≤ 5%).
- **A descoberta:** na primeira implementação, o LSTM normalizava o preço absoluto com um
  `MinMaxScaler` ajustado só no treino. Só que a PETR4 valorizou cerca de **17 vezes** entre 2015
  (início do treino) e 2026 (período de teste) — então o scaler ficava com uma faixa
  completamente incompatível com o preço real do teste, e o modelo extrapolava mal. Resultado:
  MAPE de ~5,9%, **pior que o baseline naive** (~1,1%).
- **A correção:** trocar o alvo do modelo de preço absoluto para **retorno logarítmico**
  (`log(Close_t / Close_{t-1})`), que é aproximadamente estacionário e não sofre desse problema
  de mudança de nível ao longo dos anos. Resultado: MAPE caiu para ~1,14%, no mesmo patamar do
  ARIMA e do naive.
- Isso só foi descoberto porque o pipeline foi **executado de ponta a ponta de verdade**, não só
  especificado — reforça a importância de validar hipóteses com dados reais.

**Mostrar na tela:** `models/metadata/evaluation_report.json` (o JSON com os 3 modelos e métricas)
e a seção 4 de `docs/03-preprocessamento-eda.md` (a explicação do bug/correção).

---

## 6. Serialização, versionamento e ambiente (3:30 – 4:00)

**Falar:**
- O modelo aprovado é serializado (`.keras` + `scaler.joblib`) com versionamento incremental
  (`v1`, `v2`, ...) e metadados completos (métricas, config usada no treino, versões de
  framework) — permite rollback manual trocando só um ponteiro (`current_version.json`).
- Ambiente: `requirements.txt` (treino) e `requirements-api.txt` (produção, mais enxuto), e a
  mesma imagem Docker roda local e em produção.

**Mostrar na tela:** `models/metadata/model_metadata_v2.json` e o `Dockerfile`.

---

## 7. Demo da API ao vivo (4:00 – 5:00)

**Falar enquanto demonstra:**
- Abrir a documentação Swagger da API em produção.
- Mostrar o endpoint principal `/predict/by-ticker` — o cliente manda só o ticker, e é a própria
  API que busca o histórico recente e monta a previsão (não precisa o cliente montar a janela de
  preços manualmente).
- Rodar uma predição ao vivo e mostrar a resposta.

**Mostrar na tela (ao vivo, no navegador ou terminal):**
```bash
curl https://previsao-acoes-mlops.onrender.com/health

curl -X POST https://previsao-acoes-mlops.onrender.com/predict/by-ticker \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'
```
Abrir também `https://previsao-acoes-mlops.onrender.com/docs` e mostrar os schemas.

> Se a API estiver "dormindo" (cold start do free tier do Render), avise no vídeo que a primeira
> chamada pode demorar 30-60s — é esperado, não é erro.

---

## 8. Estratégia de deploy e CI/CD (5:00 – 5:40)

**Falar:**
- Deploy via container Docker no Render, deploy automático a cada push na branch `main`.
- CI no GitHub Actions: roda os testes automatizados e builda a imagem Docker antes de qualquer
  deploy real acontecer.
- Mencionar rapidamente que já passou por outra plataforma (Railway) e a migração foi trivial
  porque a estratégia é baseada em container — a mesma imagem funciona em qualquer provedor com
  suporte a Docker.

**Mostrar na tela:** aba **Actions** do GitHub (o workflow rodando/verde) e o painel do Render.

---

## 9. Monitoramento em produção (5:40 – 6:15)

**Falar:**
- Cada predição é logada (`monitoring/logs/predictions.jsonl`) com o valor previsto, a data-alvo,
  e metadados da entrada.
- O job `run_monitoring.py` compara, depois que a data-alvo já passou, a previsão com o
  fechamento real (buscado via yfinance), calcula erro de produção e verifica *drift* de entrada
  (se os preços recebidos fogem muito da distribuição vista no treino).

**Mostrar na tela:** rodar `python -m src.monitoring.run_monitoring` localmente (se já houver
predições acumuladas) e mostrar o `monitoring_report_*.json` gerado.

---

## 10. Encerramento (6:15 – 6:45)

**Falar:**
- Resumo rápido: pipeline completo, modelo validado com critério de aceite objetivo, API em
  produção, monitoramento implementado.
- Limitações conhecidas (mencionar 1-2, ex.: não considera feriados de bolsa no cálculo da
  próxima data, sem re-treino automático) — mostra maturidade reconhecer isso.
- Agradecimento/encerramento.

**Mostrar na tela:** seção 9 do README (Limitações e próximos passos).

---

## Checklist antes de gravar

- [ ] API em produção respondendo (`curl .../health` primeiro, para "acordar" o serviço antes de
      gravar, evitando esperar o cold start ao vivo).
- [ ] Repositório GitHub aberto em uma aba.
- [ ] `evaluation_report.json` e `model_metadata_v2.json` abertos em outra aba/editor.
- [ ] Terminal com o venv ativado, pronto para rodar comandos ao vivo se quiser.
- [ ] Cronômetro rodando — o mínimo exigido é 5 minutos, esse roteiro mira ~6-7.
