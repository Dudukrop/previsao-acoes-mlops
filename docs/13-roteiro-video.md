# 13 — Roteiro do Vídeo de Apresentação

Roteiro para o vídeo de 5+ minutos exigido na entrega, explicando a estratégia de MLOps
empregada. Cada bloco tem um texto sugerido em bloco de citação — fale com suas próprias
palavras, não precisa decorar. O texto puro soma **~4,5 minutos** de fala; a demonstração ao vivo
da API (bloco 7) naturalmente estica isso para além dos 5 minutos mínimos.

Grave com a tela dividida entre você falando (webcam, opcional) e a tela compartilhada mostrando
o que está descrito em "Mostrar na tela" de cada bloco.

---

## 1. Abertura (0:00 – 0:15)

> "Oi! Esse vídeo é a apresentação do meu projeto da Fase 5 de Machine Learning Engineering: um
> pipeline completo de MLOps pra prever o preço de fechamento de uma ação, do treino até o
> deploy em produção."

**Mostrar na tela:** README.md do repositório no GitHub (topo da página).

---

## 2. Contexto e objetivo (0:15 – 0:35)

> "O desafio era prever o fechamento do próximo pregão de uma ação e colocar isso em produção de
> verdade — não só treinar um modelo, mas construir a pipeline inteira: coleta de dados, treino,
> avaliação com um critério de aceite, versionamento, containerização, deploy e monitoramento."

**Mostrar na tela:** o diagrama de arquitetura do README (seção 3).

---

## 3. Empresa escolhida e dados (0:35 – 0:55)

> "Escolhi a Petrobras, ticker PETR4, por ter bastante liquidez e um histórico longo e contínuo
> desde 2015. Os dados vêm do Yahoo Finance, e o ticker é totalmente configurável — trocar de
> empresa não exige mudar nenhuma linha de código."

**Mostrar na tela:** `config.yaml` (bloco `data:`).

---

## 4. Algoritmo escolhido (0:55 – 1:35)

> "Treinei e comparei três modelos, todos avaliados da mesma forma. O primeiro é um baseline bem
> simples, que só chuta que o preço de amanhã é igual ao de hoje — parece bobo, mas é essencial:
> se o modelo de verdade não bater isso, não faz sentido usar machine learning ali. O segundo é
> um ARIMA, ajustado automaticamente. E o terceiro, o modelo principal, que fica servindo a API,
> é uma LSTM — uma rede neural recorrente feita pra aprender padrões em sequências.
>
> Comparar os três é o que me permite provar, com números, se a complexidade extra da LSTM
> realmente compensa. E toda avaliação é 'um passo à frente': o modelo só usa dados até o dia
> anterior, exatamente como vai se comportar em produção."

**Mostrar na tela:** a tabela comparativa do README (seção 5).

---

## 5. Avaliação, métricas e um bug real encontrado (1:35 – 2:20)

Este é o bloco que mais demonstra profundidade de engenharia — não pule.

> "As métricas usadas foram erro absoluto, erro quadrático e o MAPE, que é o erro percentual —
> meu critério de aceite é MAPE de teste abaixo de 5%.
>
> E aqui teve uma descoberta real. Na primeira versão, o LSTM previa o preço absoluto,
> normalizado por um scaler ajustado só com os dados de treino. Só que a Petrobras valorizou
> cerca de 17 vezes entre 2015 e 2026, então esse scaler ficava incompatível com o preço de
> teste, e o modelo errava mais que o baseline simples.
>
> A correção foi prever o retorno logarítmico em vez do preço absoluto, o que resolveu o
> problema e trouxe o MAPE pra 1,14%, no mesmo nível do ARIMA. Isso só apareceu porque rodei o
> pipeline de ponta a ponta com dados reais, não só no papel."

**Mostrar na tela:** `models/metadata/evaluation_report.json`.

---

## 6. Serialização, versionamento e ambiente (2:20 – 2:40)

> "O modelo aprovado é serializado com versionamento incremental e metadados completos, o que
> permite fazer rollback manual trocando só um ponteiro. E o ambiente todo — treino e produção —
> roda a partir da mesma imagem Docker, com dependências travadas em arquivos separados pra
> treino e pra API."

**Mostrar na tela:** `models/metadata/model_metadata_v2.json` e o `Dockerfile`.

---

## 7. Demo da API ao vivo (2:40 – 3:50)

> "Agora vou mostrar a API rodando de verdade em produção. Esse é o endpoint principal: eu mando
> só o ticker, e é a própria API que busca o histórico recente e faz a previsão — o cliente não
> precisa montar nada manualmente."

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

## 8. Estratégia de deploy e CI/CD (3:50 – 4:15)

> "O deploy é feito via container Docker, com deploy automático a cada push na branch principal.
> Antes disso, um workflow no GitHub Actions roda os testes automatizados e builda a imagem,
> garantindo que só vai pro ar o que passou nos testes. Como tudo é baseado em container, migrar
> de plataforma — como precisei fazer quando o plano gratuito do Railway acabou — foi trivial."

**Mostrar na tela:** aba **Actions** do GitHub e o painel do Render.

---

## 9. Monitoramento em produção (4:15 – 4:40)

> "Toda predição fica registrada em um log, com o valor previsto e a data-alvo. Um job separado
> de monitoramento compara, depois que essa data já passou, a previsão com o fechamento real, e
> verifica se os dados recebidos estão fugindo muito do que o modelo viu no treino."

**Mostrar na tela:** rodar `python -m src.monitoring.run_monitoring` e mostrar o
`monitoring_report_*.json` gerado.

---

## 10. Encerramento (4:40 – 5:00)

> "Resumindo: pipeline completo, modelo validado com um critério de aceite objetivo, API em
> produção e monitoramento implementado. Existem limitações conhecidas, documentadas no
> repositório. Obrigado por assistir!"

**Mostrar na tela:** seção 9 do README (Limitações e próximos passos).

---

## Checklist antes de gravar

- [ ] API em produção respondendo (`curl .../health` primeiro, para "acordar" o serviço antes de
      gravar, evitando esperar o cold start ao vivo).
- [ ] Repositório GitHub aberto em uma aba.
- [ ] `evaluation_report.json` e `model_metadata_v2.json` abertos em outra aba/editor.
- [ ] Terminal com o venv ativado, pronto para rodar comandos ao vivo.
- [ ] Cronômetro rodando — o texto puro soma ~5 min, e a demonstração ao vivo deve levar o vídeo
      além do mínimo de 5 minutos exigido.
