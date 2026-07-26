# Índice da Documentação Técnica — Previsão de Cotação de Ações (MLOps)

Este conjunto de documentos especifica, de forma vinculante, cada etapa do projeto exigido pelo
desafio "Machine Learning Engineering - Fase 5". A ideia é que qualquer pessoa (ou o próprio autor,
meses depois) consiga implementar o sistema apenas seguindo estes documentos, sem precisar
redescobrir decisões de design.

## Convenção de leitura

- Cada documento é autocontido, mas assume os contratos definidos em [11-estrutura-projeto.md](11-estrutura-projeto.md)
  (layout de pastas) e em [12-configuracao.md](12-configuracao.md) (schema de configuração central).
- Blocos de código em Python representam **contratos de interface** (assinaturas de função,
  classes Pydantic, tipos). São o equivalente às *interfaces* e *DTOs* de um projeto C#/.NET:
  a implementação interna pode variar, mas a assinatura pública e os tipos de entrada/saída são fixos.
- Todo campo de dado tem tipo explícito. Nenhum `dict` solto ou `Any` sem justificativa — igual a
  não aceitar `object` sem necessidade em C#.

## Ordem de leitura recomendada (= ordem de implementação)

1. [01-arquitetura-geral.md](01-arquitetura-geral.md) — visão geral, diagrama de componentes, decisões de arquitetura.
2. [12-configuracao.md](12-configuracao.md) — schema de configuração (ticker, janelas, paths). Ler antes das demais.
3. [02-coleta-dados.md](02-coleta-dados.md) — ingestão de dados históricos (yfinance).
4. [03-preprocessamento-eda.md](03-preprocessamento-eda.md) — limpeza, EDA, engenharia de features, split temporal.
5. [04-modelagem-treinamento.md](04-modelagem-treinamento.md) — escolha e treino do modelo.
6. [05-avaliacao-metricas.md](05-avaliacao-metricas.md) — métricas, baseline, critérios de aceite.
7. [06-serializacao-versionamento.md](06-serializacao-versionamento.md) — serialização e registro de modelo.
8. [07-ambiente-dependencias.md](07-ambiente-dependencias.md) — ambiente virtual, Docker, dependências travadas.
9. [08-api-especificacao.md](08-api-especificacao.md) — contrato da API REST (FastAPI).
10. [09-deploy-mlops.md](09-deploy-mlops.md) — pipeline de CI/CD e deploy.
11. [10-monitoramento-observabilidade.md](10-monitoramento-observabilidade.md) — logging, drift, alertas.
12. [11-estrutura-projeto.md](11-estrutura-projeto.md) — árvore de diretórios e convenções de código.
13. [13-roteiro-video.md](13-roteiro-video.md) — roteiro do vídeo de apresentação (5+ min).

## Status

| Documento | Status | Bloqueia implementação de |
|---|---|---|
| Arquitetura geral | Especificado | Tudo |
| Configuração central | Especificado | Todos os módulos |
| Coleta de dados | Especificado | Preprocessamento |
| Preprocessamento/EDA | Especificado | Treinamento |
| Modelagem/Treinamento | Especificado | Avaliação, Serialização |
| Avaliação/Métricas | Especificado | Critério de "modelo pronto para deploy" |
| Serialização/Versionamento | Especificado | API |
| Ambiente/Dependências | Especificado | Deploy |
| API | Especificado | Deploy |
| Deploy/MLOps | Especificado | Entrega final |
| Monitoramento | Especificado | Entrega final |
| Estrutura do projeto | Especificado | — |

Nenhum código foi implementado ainda. Este é o pacote de especificação completo para implementação
subsequente.
