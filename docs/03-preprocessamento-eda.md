# 03 — Preprocessamento e EDA

## 1. Objetivo da etapa

Transformar o CSV bruto (`data/raw/{ticker}.csv`) em conjuntos de treino/validação/teste prontos
para consumo direto pelo treinamento, sem nenhuma decisão de negócio pendente dentro do script de
treino.

## 2. EDA obrigatória (notebook, `notebooks/01_eda.ipynb`)

Não é código de produção, mas suas conclusões **determinam** parâmetros do `config.yaml`. Passos:

1. **Estatísticas descritivas** de `Close` e `Volume` (`df.describe()`).
2. **Gráfico de série temporal** de `Close` — identificar tendência e possíveis regimes distintos
   (ex.: crise, mudança estrutural).
3. **Teste de estacionariedade (ADF — Augmented Dickey-Fuller)** sobre `Close` e sobre
   `Close.diff()`. Resultado esperado: série em nível é não-estacionária (p-valor > 0.05); a
   primeira diferença deve ser estacionária. Isso justifica `d=1` no ARIMA (`config.yaml →
   model.arima.order`).
4. **Autocorrelação (ACF/PACF)** sobre a série diferenciada — usada para justificar `p` e `q`
   iniciais do ARIMA (refinados depois via `auto_arima`, ver doc 04).
5. **Decomposição sazonal** (`statsmodels.tsa.seasonal_decompose`) — documentar se há sazonalidade
   relevante (para ações, tipicamente fraca/ausente; documentar o resultado encontrado).
6. Conclusões da EDA vão para uma seção do `README.md` final (não repetir o notebook inteiro, só o
   resumo com os gráficos-chave).

## 3. Contrato do módulo de preprocessamento

Local: `src/preprocessing/prepare_dataset.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_raw(raw_path: str) -> pd.DataFrame:
    """Lê o CSV salvo pela etapa de coleta, reconstruindo o DatetimeIndex."""
    df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    return df.sort_index()

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas de índice (mantém a última ocorrência) e linhas com Close nulo.
    Não faz interpolação de dias faltantes: dias sem pregão são omitidos por design, não
    preenchidos artificialmente, para não introduzir viés na série."""
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])

def chronological_split(
    df: pd.DataFrame, test_size_ratio: float, validation_size_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide em train/validation/test respeitando ORDEM CRONOLÓGICA (jamais shuffle).

    Layout resultante, do mais antigo para o mais recente:
    [ ---- train ---- ][ -- validation -- ][ -- test -- ]

    Justificativa: embaralhar série temporal vaza informação futura para o treino (data leakage).
    """
    n = len(df)
    test_n = int(n * test_size_ratio)
    val_n = int(n * validation_size_ratio)
    train_end = n - test_n - val_n
    val_end = n - test_n
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def make_windows(
    series: np.ndarray, lookback_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gera pares (X, y) por janela deslizante para o LSTM.

    X.shape == (n_amostras, lookback_window, 1)
    y.shape == (n_amostras,)

    Exemplo com lookback_window=3: série [10,11,12,13,14] gera
      X[0] = [10,11,12] -> y[0] = 13
      X[1] = [11,12,13] -> y[1] = 14
    """
    X, y = [], []
    for i in range(len(series) - lookback_window):
        X.append(series[i : i + lookback_window])
        y.append(series[i + lookback_window])
    X = np.array(X).reshape(-1, lookback_window, 1)
    y = np.array(y)
    return X, y
```

## 4. Alvo do LSTM: retorno logarítmico, não preço absoluto — e normalização

**Correção introduzida após validação prática (rodando o pipeline de ponta a ponta):** a
especificação original desta seção normalizava o `Close` absoluto com `MinMaxScaler` ajustado só
no treino. Isso quebra na prática para ações com forte valorização/desvalorização ao longo dos
anos — ex.: PETR4.SA saiu de ~R$2 (2015, início do treino) para ~R$40 (2026, período de teste).
Um `MinMaxScaler` ajustado no range de treino fica com min/max completamente incompatíveis com o
nível de preço do teste, e o modelo passa a extrapolar bem fora da faixa em que foi treinado — na
prática, isso produziu MAPE de ~5-6% no LSTM (pior que o baseline naive de ~1,1%).

**Correção:** o alvo do LSTM é o **retorno logarítmico diário**, `log(Close_t / Close_{t-1})`
(`src/preprocessing/prepare_dataset.py::compute_log_returns`), calculado sobre a série completa
**antes** do split (para preservar continuidade: o primeiro retorno de validation usa o último
Close de train, e o primeiro retorno de test usa o último Close de validation). Retorno é
aproximadamente estacionário — sua distribuição não se desloca sistematicamente com o nível de
preço — então o `MinMaxScaler` ajustado só no treino continua válido para validação/teste.

- Ajustar (`fit`) um `sklearn.preprocessing.MinMaxScaler` sobre a coluna `log_return`
  **somente do conjunto de treino**. Aplicar (`transform`) o mesmo scaler em validação e teste.
- **Regra crítica anti-vazamento de dados:** é proibido chamar `scaler.fit()` ou `fit_transform()`
  em dados de validação/teste.
- Para reconstruir o preço previsto a partir do retorno previsto:
  `Close_previsto(t) = Close_real(t-1) * exp(retorno_previsto(t))`. Isso significa que prever
  `Close(t)` exige conhecer o `Close` real do dia anterior — tanto no treino/avaliação (doc 04)
  quanto na API (doc 08, que por isso exige `lookback_window + 1` preços, não `lookback_window`).
- O `scaler` ajustado (sobre retornos, não preços) é um artefato que **também precisa ser
  serializado** junto com o modelo (ver [06-serializacao-versionamento.md](06-serializacao-versionamento.md)).
- O ARIMA e o Naive **não usam esse scaler** — continuam operando diretamente sobre `Close`
  absoluto, sem normalização (não sofrem do mesmo problema por não terem uma faixa de
  entrada/saída fixa aprendida em treino).

## 5. Script executável

Local: `src/preprocessing/run_preprocessing.py`

```python
"""
Uso:
    python -m src.preprocessing.run_preprocessing

Lê data/raw/{ticker}.csv, grava em data/processed/:
    train.parquet, validation.parquet, test.parquet   (dados em nível, para o ARIMA)
    scaler.joblib                                       (ajustado apenas no train)
"""
```

## 6. Saída (contrato para a próxima etapa)

| Arquivo | Conteúdo | Consumido por |
|---|---|---|
| `data/processed/train.parquet` | Colunas OHLCV, período mais antigo | Treino LSTM e ARIMA |
| `data/processed/validation.parquet` | Período intermediário | Early stopping do LSTM, seleção de hiperparâmetros |
| `data/processed/test.parquet` | Período mais recente, nunca visto no treino | Avaliação final (doc 05) |
| `data/processed/scaler.joblib` | `MinMaxScaler` ajustado no `log_return` de treino (não no `Close`) | Treino, Avaliação e API |

## 7. Checklist de qualidade antes de prosseguir para o treino

- `train.index.max() < validation.index.min() < test.index.min()` — nenhuma sobreposição temporal.
- `len(test) >= lookback_window + 1` — senão não é possível gerar nem uma janela de teste.
- Distribuição do `Close` em treino vs. teste plotada lado a lado — mudança de regime muito
  abrupta deve ser documentada como limitação conhecida do modelo.
