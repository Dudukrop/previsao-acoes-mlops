# 06 — Serialização e Versionamento de Modelo

## 1. Artefatos a serializar

| Artefato | Formato | Path |
|---|---|---|
| Modelo LSTM | `.keras` (formato nativo Keras 3, recomendado sobre `.h5`) | `models/artifacts/lstm_model_v{N}.keras` |
| Scaler | `joblib` (`sklearn.preprocessing.MinMaxScaler`) | `models/artifacts/scaler_v{N}.joblib` |
| Metadados do modelo | JSON | `models/metadata/model_metadata_v{N}.json` |
| Ponteiro para versão ativa | JSON | `models/metadata/current_version.json` |

`{N}` é um inteiro incremental (versionamento simples, análogo a versionar um pacote NuGet —
não sobrescreve versões anteriores, permite rollback).

## 2. Contrato do módulo

Local: `src/training/serialize_model.py`

```python
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler

def next_version(metadata_dir: str) -> int:
    """Lê os arquivos model_metadata_v*.json existentes e retorna max(N) + 1. Retorna 1 se
    nenhum artefato existir ainda (primeira serialização)."""
    existing = list(Path(metadata_dir).glob("model_metadata_v*.json"))
    if not existing:
        return 1
    versions = [int(p.stem.split("_v")[-1]) for p in existing]
    return max(versions) + 1

def serialize_artifacts(
    model: Model, scaler: MinMaxScaler,
    artifact_dir: str, metadata_dir: str,
    ticker: str, lookback_window: int,
    evaluation_metrics: dict, config_snapshot: dict,
) -> int:
    """Serializa modelo + scaler + metadados com versão incremental. Retorna o número da versão
    criada. NÃO promove automaticamente a versão a 'current' — isso é uma etapa manual/explícita
    via `promote_version()`, para permitir revisão humana antes de ir para produção."""
    version = next_version(metadata_dir)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)

    model_path = f"{artifact_dir}/lstm_model_v{version}.keras"
    scaler_path = f"{artifact_dir}/scaler_v{version}.joblib"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "version": version,
        "ticker": ticker,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lookback_window": lookback_window,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "evaluation_metrics": evaluation_metrics,
        "config_snapshot": config_snapshot,
        "framework_versions": {
            "tensorflow": _get_version("tensorflow"),
            "scikit-learn": _get_version("sklearn"),
        },
    }
    Path(f"{metadata_dir}/model_metadata_v{version}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return version

def promote_version(metadata_dir: str, version: int) -> None:
    """Grava current_version.json apontando para `version`. É este arquivo que a API lê no
    startup — nunca o maior número de versão diretamente, para permitir fixar uma versão anterior
    caso uma nova versão se mostre pior em produção (rollback manual)."""
    Path(f"{metadata_dir}/current_version.json").write_text(
        json.dumps({"active_version": version}), encoding="utf-8"
    )

def _get_version(package_name: str) -> str:
    import importlib.metadata
    return importlib.metadata.version(package_name)
```

## 3. Schema de `model_metadata_v{N}.json` (contrato consumido pela API)

```json
{
  "version": 1,
  "ticker": "PETR4.SA",
  "created_at": "2026-07-16T00:00:00Z",
  "lookback_window": 60,
  "target_transform": "log_return",
  "model_path": "models/artifacts/lstm_model_v1.keras",
  "scaler_path": "models/artifacts/scaler_v1.joblib",
  "evaluation_metrics": {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
  "train_close_stats": {"mean": 0.0, "std": 0.0},
  "config_snapshot": { "...": "cópia integral do config.yaml usado neste treino" },
  "framework_versions": {"tensorflow": "2.16.1", "scikit-learn": "1.5.0"}
}
```

`target_transform` documenta que o modelo prevê retorno logarítmico, não preço absoluto (ver doc
03 seção 4 e doc 04 seção 4). `train_close_stats` (média/desvio-padrão do `Close` de treino) é
consumido pelo job de monitoramento (doc 10) para detectar drift de entrada.

`config_snapshot` é a cópia literal do `config.yaml` no momento do treino — garante
reprodutibilidade total mesmo que o `config.yaml` do repositório mude depois.

## 4. Schema de `current_version.json`

```json
{ "active_version": 1 }
```

## 5. Regra de negócio: só serializa o que passou no gate de aceite

`serialize_artifacts` só deve ser chamado **depois** de `check_acceptance()` (doc 05) não ter
levantado `ModelNotAcceptedError`. O script `src/training/run_training.py` (ou um script separado
`run_promote.py`) encadeia: treinar → avaliar → (se aprovado) serializar → promover.

## 6. Por que não usar MLflow/DVC neste projeto

Ferramentas como MLflow Model Registry ou DVC resolveriam este mesmo problema de forma mais
robusta em um cenário real de equipe. Para o escopo do desafio (um único desenvolvedor, um
ticket, entrega em repositório GitHub simples), a convenção de versionamento por arquivo JSON
descrita acima é suficiente e evita dependência de infraestrutura adicional no deploy gratuito.
Documentar esta decisão explicitamente no README como trade-off consciente.
