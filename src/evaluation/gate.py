class ModelNotAcceptedError(Exception):
    pass


def check_acceptance(
    mape_model: float, mape_naive: float, max_mape_pct: float, must_beat_naive: bool
) -> None:
    """Levanta ModelNotAcceptedError somente se o hard gate (max_mape_pct) falhar — essa é a
    única condição que impede a serialização.

    Se `must_beat_naive=True` e o modelo não superar o naive, NÃO levanta exceção: apenas emite
    um aviso. Soft gate intencional — ver docs/05-avaliacao-metricas.md seção 4 (superar o
    passeio aleatório em previsão de ponto de um dia é notoriamente difícil para preço de
    fechamento diário de ações líquidas)."""
    if mape_model > max_mape_pct:
        raise ModelNotAcceptedError(
            f"MAPE do modelo ({mape_model:.2f}%) excede o limite aceito ({max_mape_pct}%)."
        )
    if must_beat_naive and mape_model >= mape_naive:
        print(
            f"[AVISO] MAPE do modelo ({mape_model:.2f}%) não supera o baseline naive "
            f"({mape_naive:.2f}%). Serialização prossegue; documentar esta constatação no README."
        )
