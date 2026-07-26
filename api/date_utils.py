from datetime import date, timedelta


def next_trading_day(reference_date: date) -> date:
    """Retorna o próximo dia útil (segunda a sexta) após `reference_date`, pulando fim de semana.

    Limitação conhecida e aceita para o escopo do projeto: NÃO considera feriados de bolsa
    (B3/NYSE), apenas fins de semana.
    """
    next_day = reference_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=sábado, 6=domingo
        next_day += timedelta(days=1)
    return next_day
