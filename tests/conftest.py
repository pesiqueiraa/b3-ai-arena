"""
Dados inventados para os testes, sem internet.
Os preços são um passeio aleatório: o futuro é imprevisível por construção, então nenhum modelo
honesto deveria achar padrão neles.
"""

import numpy as np
import pandas as pd
import pytest

import config


def gerar_precos(n_acoes: int = 20, inicio: str = "2015-01-01", fim: str = "2017-12-31",
                 semente: int = 7) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Devolve (precos, calendario). Colunas: data, ticker, fechamento_ajustado, volume (com BOVA11)."""
    rng = np.random.default_rng(semente)
    calendario = pd.DatetimeIndex(pd.bdate_range(inicio, fim))
    tickers = [f"AC{i:02d}3" for i in range(n_acoes)] + [config.REFERENCIA]
    partes = []
    for ticker in tickers:
        preco = 20 * np.exp(np.cumsum(rng.normal(0, 0.015, len(calendario))))
        partes.append(pd.DataFrame({
            "data": calendario,
            "ticker": ticker,
            "fechamento_ajustado": preco,
            "volume": rng.lognormal(15, 0.4, len(calendario)).round(),
        }))
    return pd.concat(partes, ignore_index=True), calendario


def universo_completo(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Universo em que todas as ações (menos o BOVA11) estão na lista em todos os meses."""
    primeiros = pd.Series(calendario).groupby(calendario.to_period("M")).first()
    acoes = sorted(t for t in precos["ticker"].unique() if t != config.REFERENCIA)
    linhas = [
        {"mes": str(mes), "data_inicio": dia, "ticker": t, "posicao": i + 1, "volume_medio": 1.0}
        for mes, dia in primeiros.items() for i, t in enumerate(acoes)
    ]
    return pd.DataFrame(linhas)


@pytest.fixture(scope="session")
def sinteticos():
    return gerar_precos()


@pytest.fixture(scope="session")
def base_sintetica(sinteticos):
    from src.base import montar_base  # criado na Tarefa 7

    precos, calendario = sinteticos
    return montar_base(precos, universo_completo(precos, calendario), calendario)
