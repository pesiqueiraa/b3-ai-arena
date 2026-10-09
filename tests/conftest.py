"""
Peças compartilhadas pelos testes: preços inventados (passeio aleatório), sem internet.
Num passeio aleatório o futuro é imprevisível por construção, então NENHUM modelo honesto
deveria acertar a direção muito mais do que 50% das vezes.
"""

import numpy as np
import pandas as pd
import pytest

from src import dados

TICKERS_FALSOS = ["AAAA3", "BBBB4", "CCCC3", "DDDD4", "EEEE3", "FFFF3", "BOVA11"]
DATA_CORTE_FALSA = "2025-06-02"


def gerar_precos(semente: int = 7, n_dias: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    datas = pd.bdate_range("2024-01-01", periods=n_dias)
    partes = []
    for ticker in TICKERS_FALSOS:
        retornos = rng.normal(0, 0.015, n_dias)
        preco = 20 * np.exp(np.cumsum(retornos))
        partes.append(pd.DataFrame({
            "data": datas,
            "ticker": ticker,
            "abertura": preco,
            "maxima": preco * 1.01,
            "minima": preco * 0.99,
            "fechamento": preco,
            "fechamento_ajustado": preco,
            "volume": rng.lognormal(15, 0.4, n_dias).round(),
        }))
    return pd.concat(partes, ignore_index=True).sort_values(["data", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def precos() -> pd.DataFrame:
    return gerar_precos()


@pytest.fixture(scope="session")
def indicadores(precos) -> pd.DataFrame:
    return dados.calcular_indicadores(precos, data_corte=DATA_CORTE_FALSA)


@pytest.fixture(scope="session")
def acoes(indicadores) -> pd.DataFrame:
    """Indicadores sem o BOVA11 (que é régua, não escolha)."""
    return indicadores[indicadores["ticker"] != "BOVA11"].reset_index(drop=True)


@pytest.fixture(scope="session")
def dias_teste(acoes) -> list:
    return sorted(acoes.loc[acoes["periodo"] == "teste_limpo", "data"].unique())
