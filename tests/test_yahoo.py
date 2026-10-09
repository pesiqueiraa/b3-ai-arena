"""Preços ajustados do Yahoo: limpeza e download de vários tickers (com um 'baixar' falso)."""

import pandas as pd

from src.coleta import yahoo


def _tabela(ticker, datas, abertura=10.0):
    return pd.DataFrame({
        "data": pd.to_datetime(datas), "ticker": ticker, "abertura": abertura, "maxima": abertura,
        "minima": abertura, "fechamento_ajustado": 10.0, "volume": 1000.0,
    })


def test_remove_pregao_incompleto_mas_mantem_dia_sem_negocio():
    precos = pd.concat([
        _tabela("AAAA3", ["2026-10-07"]),
        _tabela("AAAA3", ["2026-10-08"], abertura=0.0),                       # dia de hoje pela metade
        _tabela("BBBB3", ["2024-04-26"]).assign(volume=0.0),                  # sem negócio, preço ok
    ])
    limpo = yahoo.remover_pregoes_incompletos(precos)
    assert list(limpo["data"].dt.strftime("%Y-%m-%d")) == ["2026-10-07", "2024-04-26"]


def test_remove_datas_repetidas_ficando_com_a_ultima():
    precos = pd.concat([_tabela("AAAA3", ["2024-01-02"]), _tabela("AAAA3", ["2024-01-02"]).assign(fechamento_ajustado=11.0)])
    limpo = yahoo.remover_datas_repetidas(precos)
    assert len(limpo) == 1 and limpo["fechamento_ajustado"].iloc[0] == 11.0


def test_baixar_varios_junta_e_lista_falhas():
    def baixar_falso(ticker, inicio):
        if ticker == "ERRO3":
            raise ConnectionError("sem internet")
        if ticker == "VAZI3":
            return pd.DataFrame()
        return _tabela(ticker, ["2024-01-02", "2024-01-03"])

    precos, falharam = yahoo.baixar_varios(["AAAA3", "ERRO3", "VAZI3", "BBBB3"], "2024-01-01",
                                           pausa=0, baixar=baixar_falso)
    assert sorted(falharam) == ["ERRO3", "VAZI3"]
    assert list(precos.columns) == ["data", "ticker", "fechamento_ajustado", "volume"]
    assert sorted(precos["ticker"].unique()) == ["AAAA3", "BBBB3"]


def test_baixar_varios_sem_nada_devolve_tabela_vazia():
    precos, falharam = yahoo.baixar_varios(["ERRO3"], "2024-01-01", pausa=0,
                                           baixar=lambda t, i: pd.DataFrame())
    assert precos.empty and falharam == ["ERRO3"]
    assert list(precos.columns) == ["data", "ticker", "fechamento_ajustado", "volume"]
