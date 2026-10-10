"""Carteira em 10 fatias: cada pregão uma fatia compra os sinais do dia e segura 10 pregões."""

import pandas as pd
import pytest

from src.laboratorio.carteira import simular_compras


def _retornos(n=4, r=0.01):
    dias = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"A": [r] * n, "B": [0.0] * n}, index=dias)


def test_conta_na_mao_com_custos():
    ret = _retornos()
    sinais = pd.DataFrame({"data": [ret.index[0]], "ticker": ["A"], "prob": [0.7], "sinal": ["compra"]})
    serie = simular_compras(sinais, ret, ret.index[0], ret.index[-1], horizonte=2, custo=0.01)
    valor = (1 + serie).cumprod()
    # dia 0: fatia 0 (0,5) compra A pagando 1% -> 0,495; fatia 1 em caixa (0,5)
    assert valor.iloc[0] == pytest.approx(0.995)
    # dia 1: A rende 1% -> 0,49995
    assert valor.iloc[1] == pytest.approx(0.99995)
    # dia 2: A rende 1% -> 0,5049495; fatia 0 vende pagando 1% -> 0,49990001
    assert valor.iloc[2] == pytest.approx(0.5049495 * 0.99 + 0.5)
    assert valor.iloc[3] == pytest.approx(valor.iloc[2])     # tudo em caixa


def test_sem_sinais_carteira_fica_parada():
    ret = _retornos()
    vazio = pd.DataFrame(columns=["data", "ticker", "prob", "sinal"])
    assert (simular_compras(vazio, ret, ret.index[0], ret.index[-1]) == 0).all()


def test_sinal_de_venda_nao_vira_compra():
    ret = _retornos()
    vendas = pd.DataFrame({"data": [ret.index[0]], "ticker": ["A"], "prob": [0.2], "sinal": ["venda"]})
    assert (simular_compras(vendas, ret, ret.index[0], ret.index[-1]) == 0).all()


def test_ticker_sem_retorno_conta_como_zero():
    ret = _retornos()
    sinais = pd.DataFrame({"data": [ret.index[0]], "ticker": ["SUMIU3"], "prob": [0.9], "sinal": ["compra"]})
    valor = (1 + simular_compras(sinais, ret, ret.index[0], ret.index[-1], horizonte=2, custo=0.0)).cumprod()
    assert valor.iloc[-1] == pytest.approx(1.0)
