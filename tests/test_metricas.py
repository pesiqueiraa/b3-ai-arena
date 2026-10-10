"""Métricas do relatório e o teste contra a sorte."""

import numpy as np
import pandas as pd
import pytest

from src.laboratorio import metricas


def test_brier_casos_conhecidos():
    assert metricas.brier([1.0, 0.0], [1, 0]) == 0.0
    assert metricas.brier([0.5, 0.5], [1, 0]) == 0.25


def test_referencia_usa_so_rotulos_ja_conhecidos():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    tabela = pd.DataFrame({"data": np.repeat(cal, 2), "ticker": ["A", "B"] * 30, "rotulo": [1, 0] * 30})
    ref = metricas.prob_referencia(tabela, cal, horizonte=10)
    assert np.isnan(ref[cal[5]])                                  # nada conhecido ainda
    assert ref[cal[20]] == pytest.approx(0.5)
    bagunca = tabela.copy()
    bagunca.loc[bagunca["data"] > cal[10], "rotulo"] = 1          # muda rótulos ainda não conhecidos em cal[20]
    assert metricas.prob_referencia(bagunca, cal, horizonte=10)[cal[20]] == pytest.approx(0.5)


def test_acerto_de_compras_e_vendas():
    tabela = pd.DataFrame({"data": pd.Timestamp("2024-01-02"), "ticker": ["A", "B", "C"], "excesso": [0.02, -0.01, np.nan]})
    sinais = pd.DataFrame({"data": pd.Timestamp("2024-01-02"), "ticker": ["A", "B", "C"], "prob": 0.5,
                           "sinal": ["compra", "venda", "compra"]})
    assert metricas.acerto_sinais(sinais, tabela, "compra") == 1.0     # C sem resultado fica fora
    assert metricas.acerto_sinais(sinais, tabela, "venda") == 1.0


def test_pior_queda_conta_perda_desde_o_capital_inicial():
    assert metricas.pior_queda(pd.Series([-0.20, 0.0, 0.0])) == pytest.approx(-0.20)


def test_resumo_da_carteira():
    dias = pd.to_datetime(["2023-06-01", "2023-06-02", "2024-06-03"])
    ret = pd.Series([0.10, -0.10, 0.05], index=dias)
    bova = pd.Series([0.0, 0.0, 0.10], index=dias)
    r = metricas.resumo_carteira(ret, bova)
    assert r["retorno_total"] == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert r["pior_queda"] == pytest.approx(-0.10)
    assert r["fracao_anos_bons"] == 0.0                           # 2023: -1% vs 0%; 2024: 5% vs 10%
    assert r["excesso_medio_anual"] == pytest.approx(((1.1 * 0.9 - 1) + (0.05 - 0.10)) / 2)


def _tabela_aleatoria(n_dias=60, n_acoes=30, semente=1):
    rng = np.random.default_rng(semente)
    dias = pd.bdate_range("2024-01-01", periods=n_dias)
    return pd.DataFrame([{"data": d, "ticker": f"T{i:02d}", "excesso": rng.normal(0, 0.05)}
                         for d in dias for i in range(n_acoes)])


def test_trapaceiro_bate_todos_os_aleatorios():
    tabela = _tabela_aleatoria()
    melhores = tabela.sort_values(["data", "excesso"], ascending=[True, False]).groupby("data").head(5)
    sinais = melhores.assign(prob=0.9, sinal="compra")[["data", "ticker", "prob", "sinal"]]
    assert metricas.bateu_aleatorios(sinais, tabela, 500, np.random.default_rng(0)) == 1.0


def test_escolha_ao_acaso_nao_parece_habilidade():
    """Uma escolha ao acaso supera, em média, ~metade dos agentes (média de 10 sorteios, para não depender de sorte)."""
    tabela = _tabela_aleatoria()
    fracoes = []
    for semente in range(10):
        sorteadas = tabela.groupby("data").sample(5, random_state=semente)
        sinais = sorteadas.assign(prob=0.9, sinal="compra")[["data", "ticker", "prob", "sinal"]]
        fracoes.append(metricas.bateu_aleatorios(sinais, tabela, 300, np.random.default_rng(100 + semente)))
    assert 0.2 < np.mean(fracoes) < 0.8


def test_sem_compras_nao_ha_teste():
    tabela = _tabela_aleatoria(n_dias=5)
    vazio = pd.DataFrame(columns=["data", "ticker", "prob", "sinal"])
    assert np.isnan(metricas.bateu_aleatorios(vazio, tabela, 100, np.random.default_rng(0)))
