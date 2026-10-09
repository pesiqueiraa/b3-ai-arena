"""
Testes da métrica principal (Brier) e das comparações pré-registradas (PREREGISTRO.md).

Rodar:  python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import config
from src import avaliacao
from src.modelos.ml import walk_forward


@pytest.fixture(scope="module")
def gabarito(indicadores):
    return avaliacao.separar_gabarito(indicadores)[0]


@pytest.fixture(scope="module")
def janela(gabarito):
    return gabarito[gabarito["periodo"] == "teste_limpo"].dropna(subset=["gabarito"])


def _decisoes(janela, prob):
    d = janela[["data", "ticker"]].copy()
    d["prob_subir"] = prob
    d["decisao"] = np.where(d["prob_subir"] > 0.5, "comprar", "ficar_de_fora")
    return d


def test_brier_dos_casos_conhecidos():
    subiu = pd.Series([1, 0, 1, 0])
    assert avaliacao.brier(pd.Series([1.0, 0.0, 1.0, 0.0]), subiu) == 0.0   # acertou tudo com certeza
    assert avaliacao.brier(pd.Series([0.5] * 4), subiu) == 0.25            # sempre 50%
    assert avaliacao.brier(pd.Series([0.0, 1.0, 0.0, 1.0]), subiu) == 1.0   # errou tudo com certeza


def test_referencia_nao_olha_o_futuro(gabarito, janela):
    """A taxa histórica do dia D não pode mudar se o gabarito de D em diante mudar."""
    dia = sorted(janela["data"].unique())[10]
    original = avaliacao.prob_referencia(gabarito)

    bagunca = gabarito.copy()
    futuro = bagunca["data"] >= dia
    bagunca.loc[futuro, "gabarito"] = np.random.default_rng(1).normal(size=futuro.sum())
    assert avaliacao.prob_referencia(bagunca)[dia] == original[dia]

    # E ela é exatamente a fração de altas até a véspera.
    antes = gabarito[gabarito["data"] < dia].dropna(subset=["gabarito"])
    assert original[dia] == pytest.approx((antes["gabarito"] > 0).mean())


def test_modelo_contra_ele_mesmo_da_empate(gabarito, janela):
    d = _decisoes(janela, 0.6)
    c = avaliacao.comparar_brier(d, d, gabarito)
    assert c["dif_media"] == 0.0


def test_trapaceiro_vence_o_ml_com_folga(gabarito, janela, acoes, dias_teste, monkeypatch):
    monkeypatch.setattr(config, "PREGOES_MINIMOS", 10)
    trapaceiro = _decisoes(janela, (janela["gabarito"] > 0).astype(float).values)
    ml = walk_forward.gerar_decisoes(acoes, dias_teste)

    c = avaliacao.comparar_brier(trapaceiro, ml, gabarito)
    assert c["ic_alto"] < 0
    assert avaliacao.veredito(c) == "A melhor que B"
    assert avaliacao.veredito(avaliacao.comparar_brier(ml, trapaceiro, gabarito)) == "B melhor que A"


def test_veredito_fica_parcial_antes_do_minimo(gabarito, janela, monkeypatch):
    monkeypatch.setattr(config, "PREGOES_MINIMOS", 10_000)
    trapaceiro = _decisoes(janela, (janela["gabarito"] > 0).astype(float).values)
    c = avaliacao.comparar_brier(trapaceiro, _decisoes(janela, 0.5), gabarito)
    assert avaliacao.veredito(c).startswith("parcial")


def test_comparacao_usa_so_pares_em_comum(gabarito, janela):
    a = _decisoes(janela, 0.6)
    b = _decisoes(janela, 0.4).iloc[: len(janela) // 2]
    assert avaliacao.comparar_brier(a, b, gabarito)["pares"] == len(b)


def test_comparacoes_principais_aparecem_quando_jev_e_llm_existem(gabarito, janela):
    rng = np.random.default_rng(5)
    todas = {nome: _decisoes(janela, rng.uniform(0.4, 0.6, len(janela)))
             for nome in ["jev", "llm", "ml"]}
    todas["sempre_comprar"] = _decisoes(janela, 1.0)

    c = avaliacao.comparacoes_preregistradas(todas, gabarito)
    principais = c[c["tipo"] == "principal"]
    assert set(zip(principais["modelo_a"], principais["modelo_b"])) == {("jev", "ml"), ("jev", "llm")}
    assert (principais["confianca"] == 0.975).all()
    assert ("llm", "ml") in set(zip(c["modelo_a"], c["modelo_b"]))
    assert "sempre_comprar" not in set(c["modelo_a"])  # não é modelo de probabilidade
