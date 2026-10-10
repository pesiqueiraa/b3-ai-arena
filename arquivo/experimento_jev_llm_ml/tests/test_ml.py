"""
Testes das correções feitas na revisão dos scripts originais.
Cada teste corresponde a um problema encontrado (ver "Correções" no README).

Rodar:  python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import config
from src import avaliacao, dados, decisoes
from src.modelos.ml import walk_forward


# --- Correção 1: pregão incompleto (linha do dia com preços zerados) ----------

def test_remove_pregao_incompleto_mas_mantem_dia_sem_negocio(precos):
    p = precos.copy()
    ultimo = p["data"].max()
    p.loc[p["data"] == ultimo, ["abertura", "maxima", "minima", "volume"]] = 0  # "08/10/2026"
    dia_sem_negocio = p["data"].unique()[100]
    p.loc[(p["data"] == dia_sem_negocio) & (p["ticker"] == "AAAA3"), "volume"] = 0  # "26/04/2024"

    limpo = dados.remover_pregoes_incompletos(p)
    assert ultimo not in set(limpo["data"])
    assert ((limpo["data"] == dia_sem_negocio) & (limpo["ticker"] == "AAAA3")).sum() == 1


# --- Correção 2: indicador vazio não pode fazer o modelo "pular" o dia ---------

def test_ml_decide_mesmo_com_indicador_vazio(acoes, dias_teste):
    dia = dias_teste[5]
    com_buraco = acoes.copy()
    com_buraco.loc[com_buraco["data"] == dia, "volume_relativo_20d"] = np.nan

    saida = walk_forward.decidir_um_dia(com_buraco, dia)
    assert len(saida) == (acoes["data"] == dia).sum()
    assert saida["prob_subir"].notna().all()


def test_avaliador_conta_pares_faltando(indicadores):
    gabarito, _ = avaliacao.separar_gabarito(indicadores)
    janela = gabarito[gabarito["periodo"] == "teste_limpo"][["data", "ticker"]]
    completo = janela.assign(prob_subir=0.6, decisao="comprar")
    assert avaliacao.faltando_na_janela(completo, gabarito) == 0
    assert avaliacao.faltando_na_janela(completo.iloc[3:], gabarito) == 3


# --- Correção 3: sorteio dos aleatórios próprio de cada modelo -----------------

def test_sorteio_de_um_modelo_nao_depende_dos_outros():
    a1 = avaliacao.rng_do_modelo("ml").random(5)
    avaliacao.rng_do_modelo("jev").random(1000)  # outro modelo sorteando no meio
    a2 = avaliacao.rng_do_modelo("ml").random(5)
    b = avaliacao.rng_do_modelo("llm").random(5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


# --- Correção 4: acerto de direção usa a própria decisão -----------------------

def test_acerto_de_direcao_segue_a_decisao(indicadores):
    gabarito, bova = avaliacao.separar_gabarito(indicadores)
    janela = gabarito[gabarito["periodo"] == "teste_limpo"].dropna(subset=["gabarito"])
    # prob_subir "na beirada": 0,5 arredondado, mas a decisão foi comprar.
    d = janela[["data", "ticker"]].assign(prob_subir=0.5, decisao="comprar")
    r = avaliacao.avaliar_modelo("x", d, gabarito, bova, avaliacao.rng_do_modelo("x"))
    assert r["acerto_de_direcao"] == pytest.approx((janela["gabarito"] > 0).mean())


# --- Arquivos de decisão: formato (regra 3) e "só acréscimo" -------------------

@pytest.fixture
def pasta_decisoes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_DECISOES", tmp_path)
    monkeypatch.setattr(config, "RAIZ", tmp_path.parent)
    return tmp_path


def _decisoes(dias, prob):
    return pd.DataFrame({
        "data": pd.to_datetime(dias), "ticker": "AAAA3",
        "prob_subir": prob, "decisao": "comprar",
    })


def test_decisoes_gravadas_nao_sao_reescritas(pasta_decisoes):
    decisoes.acrescentar(_decisoes(["2026-09-15", "2026-09-16"], 0.7), "teste")
    assert decisoes.dias_pendentes("teste", pd.to_datetime(["2026-09-16", "2026-09-17"])) == \
        [pd.Timestamp("2026-09-17")]

    decisoes.acrescentar(_decisoes(["2026-09-17"], 0.2), "teste")
    gravado = decisoes.carregar("teste")
    assert len(gravado) == 3
    assert list(gravado["prob_subir"]) == [0.7, 0.7, 0.2]


def test_formato_invalido_e_recusado(pasta_decisoes):
    ruim = _decisoes(["2026-09-15"], 0.7).assign(gabarito_retorno_amanha=0.01)
    with pytest.raises(ValueError, match="não permitidas"):
        decisoes.acrescentar(ruim, "teste")
    with pytest.raises(ValueError, match="entre 0 e 1"):
        decisoes.acrescentar(_decisoes(["2026-09-15"], 1.5), "teste")
    with pytest.raises(ValueError, match="inválidos"):
        decisoes.acrescentar(_decisoes(["2026-09-15"], 0.5).assign(decisao="vender"), "teste")
