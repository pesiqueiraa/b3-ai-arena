"""Walk-forward mensal com embargo: o modelo nunca vê rótulos que ainda não existiam."""

import numpy as np
import pandas as pd
import pytest

from src.features.preco import COLUNAS_MODELO
from src.laboratorio import walk_forward as wf
from src.modelos import combinador


def test_datas_de_retreino_sao_o_primeiro_pregao_de_cada_mes():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-31"))
    assert [d.strftime("%m-%d") for d in wf.datas_de_retreino(cal, "2024-01-01", "2024-03-31")] == ["01-01", "02-01", "03-01"]


def test_limite_e_ultimo_dia_avaliavel():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    assert wf.limite_de_treino(cal, cal[15], horizonte=10) == cal[5]
    assert wf.limite_de_treino(cal, cal[5], horizonte=10) is None
    assert wf.ultimo_dia_avaliavel(cal, cal[-1], horizonte=10) == cal[-11]


def test_embargo_o_treino_so_ve_rotulos_ja_conhecidos(base_sintetica):
    vistos = []

    def treinar_espiao(x, y, tipo):
        vistos.append(len(x))
        return combinador.treinar(x, y, tipo)

    b = base_sintetica
    inicio, fim = b.calendario[300], b.calendario[420]
    wf.gerar_probabilidades(b.tabela, b.calendario, inicio, fim, "logistica", treinar=treinar_espiao)
    for dia, n in zip(wf.datas_de_retreino(b.calendario, inicio, fim), vistos):
        limite = wf.limite_de_treino(b.calendario, dia)
        esperado = b.tabela[(b.tabela["data"] <= limite) & b.tabela["rotulo"].notna()]
        assert n == len(esperado)


def test_decisao_nao_muda_se_o_futuro_mudar(base_sintetica):
    """
    Bagunça (1) as features de depois de 'fim' e (2) os rótulos que nenhum retreino podia conhecer
    (depois do limite do ÚLTIMO retreino da janela). As probabilidades têm de ficar idênticas.
    """
    b = base_sintetica
    inicio, fim = b.calendario[300], b.calendario[330]
    original = wf.gerar_probabilidades(b.tabela, b.calendario, inicio, fim, "logistica")

    ultimo_retreino = wf.datas_de_retreino(b.calendario, inicio, fim)[-1]
    rotulo_desconhecido = b.tabela["data"] > wf.limite_de_treino(b.calendario, ultimo_retreino)
    depois_do_fim = b.tabela["data"] > fim
    rng = np.random.default_rng(3)
    bagunca = b.tabela.copy()
    bagunca.loc[depois_do_fim, COLUNAS_MODELO] = rng.normal(size=(depois_do_fim.sum(), len(COLUNAS_MODELO)))
    bagunca.loc[rotulo_desconhecido, "rotulo"] = rng.integers(0, 2, rotulo_desconhecido.sum())

    pd.testing.assert_frame_equal(original, wf.gerar_probabilidades(bagunca, b.calendario, inicio, fim, "logistica"))


def test_passeio_aleatorio_acerto_perto_de_50(base_sintetica):
    b = base_sintetica
    probs = wf.gerar_probabilidades(b.tabela, b.calendario, b.calendario[250], b.calendario[-11], "logistica")
    m = probs.merge(b.tabela[["data", "ticker", "rotulo"]], on=["data", "ticker"]).dropna()
    acerto = ((m["prob"] > 0.5) == (m["rotulo"] == 1)).mean()
    assert len(m) > 5000
    assert 0.40 <= acerto <= 0.60


def test_boosting_tambem_funciona(base_sintetica):
    b = base_sintetica
    probs = wf.gerar_probabilidades(b.tabela, b.calendario, b.calendario[400], b.calendario[430], "boosting")
    assert probs["prob"].between(0, 1).all() and len(probs) > 0


def test_tipo_desconhecido_da_erro():
    with pytest.raises(ValueError, match="tipo"):
        combinador.criar_modelo("rede-neural")
