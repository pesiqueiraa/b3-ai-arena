"""
Testes de "sem vazamento": provam que nenhum modelo enxerga o futuro
e que o avaliador percebe quando alguém enxerga.

Rodar:  python -m pytest
"""

import numpy as np
import pandas as pd

from src import avaliacao, dados
from src.modelos import baselines
from src.modelos.ml import walk_forward
from src.modelos.ml.features import FEATURES
from tests.conftest import DATA_CORTE_FALSA


def test_ml_acerta_perto_de_50_por_cento_em_passeio_aleatorio(acoes, dias_teste):
    """Muito acima de 60% num passeio aleatório só é possível com vazamento."""
    decisoes = walk_forward.gerar_decisoes(acoes, dias_teste)
    m = decisoes.merge(acoes[["data", "ticker", "gabarito_retorno_amanha"]], on=["data", "ticker"])
    m = m.dropna(subset=["gabarito_retorno_amanha"])

    acerto = ((m["decisao"] == "comprar") == (m["gabarito_retorno_amanha"] > 0)).mean()
    print(f"acerto de direção do ML no passeio aleatório: {acerto:.1%} ({len(m)} decisões)")
    assert len(m) > 500
    assert 0.40 <= acerto <= 0.60


def test_avaliador_detecta_trapaceiro(indicadores):
    """Um agente que espia o gabarito precisa acertar 100% e bater 100% dos aleatórios."""
    gabarito, bova = avaliacao.separar_gabarito(indicadores)
    janela = gabarito[(gabarito["periodo"] == "teste_limpo")].dropna(subset=["gabarito"])

    trapaceiro = janela[["data", "ticker"]].copy()
    subiu = janela["gabarito"] > 0
    trapaceiro["prob_subir"] = subiu.astype(float).values
    trapaceiro["decisao"] = np.where(subiu, "comprar", "ficar_de_fora")

    r = avaliacao.avaliar_modelo("trapaceiro", trapaceiro, gabarito, bova,
                                 avaliacao.rng_do_modelo("trapaceiro"))
    assert r["acerto_de_direcao"] == 1.0
    assert r["acerto_das_compras"] == 1.0
    assert r["bateu_aleatorios"] == 1.0


def test_agente_aleatorio_nao_parece_habilidoso(indicadores):
    """Contraprova: um agente que joga moeda não deve 'bater' quase todos os aleatórios."""
    gabarito, bova = avaliacao.separar_gabarito(indicadores)
    janela = gabarito[gabarito["periodo"] == "teste_limpo"]

    rng = np.random.default_rng(123)
    moeda = janela[["data", "ticker"]].copy()
    moeda["prob_subir"] = rng.random(len(moeda))
    moeda["decisao"] = np.where(moeda["prob_subir"] > 0.5, "comprar", "ficar_de_fora")

    r = avaliacao.avaliar_modelo("moeda", moeda, gabarito, bova, avaliacao.rng_do_modelo("moeda"))
    assert 0.40 <= r["acerto_de_direcao"] <= 0.60
    assert r["bateu_aleatorios"] < 0.95


def test_modelo_de_ml_nunca_recebe_o_gabarito(acoes, dias_teste):
    """Espia tudo o que as funções do modelo recebem: só features, nunca o gabarito."""
    recebido = []

    def treinar_espiao(x, subiu):
        recebido.append(("treino", list(x.columns), len(x)))
        return walk_forward.ml_modelo.treinar(x, subiu)

    def prob_espiao(modelo, x):
        recebido.append(("previsao", list(x.columns), len(x)))
        return walk_forward.ml_modelo.prob_subir(modelo, x)

    dia = dias_teste[10]
    walk_forward.decidir_um_dia(acoes, dia, treinar=treinar_espiao, prob_subir=prob_espiao)

    assert [r[0] for r in recebido] == ["treino", "previsao"]
    for _, colunas, _ in recebido:
        assert colunas == FEATURES
        assert not any("gabarito" in c for c in colunas)

    # O treino tem exatamente as linhas ANTERIORES ao dia (com gabarito já conhecido).
    antes = acoes[acoes["data"] < dia].dropna(subset=FEATURES + ["gabarito_retorno_amanha"])
    assert recebido[0][2] == len(antes)
    # A previsão tem exatamente as linhas do próprio dia.
    assert recebido[1][2] == (acoes["data"] == dia).sum()


def test_decisao_de_hoje_nao_muda_se_o_futuro_mudar(acoes, dias_teste):
    """
    Bagunça TUDO o que vem depois do dia D (inclusive o gabarito de D).
    Se a decisão de D mudar, o modelo estava olhando o futuro.
    """
    dia = dias_teste[20]
    original = walk_forward.decidir_um_dia(acoes, dia)

    bagunca = acoes.copy()
    rng = np.random.default_rng(99)
    futuro = bagunca["data"] > dia
    bagunca.loc[futuro, FEATURES] = rng.normal(size=(futuro.sum(), len(FEATURES)))
    de_hoje_em_diante = bagunca["data"] >= dia
    bagunca.loc[de_hoje_em_diante, "gabarito_retorno_amanha"] = rng.normal(size=de_hoje_em_diante.sum())

    alterado = walk_forward.decidir_um_dia(bagunca, dia)
    pd.testing.assert_frame_equal(original, alterado)


def test_indicadores_nao_olham_para_frente(precos):
    """Mudar os preços depois de D não pode mudar nenhum indicador até D (só o gabarito de D)."""
    dia = pd.Timestamp("2025-03-03")
    original = dados.calcular_indicadores(precos, data_corte=DATA_CORTE_FALSA)

    alterados = precos.copy()
    futuro = alterados["data"] > dia
    alterados.loc[futuro, ["fechamento", "fechamento_ajustado"]] *= 3.0
    alterados.loc[futuro, "volume"] *= 10
    novo = dados.calcular_indicadores(alterados, data_corte=DATA_CORTE_FALSA)

    ate_hoje = original["data"] <= dia
    pd.testing.assert_frame_equal(
        original.loc[ate_hoje, FEATURES].reset_index(drop=True),
        novo.loc[novo["data"] <= dia, FEATURES].reset_index(drop=True),
    )
    # Contraprova: o gabarito de D tem que mudar (ele É o futuro).
    g_orig = original.loc[original["data"] == dia, "gabarito_retorno_amanha"].to_numpy()
    g_novo = novo.loc[novo["data"] == dia, "gabarito_retorno_amanha"].to_numpy()
    assert not np.allclose(g_orig, g_novo)


def test_baseline_funciona_so_com_data_e_ticker(acoes):
    """O 'sempre comprar' não precisa (nem recebe) nenhum indicador ou gabarito."""
    saida = baselines.sempre_comprar(acoes[["data", "ticker"]])
    assert set(saida["decisao"]) == {"comprar"}
    assert list(saida.columns) == ["data", "ticker", "prob_subir", "decisao"]
