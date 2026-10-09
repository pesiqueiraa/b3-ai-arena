"""Features de preço: só passado, 120 pregões de histórico e ranks dentro do universo do dia."""

import numpy as np
import pandas as pd
import pytest

from src.features import preco


def test_nenhuma_feature_olha_para_frente(sinteticos):
    precos, cal = sinteticos
    dia = cal[400]
    original = preco.calcular_features(precos, cal)
    alterados = precos.copy()
    futuro = alterados["data"] > dia
    alterados.loc[futuro, "fechamento_ajustado"] *= 3.0
    alterados.loc[futuro, "volume"] *= 10
    novo = preco.calcular_features(alterados, cal)
    ate_hoje = lambda f: f[f["data"] <= dia].sort_values(["data", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ate_hoje(original), ate_hoje(novo))


def test_precisa_de_120_pregoes_e_bova_nao_e_linha(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal)
    assert f["data"].min() == cal[120]
    assert "BOVA11" not in set(f["ticker"])
    assert f[preco.COLUNAS].notna().all().all()


def test_valores_conferem_com_a_conta_na_mao(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal).set_index(["data", "ticker"])
    p = precos.pivot(index="data", columns="ticker", values="fechamento_ajustado")
    d, t = cal[300], "AC003"
    assert f.loc[(d, t), "p_ret_5"] == pytest.approx(p.loc[d, t] / p[t].iloc[295] - 1)
    rel = (p.loc[d, t] / p[t].iloc[280] - 1) - (p.loc[d, "BOVA11"] / p["BOVA11"].iloc[280] - 1)
    assert f.loc[(d, t), "p_rel_bova_20"] == pytest.approx(rel)


def test_ranks_so_entre_os_membros_do_dia(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal)
    dia = cal[200]
    membros = pd.DataFrame({"data": [dia] * 4, "ticker": ["AC003", "AC013", "AC023", "AC033"]})
    r = preco.adicionar_ranks(f, membros)
    assert set(r["ticker"]) == {"AC003", "AC013", "AC023", "AC033"}
    assert set(r["data"]) == {dia}
    assert sorted(r["p_ret_20_rank"]) == [0.25, 0.5, 0.75, 1.0]
    assert len(preco.COLUNAS_MODELO) == 22
