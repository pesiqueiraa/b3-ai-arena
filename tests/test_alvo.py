"""Alvo: 'a ação superou o BOVA11 nos próximos 10 pregões?' e retornos para a simulação."""

import numpy as np
import pandas as pd
import pytest

from src import alvo


def _precos(series: dict[str, list[float]], inicio="2024-01-01"):
    n = len(next(iter(series.values())))
    cal = pd.DatetimeIndex(pd.bdate_range(inicio, periods=n))
    linhas = [{"data": d, "ticker": t, "fechamento_ajustado": p, "volume": 1.0}
              for t, ps in series.items() for d, p in zip(cal, ps) if not np.isnan(p)]
    return pd.DataFrame(linhas), cal


def test_excesso_sobre_o_bova11_em_10_pregoes():
    precos, cal = _precos({"AAAA3": [100 * 1.01 ** i for i in range(15)], "BOVA11": [50.0] * 15})
    a = alvo.calcular_alvo(precos, cal, horizonte=10)
    assert len(a) == 5                                     # só os 5 dias com D+10 conhecido
    assert a["excesso"].iloc[0] == pytest.approx(1.01 ** 10 - 1)
    assert (a["rotulo"] == 1).all()
    assert list(a.columns) == ["data", "ticker", "excesso", "rotulo"]


def test_sem_preco_em_d_mais_10_nao_ha_rotulo():
    precos_a = [100.0] * 15
    precos_a[10] = np.nan                                  # suspensa no dia D+10 de D=0
    precos, cal = _precos({"AAAA3": precos_a, "BOVA11": [50.0] * 15})
    a = alvo.calcular_alvo(precos, cal, horizonte=10)
    assert cal[0] not in set(a["data"])


def test_rotulo_zero_quando_fica_atras():
    precos, cal = _precos({"AAAA3": [100.0] * 12, "BOVA11": [50 * 1.01 ** i for i in range(12)]})
    assert (alvo.calcular_alvo(precos, cal, horizonte=10)["rotulo"] == 0).all()


def test_sem_bova11_da_erro_claro():
    precos, cal = _precos({"AAAA3": [100.0] * 12})
    with pytest.raises(ValueError, match="BOVA11"):
        alvo.calcular_alvo(precos, cal)


def test_retornos_diarios_apos_a_acao_sumir_sao_zero():
    precos, cal = _precos({"AAAA3": [10.0, 11.0, np.nan, np.nan], "BOVA11": [1.0, 1.0, 1.0, 1.0]})
    r = alvo.retornos_diarios(precos, cal)
    assert list(r.index) == list(cal)
    assert r["AAAA3"].tolist() == pytest.approx([0.0, 0.1, 0.0, 0.0])
