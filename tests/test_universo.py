"""Universo mensal: as mais líquidas pelo volume PASSADO, uma ação por empresa."""

import pandas as pd

from src import universo


def b3_falso(volumes: dict[str, float], n_dias: int = 70, inicio: str = "2020-01-01") -> pd.DataFrame:
    dias = pd.bdate_range(inicio, periods=n_dias)
    return pd.DataFrame([
        {"data": d, "ticker": t, "volume_financeiro": v} for d in dias for t, v in volumes.items()
    ])


def test_ordena_pelo_volume_e_limita_o_tamanho():
    b3 = b3_falso({"AAAA3": 5e6, "BBBB3": 4e6, "CCCC3": 3e6, "DDDD3": 2e6, "EEEE3": 1e6})
    u = universo.montar_universo(b3, tamanho=3, janela=20)
    primeiro = u[u["mes"] == u["mes"].min()]
    assert list(primeiro["ticker"]) == ["AAAA3", "BBBB3", "CCCC3"]
    assert list(primeiro["posicao"]) == [1, 2, 3]


def test_uma_acao_por_empresa_fica_a_mais_liquida():
    b3 = b3_falso({"PETR3": 1e6, "PETR4": 5e6, "VALE3": 3e6})
    u = universo.montar_universo(b3, tamanho=10, janela=20)
    assert set(u["ticker"]) == {"PETR4", "VALE3"}


def test_volume_futuro_nao_muda_o_mes():
    b3 = b3_falso({"AAAA3": 5e6, "BBBB3": 4e6, "CCCC3": 3e6})
    antes = universo.montar_universo(b3, tamanho=2, janela=20)
    mes = sorted(antes["mes"].unique())[0]
    inicio_do_mes = antes.loc[antes["mes"] == mes, "data_inicio"].iloc[0]
    alterado = b3.copy()
    alterado.loc[(alterado["data"] >= inicio_do_mes) & (alterado["ticker"] == "CCCC3"), "volume_financeiro"] = 1e12
    depois = universo.montar_universo(alterado, tamanho=2, janela=20)
    pd.testing.assert_frame_equal(antes[antes["mes"] == mes].reset_index(drop=True),
                                  depois[depois["mes"] == mes].reset_index(drop=True))


def test_dia_sem_negocio_conta_como_zero_na_media():
    dias = pd.bdate_range("2020-01-01", periods=70)
    linhas = [{"data": d, "ticker": "CONS3", "volume_financeiro": 1e6} for d in dias]
    linhas += [{"data": d, "ticker": "RARA3", "volume_financeiro": 3e6} for d in dias[::4]]  # 1 a cada 4 dias
    u = universo.montar_universo(pd.DataFrame(linhas), tamanho=1, janela=20)
    assert set(u["ticker"]) == {"CONS3"}  # média do RARA3 ~ 0,75 mi < 1 mi


def test_meses_sem_historico_suficiente_ficam_de_fora():
    b3 = b3_falso({"AAAA3": 1e6}, n_dias=70, inicio="2020-01-01")
    u = universo.montar_universo(b3, tamanho=5, janela=63)
    assert u["data_inicio"].min() >= pd.bdate_range("2020-01-01", periods=70)[63]


def test_membros_por_dia_usa_a_lista_do_proprio_mes():
    u = pd.DataFrame({"mes": ["2020-01", "2020-02"], "data_inicio": pd.to_datetime(["2020-01-02", "2020-02-03"]),
                      "ticker": ["AAAA3", "BBBB3"], "posicao": [1, 1], "volume_medio": [1.0, 1.0]})
    cal = pd.DatetimeIndex(pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"]))
    m = universo.membros_por_dia(u, cal)
    assert list(zip(m["data"].dt.strftime("%m-%d"), m["ticker"])) == [
        ("01-30", "AAAA3"), ("01-31", "AAAA3"), ("02-03", "BBBB3")]


def test_cobertura_conta_preco_no_proprio_mes():
    u = pd.DataFrame({"mes": ["2020-01", "2020-01", "2020-02"], "ticker": ["AAAA3", "MORT3", "MORT3"],
                      "data_inicio": pd.to_datetime(["2020-01-02"] * 2 + ["2020-02-03"]),
                      "posicao": [1, 2, 1], "volume_medio": 1.0})
    precos = pd.DataFrame({"data": pd.to_datetime(["2020-01-15", "2020-01-15"]), "ticker": ["AAAA3", "MORT3"],
                           "fechamento_ajustado": 1.0, "volume": 1.0})
    c = universo.calcular_cobertura(u, precos).set_index("mes")
    assert c.loc["2020-01", "cobertura"] == 1.0
    assert c.loc["2020-02", "cobertura"] == 0.0     # MORT3 sem preço em fevereiro
