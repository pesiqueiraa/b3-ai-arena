"""Sinais: top-k compras acima do mínimo, top-k vendas abaixo de 1 − mínimo, sem conflito."""

import pandas as pd

from src.sinais import gerar_sinais


def _probs(valores: dict[str, float], data="2024-01-02"):
    return pd.DataFrame({"data": pd.Timestamp(data), "ticker": list(valores), "prob": list(valores.values())})


def test_compra_e_venda_respeitam_k_e_minimo():
    p = _probs({"A": 0.70, "B": 0.60, "C": 0.54, "D": 0.50, "E": 0.44, "F": 0.30})
    s = gerar_sinais(p, k=2, p_min=0.55)
    assert list(s.loc[s["sinal"] == "compra", "ticker"]) == ["A", "B"]
    assert list(s.loc[s["sinal"] == "venda", "ticker"]) == ["F", "E"]


def test_sem_conviccao_nao_ha_sinal():
    s = gerar_sinais(_probs({"A": 0.52, "B": 0.50, "C": 0.48}), k=3, p_min=0.55)
    assert s.empty and list(s.columns) == ["data", "ticker", "prob", "sinal"]


def test_mesma_acao_nunca_e_compra_e_venda_no_mesmo_dia():
    s = gerar_sinais(_probs({"A": 0.50, "B": 0.50}), k=5, p_min=0.50)   # empate exato no limite
    assert not s.duplicated(["data", "ticker"]).any()


def test_cada_dia_tem_seus_proprios_sinais():
    p = pd.concat([_probs({"A": 0.9, "B": 0.1}, "2024-01-02"), _probs({"A": 0.1, "B": 0.9}, "2024-01-03")])
    s = gerar_sinais(p, k=1, p_min=0.55).set_index(["data", "sinal"])["ticker"]
    assert s[(pd.Timestamp("2024-01-02"), "compra")] == "A"
    assert s[(pd.Timestamp("2024-01-03"), "compra")] == "B"
