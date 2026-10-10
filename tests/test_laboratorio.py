"""O laboratório: nunca lê a reserva, registra cada tentativa e escolhe pela regra da especificação."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.base import montar_base
from src.laboratorio import executar
from tests.conftest import gerar_precos, universo_completo


@pytest.fixture
def lab_rapido(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LAB_INICIO", "2016-01-01")
    monkeypatch.setattr(config, "RESERVA_INICIO", "2017-01-01")
    monkeypatch.setattr(config, "N_ALEATORIOS", 50)
    monkeypatch.setattr(config, "GRADE_K", (3,))
    monkeypatch.setattr(config, "GRADE_P_MIN", (0.50, 0.55))
    monkeypatch.setattr(config, "ARQUIVO_TENTATIVAS", tmp_path / "tentativas.csv")


def _base(precos, cal):
    return montar_base(precos, universo_completo(precos, cal), cal)


def test_recorte_tira_a_reserva_e_os_rotulos_que_dependem_dela(lab_rapido, base_sintetica):
    b = executar.recortar_laboratorio(base_sintetica)
    corte = pd.Timestamp("2017-01-01")
    assert b.tabela["data"].max() < corte and b.calendario[-1] < corte and b.retornos.index[-1] < corte
    assert b.tabela.loc[b.tabela["data"] > b.calendario[-11], "rotulo"].isna().all()


def test_mudar_a_reserva_nao_muda_nada_no_laboratorio(lab_rapido):
    precos, cal = gerar_precos(n_acoes=12, fim="2017-06-30")
    alterados = precos.copy()
    na_reserva = alterados["data"] >= pd.Timestamp("2017-01-01")
    alterados.loc[na_reserva, "fechamento_ajustado"] *= np.random.default_rng(5).uniform(0.5, 2.0, na_reserva.sum())

    linhas_a, escolha_a, _ = executar.executar_laboratorio(_base(precos, cal))
    linhas_b, escolha_b, _ = executar.executar_laboratorio(_base(alterados, cal))
    assert escolha_a == escolha_b
    pd.testing.assert_frame_equal(pd.DataFrame(linhas_a), pd.DataFrame(linhas_b))


def test_cada_configuracao_vira_uma_tentativa_registrada(lab_rapido):
    precos, cal = gerar_precos(n_acoes=12, fim="2017-03-31")
    linhas, _, _ = executar.executar_laboratorio(_base(precos, cal))
    for linha in linhas:
        executar.registrar_tentativa(linha)
    registradas = pd.read_csv(config.ARQUIVO_TENTATIVAS)
    assert len(registradas) == len(linhas) == 2 * 1 * 2       # 2 modelos x 1 k x 2 p_min
    assert {"registrado_em", "tipo", "k", "p_min", "brier", "excesso_medio_anual"} <= set(registradas.columns)


def test_escolhe_primeiro_pelo_brier_depois_pelo_excesso():
    linhas = [
        {"tipo": "logistica", "k": 3, "p_min": 0.5, "excesso_medio_anual": 0.01},
        {"tipo": "logistica", "k": 5, "p_min": 0.55, "excesso_medio_anual": 0.03},
        {"tipo": "boosting", "k": 3, "p_min": 0.5, "excesso_medio_anual": 0.50},   # melhor excesso, pior Brier
    ]
    escolha = executar.escolher({"logistica": 0.249, "boosting": 0.251}, linhas)
    assert escolha == {"tipo": "logistica", "k": 5, "p_min": 0.55}


def test_excesso_vazio_nunca_e_escolhido():
    linhas = [{"tipo": "logistica", "k": 3, "p_min": 0.6, "excesso_medio_anual": float("nan")},
              {"tipo": "logistica", "k": 5, "p_min": 0.5, "excesso_medio_anual": -0.02}]
    assert executar.escolher({"logistica": 0.25}, linhas)["k"] == 5


def test_relatorio_traz_cobertura_por_ano(monkeypatch, tmp_path):
    arquivo = tmp_path / "cobertura.csv"
    pd.DataFrame({"mes": ["2016-01", "2016-02", "2017-01"], "no_universo": [100, 100, 100],
                  "com_preco": [50, 60, 90], "cobertura": [0.5, 0.6, 0.9]}).to_csv(arquivo, index=False)
    monkeypatch.setattr(config, "ARQUIVO_COBERTURA", arquivo)
    texto = executar._relatorio([], {"tipo": "logistica", "k": 5, "p_min": 0.55}, {})
    assert "Cobertura por ano" in texto
    assert "| 2016 | 55.0% |" in texto and "| 2017 | 90.0% |" in texto


def test_relatorio_sem_arquivo_de_cobertura_nao_quebra(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQUIVO_COBERTURA", tmp_path / "nao_existe.csv")
    texto = executar._relatorio([], {"tipo": "logistica", "k": 5, "p_min": 0.55}, {})
    assert "Cobertura por ano" not in texto
