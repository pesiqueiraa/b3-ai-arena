"""Prova final (uma vez só) e sinais do próximo pregão."""

import json

import pandas as pd
import pytest

import config
from src import hoje
from src.laboratorio import prova_final, walk_forward as wf
from src.modelos import combinador


@pytest.fixture
def arquivos(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQUIVO_PROVA_FINAL", tmp_path / "prova_final.json")
    monkeypatch.setattr(config, "ARQUIVO_RELATORIO_PROVA", tmp_path / "prova_final.md")
    monkeypatch.setattr(config, "ARQUIVO_ESCOLHA", tmp_path / "escolha.json")
    monkeypatch.setattr(config, "RESERVA_INICIO", "2017-07-01")
    monkeypatch.setattr(config, "N_ALEATORIOS", 50)
    return tmp_path


def _gravar_escolha():
    config.ARQUIVO_ESCOLHA.write_text(json.dumps({"tipo": "logistica", "k": 3, "p_min": 0.5}), encoding="utf-8")


def test_sem_escolha_do_laboratorio_recusa(arquivos, base_sintetica):
    with pytest.raises(prova_final.ErroProvaFinal, match="laboratorio"):
        prova_final.rodar(base_sintetica)


def test_roda_uma_vez_e_recusa_a_segunda(arquivos, base_sintetica):
    _gravar_escolha()
    resultado = prova_final.rodar(base_sintetica)
    assert config.ARQUIVO_PROVA_FINAL.exists() and config.ARQUIVO_RELATORIO_PROVA.exists()
    assert set(resultado["criterios"]) == {"excesso_total_positivo", "bateu_aleatorios", "acerto_compras"}
    assert resultado["aprovado"] == all(resultado["criterios"].values())
    assert resultado["periodo"][0] == "2017-07-03"                 # primeiro pregão da reserva
    with pytest.raises(prova_final.ErroProvaFinal, match="já foi rodada"):
        prova_final.rodar(base_sintetica)


def test_criterios_exigem_os_tres():
    ok = {"excesso_total": 0.01, "bateu_aleatorios": 0.95, "acerto_compras": 0.55}
    assert all(prova_final.criterios(ok).values())
    assert not prova_final.criterios({**ok, "bateu_aleatorios": 0.89})["bateu_aleatorios"]
    assert not prova_final.criterios({**ok, "acerto_compras": float("nan")})["acerto_compras"]


def test_sinais_do_dia_so_treinam_com_o_que_ja_era_conhecido(base_sintetica):
    vistos = []

    def treinar_espiao(x, y, tipo):
        vistos.append(len(x))
        return combinador.treinar(x, y, tipo)

    sinais, dia = hoje.calcular(base_sintetica, {"tipo": "logistica", "k": 3, "p_min": 0.5}, treinar=treinar_espiao)
    b = base_sintetica
    limite = wf.limite_de_treino(b.calendario, dia)
    assert dia == b.tabela["data"].max()
    assert vistos == [len(b.tabela[(b.tabela["data"] <= limite) & b.tabela["rotulo"].notna()])]
    assert set(sinais["data"]) <= {dia}


def test_sem_escolha_usa_o_padrao(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQUIVO_ESCOLHA", tmp_path / "nao_existe.json")
    assert hoje.escolha_atual() == {"tipo": "logistica", "k": config.K_PADRAO, "p_min": config.P_MIN_PADRAO}


def test_dia_com_metade_das_acoes_usa_o_pregao_anterior(base_sintetica, capsys):
    import dataclasses
    tabela = base_sintetica.tabela
    ultimo = tabela["data"].max()
    do_ultimo = tabela.index[tabela["data"] == ultimo]
    cortada = tabela.drop(do_ultimo[: len(do_ultimo) // 2])             # metade das ações sumiu no último dia
    base = dataclasses.replace(base_sintetica, tabela=cortada)

    dia, faltando = hoje.escolher_dia(cortada)
    assert dia < ultimo and len(faltando) > 0
    sinais, dia_usado = hoje.calcular(base, {"tipo": "logistica", "k": 3, "p_min": 0.5})
    assert dia_usado == dia and set(sinais["data"]) <= {dia}
    assert "incompletos" in capsys.readouterr().out


def test_dia_completo_continua_sendo_o_ultimo(base_sintetica):
    dia, faltando = hoje.escolher_dia(base_sintetica.tabela)
    assert dia == base_sintetica.tabela["data"].max() and faltando == []
