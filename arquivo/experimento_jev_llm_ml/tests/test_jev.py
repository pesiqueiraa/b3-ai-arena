"""
Testes do jev.py com uma API FALSA: sem internet, sem chave e sem gastar nada.
O formato das respostas falsas copia a documentação oficial (https://docs.typesafe.ai/api.md).

Rodar:  python -m pytest
"""

import json

import pandas as pd
import pytest

import config
from src import dados, decisoes, estado
from src.modelos import jev, llm


class ApiFalsa:
    """Finge ser a API da TypeSafe. 'responder' decide o valor de cada chamada."""

    def __init__(self, responder=None, versao="jev-1.13.0"):
        self.pedidos = []
        self.versao = versao
        self.responder = responder or (lambda pedido: 0.46)

    def __call__(self, pedido, chave):
        self.pedidos.append(pedido)
        valor = self.responder(pedido)
        if isinstance(valor, Exception):
            raise valor
        resposta = valor if isinstance(valor, dict) else {"type": "noul", "noul": valor}
        return {"model": self.versao, "answers": {"sobe": resposta},
                "usage": {"input_tokens": 400, "output_tokens": 21}}


@pytest.fixture
def ambiente(tmp_path, monkeypatch, indicadores):
    monkeypatch.setattr(config, "RAIZ", tmp_path)
    monkeypatch.setattr(config, "PASTA_CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "PASTA_DECISOES", tmp_path / "decisoes")
    monkeypatch.setattr(config, "ARQUIVO_FALHAS_JEV", tmp_path / "falhas_jev.csv")
    monkeypatch.setattr(config, "ARQUIVO_PREREGISTRO", tmp_path / "PREREGISTRO.md")
    (tmp_path / "PREREGISTRO.md").write_text(
        f"Adendo 2: versão `{config.JEV_VERSAO}`, pergunta `{jev.PERGUNTA_VERSAO}`.", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "chave-falsa")
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)

    dias = sorted(indicadores.loc[indicadores["periodo"] == "teste_limpo", "data"].unique())[:3]
    poucos = indicadores[(indicadores["periodo"] == "historico") | indicadores["data"].isin(dias)]
    monkeypatch.setattr(dados, "carregar_indicadores", lambda: poucos.copy())

    api = ApiFalsa()
    monkeypatch.setattr(jev, "_post", api)
    return {"api": api, "dias": dias, "dados": poucos}


# --- Pedido no formato da documentação -----------------------------------------

def test_pedido_no_formato_da_documentacao():
    pedido = jev.montar_pedido("Ação A. Retorno no último pregão: +1,0%.")
    assert pedido == {
        "state": "Ação A. Retorno no último pregão: +1,0%.",
        "model": "jev-1.13.0",
        "questions": {"sobe": {"type": "noul", "instructions": estado.PERGUNTA}},
    }


def test_jev_e_llm_recebem_a_mesma_pergunta():
    assert jev.PERGUNTA == llm.PERGUNTA == estado.PERGUNTA


def test_apelido_que_muda_sozinho_e_recusado(ambiente, monkeypatch):
    monkeypatch.setattr(config, "JEV_VERSAO", "jev-latest")
    with pytest.raises(jev.ErroDeConfiguracao, match="apelido"):
        jev.carregar_chave()


# --- Leitura da resposta --------------------------------------------------------

@pytest.mark.parametrize("resposta, esperado", [
    ({"type": "noul", "noul": 0.44}, 0.44),
    ({"type": "noul", "noul": 0}, 0.0),
    ({"type": "noul", "noul": 1}, 1.0),
    ({"type": "noul", "noul": 1.3}, None),
    ({"type": "noul", "noul": "0.4"}, None),
    ({"type": "choice", "noul": 0.4}, None),
    ({"type": "noul"}, None),
])
def test_ler_resposta(resposta, esperado):
    assert jev.ler_resposta({"model": "jev-1.13.0", "answers": {"sobe": resposta}}) == esperado


def test_versao_errada_para_tudo(ambiente):
    ambiente["api"].versao = "jev-1.14.0"
    with pytest.raises(jev.ErroDeVersao):
        jev.decidir("Ação A.", "chave")
    assert not list((config.PASTA_CACHE / "jev").glob("*.json"))  # resposta errada não vai para o cache


# --- Regras: pré-registro, mesmo estado, cache ---------------------------------

def test_nao_roda_sem_preregistro(ambiente):
    config.ARQUIVO_PREREGISTRO.write_text(f"só a versão `{config.JEV_VERSAO}`", encoding="utf-8")
    with pytest.raises(jev.ErroDeConfiguracao, match="PREREGISTRO"):
        jev.rodar()
    assert ambiente["api"].pedidos == []


def test_jev_recebe_exatamente_o_estado_de_estado_py(ambiente):
    """Regra 4: o state enviado é o texto de src/estado.py (o mesmo que o LLM recebe)."""
    jev.rodar()
    janela = ambiente["dados"]
    janela = janela[(janela["periodo"] == "teste_limpo") & (janela["ticker"] != config.REFERENCIA)]
    assert {p["state"] for p in ambiente["api"].pedidos} == set(estado.gerar_estados(janela)["estado"])
    assert all("gabarito" not in json.dumps(p) for p in ambiente["api"].pedidos)


def test_mesma_pergunta_nao_e_paga_duas_vezes(ambiente):
    a = jev.perguntar("Ação A.", "chave")
    b = jev.perguntar("Ação A.", "chave")
    assert len(ambiente["api"].pedidos) == 1
    assert (a["do_cache"], b["do_cache"]) == (False, True)
    assert a["custo_usd"] == pytest.approx(400 * config.JEV_PRECO_USD_POR_TOKEN)


def test_erro_de_rede_tenta_de_novo(ambiente):
    erros = [ConnectionError("529"), ConnectionError("529")]
    ambiente["api"].responder = lambda p: erros.pop(0) if erros else 0.5
    assert jev.decidir("Ação A.", "chave")["prob_subir"] == 0.5
    assert len(ambiente["api"].pedidos) == 3


# --- Resposta fora do formato --------------------------------------------------

def test_resposta_ruim_repete_uma_vez_sem_cache(ambiente):
    respostas = [{"type": "noul"}, {"type": "noul", "noul": 0.47}]
    ambiente["api"].responder = lambda p: respostas.pop(0)
    r = jev.decidir("Ação A.", "chave")
    assert r["prob_subir"] == 0.47
    assert len(ambiente["api"].pedidos) == 2


def test_falha_dupla_e_registrada_e_nao_e_paga_de_novo(ambiente):
    alvo = "BBBB4"
    ambiente["api"].responder = lambda p: {"type": "noul"} if alvo in p["state"] else 0.46
    jev.rodar()
    assert set(jev.carregar_falhas()["ticker"]) == {alvo}
    assert alvo not in set(decisoes.carregar("jev")["ticker"])
    chamadas = len(ambiente["api"].pedidos)
    jev.rodar()
    assert len(ambiente["api"].pedidos) == chamadas


# --- Arquivo de decisões ------------------------------------------------------

def test_decisoes_no_formato_e_so_na_janela_limpa(ambiente):
    jev.rodar()
    d = decisoes.carregar("jev")
    assert list(d.columns) == ["data", "ticker", "prob_subir", "decisao", "latencia_ms", "custo_usd"]
    assert set(d["data"]) == set(pd.to_datetime(ambiente["dias"]))
    assert config.REFERENCIA not in set(d["ticker"])
    assert (d["decisao"] == "ficar_de_fora").all()  # 0,46 < limiar de 0,5
    chamadas = len(ambiente["api"].pedidos)
    jev.rodar()
    assert len(ambiente["api"].pedidos) == chamadas
