"""
Testes do llm.py com uma API FALSA: sem internet, sem chave e sem gastar nada.

Rodar:  python -m pytest
"""

import json

import pandas as pd
import pytest

import config
from src import dados, decisoes, estado
from src.modelos import llm

MODELO = "provedor/modelo-de-teste"


class ApiFalsa:
    """Finge ser o OpenRouter. 'responder' decide o texto de cada chamada."""

    def __init__(self, responder=None):
        self.pedidos = []
        self.responder = responder or (lambda pedido: '{"prob_subir": 0.62}')

    def __call__(self, pedido, chave):
        self.pedidos.append(pedido)
        texto = self.responder(pedido)
        if isinstance(texto, Exception):
            raise texto
        return {"model": pedido["model"], "choices": [{"message": {"content": texto}}],
                "usage": {"cost": 0.0002}}


@pytest.fixture
def ambiente(tmp_path, monkeypatch, indicadores):
    """Pastas temporárias, .env falso, pré-registro com o modelo e poucos dias de dados."""
    monkeypatch.setattr(config, "RAIZ", tmp_path)
    monkeypatch.setattr(config, "PASTA_CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "PASTA_DECISOES", tmp_path / "decisoes")
    monkeypatch.setattr(config, "ARQUIVO_FALHAS_LLM", tmp_path / "falhas_llm.csv")
    monkeypatch.setattr(config, "ARQUIVO_PREREGISTRO", tmp_path / "PREREGISTRO.md")
    (tmp_path / "PREREGISTRO.md").write_text(
        f"Adendo: modelo `{MODELO}`, prompt `{llm.PROMPT_VERSAO}`.", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "chave-falsa")
    monkeypatch.setenv("LLM_MODELO", MODELO)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    # Só os 3 primeiros dias da janela limpa (mais o histórico), para o teste ser rápido.
    dias = sorted(indicadores.loc[indicadores["periodo"] == "teste_limpo", "data"].unique())[:3]
    poucos = indicadores[(indicadores["periodo"] == "historico") | indicadores["data"].isin(dias)]
    monkeypatch.setattr(dados, "carregar_indicadores", lambda: poucos.copy())

    api = ApiFalsa()
    monkeypatch.setattr(llm, "_post", api)
    return {"api": api, "dias": dias, "dados": poucos}


# --- Formato da resposta -------------------------------------------------------

@pytest.mark.parametrize("texto, esperado", [
    ('{"prob_subir": 0.62}', 0.62),
    ('  {"prob_subir": 1}  ', 1.0),
    ('{"prob_subir": 0}', 0.0),
    ('```json\n{"prob_subir": 0.4}\n```', 0.4),
    ('{"prob_subir": 1.2}', None),          # fora de 0 a 1
    ('{"prob_subir": -0.1}', None),
    ('{"prob_subir": "0.6"}', None),        # texto, não número
    ('{"prob_subir": true}', None),
    ('{"prob_subir": NaN}', None),
    ('{"prob_subir": 0.6, "motivo": "x"}', None),   # chave extra
    ('Acho que {"prob_subir": 0.6}', None),          # texto fora do JSON
    ('', None), (None, None),
])
def test_validar_resposta(texto, esperado):
    assert llm.validar_resposta(texto) == esperado


# --- Regras: temperatura 0, sem busca na web, pré-registro ---------------------

def test_pedido_sem_busca_na_web_e_temperatura_zero():
    pedido = llm.montar_pedido(llm.montar_mensagens("Ação A."), MODELO)
    assert set(pedido) == {"model", "messages", "temperature", "seed", "provider"}
    assert pedido["temperature"] == 0
    assert pedido["provider"]["allow_fallbacks"] is False   # nunca troca de provedor sozinho
    assert not any(k in pedido for k in ("tools", "plugins", "web_search_options"))


def test_modelo_online_e_recusado(ambiente, monkeypatch):
    monkeypatch.setenv("LLM_MODELO", MODELO + ":online")
    with pytest.raises(llm.ErroDeConfiguracao, match="busca na web"):
        llm.carregar_configuracao()


def test_nao_roda_sem_preregistro(ambiente):
    config.ARQUIVO_PREREGISTRO.write_text("sem adendo ainda", encoding="utf-8")
    with pytest.raises(llm.ErroDeConfiguracao, match="PREREGISTRO"):
        llm.rodar()
    assert ambiente["api"].pedidos == []  # nenhuma chamada paga


def test_llm_recebe_exatamente_o_estado_de_estado_py(ambiente):
    """Regra 4: o texto enviado é o de src/estado.py (o mesmo que o Jev vai receber)."""
    llm.rodar()
    janela = ambiente["dados"]
    janela = janela[(janela["periodo"] == "teste_limpo") & (janela["ticker"] != config.REFERENCIA)]
    esperados = {f"{t}\n\n{llm.ROTULO_PERGUNTA}: {llm.PERGUNTA}" for t in estado.gerar_estados(janela)["estado"]}
    enviados = {p["messages"][1]["content"] for p in ambiente["api"].pedidos}
    assert enviados == esperados
    assert all("gabarito" not in json.dumps(p) for p in ambiente["api"].pedidos)


# --- Cache: nunca pagar duas vezes ---------------------------------------------

def test_mesma_pergunta_nao_e_paga_duas_vezes(ambiente):
    msgs = llm.montar_mensagens("Ação A. Retorno no último pregão: +1,0%.")
    a = llm.perguntar(msgs, MODELO, "chave")
    b = llm.perguntar(msgs, MODELO, "chave")
    assert len(ambiente["api"].pedidos) == 1
    assert (a["do_cache"], b["do_cache"]) == (False, True)
    assert a["custo_usd"] == b["custo_usd"] == 0.0002


def test_erro_de_rede_tenta_de_novo(ambiente):
    erros = [ConnectionError("caiu"), ConnectionError("caiu de novo")]
    ambiente["api"].responder = lambda p: erros.pop(0) if erros else '{"prob_subir": 0.3}'
    r = llm.decidir("Ação A.", MODELO, "chave")
    assert r["prob_subir"] == 0.3
    assert len(ambiente["api"].pedidos) == 3


# --- Resposta inválida: uma nova tentativa, depois falha registrada -------------

def test_resposta_invalida_tenta_de_novo_uma_vez(ambiente):
    respostas = ["Não sei prever ações.", '{"prob_subir": 0.55}']
    ambiente["api"].responder = lambda p: respostas.pop(0)
    r = llm.decidir("Ação A.", MODELO, "chave")
    assert r["prob_subir"] == 0.55
    assert r["custo_usd"] == pytest.approx(0.0004)  # as duas chamadas
    segunda = ambiente["api"].pedidos[1]["messages"]
    assert segunda[-1]["content"] == llm.CORRECAO
    assert segunda[-2] == {"role": "assistant", "content": "Não sei prever ações."}


def test_falha_dupla_e_registrada_e_nao_e_paga_de_novo(ambiente):
    alvo = "BBBB4"

    def responder(pedido):
        if alvo in pedido["messages"][1]["content"]:
            return "resposta sem JSON"
        return '{"prob_subir": 0.7}'

    ambiente["api"].responder = responder
    llm.rodar()

    falhas = llm.carregar_falhas()
    assert set(falhas["ticker"]) == {alvo}
    assert len(falhas) == len(ambiente["dias"])
    gravadas = decisoes.carregar("llm")
    assert alvo not in set(gravadas["ticker"])

    chamadas = len(ambiente["api"].pedidos)
    llm.rodar()  # de novo: nada pendente, nada pago
    assert len(ambiente["api"].pedidos) == chamadas


# --- Arquivo de decisões ------------------------------------------------------

def test_decisoes_no_formato_e_so_na_janela_limpa(ambiente):
    llm.rodar()
    d = decisoes.carregar("llm")
    assert list(d.columns) == ["data", "ticker", "prob_subir", "decisao", "latencia_ms", "custo_usd"]
    assert set(d["data"]) == set(pd.to_datetime(ambiente["dias"]))
    assert config.REFERENCIA not in set(d["ticker"])
    assert (d["decisao"] == "comprar").all()  # 0,62 > limiar de 0,5
    assert (d["custo_usd"] == 0.0002).all()

    chamadas = len(ambiente["api"].pedidos)
    llm.rodar()  # só acréscimo: nada é refeito
    assert len(ambiente["api"].pedidos) == chamadas
