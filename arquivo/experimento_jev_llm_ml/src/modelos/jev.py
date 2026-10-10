"""
jev.py
Decisões do Jev (TypeSafe AI), o "System One model", via API oficial.

Como funciona, para cada ação em cada dia da janela limpa:
    1. pega o estado em texto de src/estado.py (o MESMO que o LLM recebe);
    2. faz uma pergunta do tipo Noul (sim/não): "A ação vai fechar o próximo pregão acima
       do fechamento de hoje?". O Jev devolve a probabilidade de "sim" (0 a 1), que vira prob_subir;
    3. confere se quem respondeu foi mesmo a versão fixa (config.JEV_VERSAO = "jev-1.13.0");
    4. grava a decisão em resultados/decisoes/decisoes_jev.csv, com latência e custo.

Regras:
    - Versão FIXA jev-1.13.0. Se a API responder com outra versão, a etapa para.
    - Toda resposta fica guardada em resultados/cache/jev/. A mesma pergunta nunca é paga duas vezes.
    - Só roda se a versão da pergunta estiver registrada no PREREGISTRO.md.

Configuração no arquivo .env: TYPESAFE_API_KEY.
Documentação usada (consultada em 09/10/2026): https://docs.typesafe.ai/api.md,
.../primitives/noul.md, .../models.md, .../concepts/state.md, .../model-jaggedness/jev-1.13.md
"""

import os
import time
from datetime import datetime

import pandas as pd

import config
from src import dados, decisoes, estado
from src.modelos import api_comum
from src.modelos.api_comum import ErroDeAPI, ErroDeConfiguracao

NOME = "jev"

# Versão da pergunta. Mudou a pergunta ou o formato do pedido? Mude a versão e registre no PREREGISTRO.md.
PERGUNTA_VERSAO = "jev-pergunta-v1"

# A MESMA pergunta do LLM (regra 4). A chave "sobe" é só um nome para achar a resposta:
# segundo a documentação, o modelo não vê os nomes das perguntas.
PERGUNTA = estado.PERGUNTA
ID_PERGUNTA = "sobe"


class ErroDeVersao(Exception):
    """A API respondeu com uma versão diferente da fixada: a comparação perderia o sentido."""


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def carregar_chave() -> str:
    from dotenv import load_dotenv

    load_dotenv(config.RAIZ / ".env")
    chave = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not chave:
        raise ErroDeConfiguracao(
            "falta TYPESAFE_API_KEY no arquivo .env (copie o .env.example para .env e preencha)."
        )
    if config.JEV_VERSAO.endswith(("latest", "preview")):
        raise ErroDeConfiguracao(
            f"JEV_VERSAO = '{config.JEV_VERSAO}' é um apelido que muda sozinho. Use uma versão fixa."
        )
    return chave


def verificar_preregistro() -> None:
    """A versão do Jev e a versão da pergunta precisam estar no PREREGISTRO.md antes da 1ª chamada."""
    api_comum.exigir_no_preregistro([config.JEV_VERSAO, PERGUNTA_VERSAO], "python main.py jev --teste")


# ---------------------------------------------------------------------------
# A pergunta ao Jev
# ---------------------------------------------------------------------------

def montar_pedido(texto_estado: str) -> dict:
    """Corpo do POST /v1/systemone: state + model + questions (uma pergunta Noul, sem criteria)."""
    return {
        "state": texto_estado,
        "model": config.JEV_VERSAO,
        "questions": {
            ID_PERGUNTA: {"type": "noul", "instructions": PERGUNTA},
        },
    }


def ler_resposta(bruto: dict) -> float | None:
    """
    Tira a probabilidade de "sim" da resposta: answers.sobe.noul (0 a 1).
    Versão errada é erro grave (ErroDeVersao); resposta fora do formato vale None.
    """
    versao = bruto.get("model")
    if versao != config.JEV_VERSAO:
        raise ErroDeVersao(f"a API respondeu com '{versao}', mas o experimento usa '{config.JEV_VERSAO}'.")
    try:
        resposta = bruto["answers"][ID_PERGUNTA]
        valor = resposta["noul"]
    except (KeyError, TypeError):
        return None
    if resposta.get("type") != "noul" or isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    if not 0 <= valor <= 1:
        return None
    return float(valor)


def _post(pedido: dict, chave: str) -> dict:
    """A chamada HTTP de verdade (trocada por uma falsa nos testes)."""
    import requests

    resposta = requests.post(
        config.JEV_URL,
        headers={"Authorization": f"Bearer {chave}", "Content-Type": "application/json"},
        json=pedido,
        timeout=config.JEV_TIMEOUT_S,
    )
    # 429 (limite de uso), 529 (sobrecarga) e 5xx: vale tentar de novo.
    if resposta.status_code in (429, 529) or resposta.status_code >= 500:
        raise ConnectionError(f"HTTP {resposta.status_code}: {resposta.text[:200]}")
    if resposta.status_code != 200:  # 401 (chave), 422 (pedido malformado)...
        raise ErroDeAPI(f"HTTP {resposta.status_code}: {resposta.text[:500]}")
    return resposta.json()


def perguntar(texto_estado: str, chave: str, usar_cache: bool = True, sufixo_cache: str = "") -> dict:
    """
    Faz UMA pergunta ao Jev, usando o cache sempre que possível.
    Devolve: prob (ou None), latência e custo da chamada ORIGINAL, e se veio do cache.
    """
    pedido = montar_pedido(texto_estado)
    arquivo = api_comum.arquivo_do_cache(NOME, pedido, sufixo_cache)
    if usar_cache:
        guardado = api_comum.ler_cache(arquivo)
        if guardado is not None:
            return {**guardado, "do_cache": True}

    for tentativa in range(1, config.JEV_TENTATIVAS_REDE + 1):
        inicio = time.perf_counter()
        try:
            bruto = _post(pedido, chave)
            break
        except (ConnectionError, TimeoutError, OSError) as erro:
            if tentativa == config.JEV_TENTATIVAS_REDE:
                raise ErroDeAPI(f"a API não respondeu após {tentativa} tentativas: {erro}") from erro
            espera = 2 ** tentativa
            print(f"  (erro de rede: {erro}. Tentando de novo em {espera}s...)")
            time.sleep(espera)
    latencia_ms = (time.perf_counter() - inicio) * 1000

    tokens = (bruto.get("usage") or {}).get("input_tokens")
    resultado = {
        "prob": ler_resposta(bruto),
        "latencia_ms": round(latencia_ms, 1),
        "custo_usd": tokens * config.JEV_PRECO_USD_POR_TOKEN if tokens is not None else None,
        "versao_respondeu": bruto.get("model"),
        "tokens_entrada": tokens,
    }
    api_comum.gravar_cache(arquivo, pedido, bruto, resultado)
    return {**resultado, "do_cache": False}


def decidir(texto_estado: str, chave: str) -> dict:
    """
    Uma decisão completa. Se a resposta vier fora do formato, repete UMA vez a mesma pergunta
    (sem usar o cache, já que o pedido é idêntico). Latência e custo somam as duas tentativas.
    """
    tentativas = [perguntar(texto_estado, chave)]
    if tentativas[0]["prob"] is None:
        tentativas.append(perguntar(texto_estado, chave, usar_cache=False, sufixo_cache="_tentativa2"))
    prob = tentativas[-1]["prob"]
    return {
        "prob_subir": prob,
        "latencia_ms": sum(t["latencia_ms"] or 0 for t in tentativas),
        "custo_usd": sum(t["custo_usd"] or 0 for t in tentativas),
        "chamadas_pagas": sum(not t["do_cache"] for t in tentativas),
        "custo_pago_agora": sum(t["custo_usd"] or 0 for t in tentativas if not t["do_cache"]),
    }


# ---------------------------------------------------------------------------
# Etapas "python main.py jev" e "python main.py jev --teste"
# ---------------------------------------------------------------------------

def carregar_falhas() -> pd.DataFrame:
    return api_comum.carregar_falhas(config.ARQUIVO_FALHAS_JEV)


def estados_da_janela() -> pd.DataFrame:
    df = dados.carregar_indicadores()
    df = df[(df["ticker"] != config.REFERENCIA) & (df["periodo"] == "teste_limpo")]
    return estado.gerar_estados(df)  # o MESMO texto que o LLM recebe


def rodar(refazer: bool = False) -> None:
    chave = carregar_chave()
    verificar_preregistro()

    estados = estados_da_janela()
    if refazer:
        decisoes.apagar(NOME)
        config.ARQUIVO_FALHAS_JEV.unlink(missing_ok=True)
    pendentes = decisoes.pares_pendentes(NOME, estados, ja_resolvidos=carregar_falhas())
    if pendentes.empty:
        print("Jev: todos os pares (dia, ação) da janela limpa já foram decididos. Nada a fazer.")
        return

    print(f"Jev ({config.JEV_VERSAO}): {len(pendentes)} decisões a tomar em {pendentes['data'].nunique()} dia(s). "
          "Respostas já guardadas no cache não são pagas de novo.")
    pagas, custo_total = 0, 0.0
    for dia, do_dia in pendentes.groupby("data"):
        linhas, falhas = [], []
        for _, linha in do_dia.iterrows():
            r = decidir(linha["estado"], chave)
            pagas += r["chamadas_pagas"]
            custo_total += r["custo_pago_agora"]
            if r["prob_subir"] is None:
                falhas.append({
                    "data": dia, "ticker": linha["ticker"], "versao": config.JEV_VERSAO,
                    "pergunta_versao": PERGUNTA_VERSAO,
                    "registrado_em": datetime.now().isoformat(timespec="seconds"),
                })
                continue
            linhas.append({
                "data": dia, "ticker": linha["ticker"], "prob_subir": r["prob_subir"],
                "decisao": "comprar" if r["prob_subir"] > config.LIMIAR_COMPRA else "ficar_de_fora",
                "latencia_ms": r["latencia_ms"], "custo_usd": r["custo_usd"],
            })
        # Grava dia a dia: se algo der errado no meio, o que já foi pago fica salvo.
        if linhas:
            decisoes.acrescentar(pd.DataFrame(linhas), NOME)
        api_comum.registrar_falhas(config.ARQUIVO_FALHAS_JEV, falhas)
        if falhas:
            print(f"  {dia:%d/%m/%Y}: {len(falhas)} resposta(s) fora do formato mesmo após nova tentativa "
                  f"-> registradas em {config.ARQUIVO_FALHAS_JEV.relative_to(config.RAIZ)}")

    print(f"\nJev: {pagas} chamada(s) paga(s) nesta execução, custo estimado US$ {custo_total:.6f} "
          "(pelo preço publicado por token de entrada).")


def testar() -> None:
    """
    Uma única pergunta, num pregão ANTIGO (fora da janela limpa), só para conferir chave,
    versão e formato da resposta. Não grava decisão nenhuma e não exige o pré-registro.
    """
    chave = carregar_chave()
    df = dados.carregar_indicadores()
    antigo = df[(df["ticker"] != config.REFERENCIA) & (df["periodo"] == "historico")].iloc[[0]]
    texto = estado.gerar_estados(antigo)["estado"].iloc[0]

    print(f"Versão: {config.JEV_VERSAO}\nEstado enviado: {texto}\nPergunta (Noul): {PERGUNTA}\n")
    r = decidir(texto, chave)
    if r["prob_subir"] is None:
        print("-> resposta fora do formato mesmo após nova tentativa.")
    else:
        print(f"-> probabilidade de \"sim\" (prob_subir) = {r['prob_subir']}")
    print(f"Latência: {r['latencia_ms']:.0f} ms | custo estimado: US$ {r['custo_usd']:.8f} | "
          f"chamadas pagas: {r['chamadas_pagas']}")
