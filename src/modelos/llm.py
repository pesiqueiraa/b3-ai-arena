"""
llm.py
Decisões de um LLM de uso geral, via API compatível com OpenRouter.

Como funciona, para cada ação em cada dia da janela limpa:
    1. pega o estado em texto de src/estado.py (o MESMO que o Jev recebe);
    2. pergunta ao LLM a probabilidade de a ação fechar o próximo pregão em alta,
       pedindo como resposta SÓ o JSON {"prob_subir": <número de 0 a 1>};
    3. se a resposta vier fora do formato, pede de novo UMA vez; se falhar de novo,
       registra a falha em resultados/falhas_llm.csv e segue em frente;
    4. grava a decisão em resultados/decisoes/decisoes_llm.csv, com latência e custo.

Regras:
    - Temperatura 0 e NENHUMA busca na web (sem ":online", sem plugin "web", sem ferramentas).
    - Toda resposta fica guardada em resultados/cache/llm/. A mesma pergunta nunca é paga duas vezes.
    - Só roda se o modelo e a versão do prompt estiverem registrados no PREREGISTRO.md.

Configuração no arquivo .env: LLM_API_KEY e LLM_MODELO (ex.: provedor/nome-do-modelo).
Documentação usada: https://openrouter.ai/docs/quickstart,
https://openrouter.ai/docs/api-reference/chat-completion, .../use-cases/usage-accounting,
.../features/web-search (consultadas em 09/10/2026).
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime

import pandas as pd

import config
from src import dados, decisoes, estado

NOME = "llm"

# ---------------------------------------------------------------------------
# O prompt. Mudou o texto? Mude a versão: perguntas diferentes não podem dividir o cache,
# e a versão nova precisa ser registrada no PREREGISTRO.md antes de rodar.
# ---------------------------------------------------------------------------

PROMPT_VERSAO = "llm-prompt-v1"

PERGUNTA = "A ação vai fechar o próximo pregão acima do fechamento de hoje?"

INSTRUCOES = (
    "Você vai receber um resumo dos indicadores de uma ação da bolsa brasileira (B3) "
    "no fechamento de um pregão. Responda à pergunta com a probabilidade de a resposta ser \"sim\". "
    "Responda APENAS com um JSON no formato {\"prob_subir\": <número de 0 a 1>}, sem nenhum outro texto."
)

CORRECAO = (
    "Sua resposta não seguiu o formato pedido. Responda APENAS com o JSON "
    "{\"prob_subir\": <número de 0 a 1>}, sem nenhum outro texto."
)


class ErroDeConfiguracao(Exception):
    """Algo no .env ou no pré-registro impede a etapa de rodar."""


class ErroDeAPI(Exception):
    """A API não respondeu direito mesmo depois das novas tentativas de rede."""


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def carregar_configuracao() -> tuple[str, str]:
    """Lê LLM_API_KEY e LLM_MODELO do .env e confere se nada liga busca na web."""
    from dotenv import load_dotenv

    load_dotenv(config.RAIZ / ".env")
    chave = os.getenv("LLM_API_KEY", "").strip()
    modelo = os.getenv("LLM_MODELO", "").strip()
    if not chave or not modelo:
        raise ErroDeConfiguracao(
            "faltam LLM_API_KEY e/ou LLM_MODELO no arquivo .env "
            "(copie o .env.example para .env e preencha)."
        )
    if ":online" in modelo:
        raise ErroDeConfiguracao(
            f"o modelo '{modelo}' tem ':online', que liga a busca na web. Use o nome sem ':online'."
        )
    return chave, modelo


def verificar_preregistro(modelo: str) -> None:
    """
    O PREREGISTRO.md manda congelar o modelo e o prompt ANTES da primeira chamada.
    Aqui conferimos que os dois aparecem lá (entre crases), senão a etapa não roda.
    """
    texto = config.ARQUIVO_PREREGISTRO.read_text(encoding="utf-8")
    faltando = [item for item in (modelo, PROMPT_VERSAO) if f"`{item}`" not in texto]
    if faltando:
        raise ErroDeConfiguracao(
            "o PREREGISTRO.md ainda não registra " + " e ".join(f"`{f}`" for f in faltando)
            + ". Escreva o adendo (modelo, versão do prompt e texto do prompt), faça o commit "
            "e só então rode o LLM. Para só testar a chave e o modelo: python main.py llm --teste"
        )


# ---------------------------------------------------------------------------
# A conversa com o modelo
# ---------------------------------------------------------------------------

def montar_mensagens(texto_estado: str) -> list[dict]:
    return [
        {"role": "system", "content": INSTRUCOES},
        {"role": "user", "content": f"{texto_estado}\n\nPergunta: {PERGUNTA}"},
    ]


def montar_pedido(mensagens: list[dict], modelo: str) -> dict:
    """
    O corpo da requisição. De propósito, NÃO há "tools", "plugins" nem "web_search_options":
    é assim que a busca na web fica desligada no OpenRouter.
    'provider' fixa quem roda o modelo; 'reasoning' fixa o nível de raciocínio (ver config.py).
    """
    pedido = {
        "model": modelo,
        "messages": mensagens,
        "temperature": config.LLM_TEMPERATURA,
        "seed": config.SEMENTE,
        "provider": config.LLM_PROVEDOR,
    }
    if config.LLM_RACIOCINIO is not None:
        pedido["reasoning"] = config.LLM_RACIOCINIO
    return pedido


def validar_resposta(texto: str | None) -> float | None:
    """
    Devolve a probabilidade se a resposta for exatamente {"prob_subir": <0 a 1>}.
    Aceita só um detalhe de forma: o JSON vir dentro de um bloco ```json ... ```.
    Qualquer outra coisa (texto extra, chave diferente, número fora de 0 a 1) vale None.
    """
    if not isinstance(texto, str):
        return None
    limpo = texto.strip()
    bloco = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", limpo, flags=re.DOTALL)
    if bloco:
        limpo = bloco.group(1)
    try:
        objeto = json.loads(limpo)
    except json.JSONDecodeError:
        return None
    if not isinstance(objeto, dict) or set(objeto) != {"prob_subir"}:
        return None
    valor = objeto["prob_subir"]
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    if not 0 <= valor <= 1:
        return None
    return float(valor)


def _post(pedido: dict, chave: str) -> dict:
    """A chamada HTTP de verdade (trocada por uma falsa nos testes)."""
    import requests

    resposta = requests.post(
        config.LLM_URL,
        headers={"Authorization": f"Bearer {chave}", "Content-Type": "application/json"},
        json=pedido,
        timeout=config.LLM_TIMEOUT_S,
    )
    if resposta.status_code == 429 or resposta.status_code >= 500:
        raise ConnectionError(f"HTTP {resposta.status_code}: {resposta.text[:200]}")
    if resposta.status_code != 200:
        raise ErroDeAPI(f"HTTP {resposta.status_code}: {resposta.text[:500]}")
    return resposta.json()


def _chave_do_cache(pedido: dict) -> str:
    return hashlib.sha256(json.dumps(pedido, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def perguntar(mensagens: list[dict], modelo: str, chave: str) -> dict:
    """
    Faz UMA pergunta, usando o cache sempre que possível.
    Devolve: texto da resposta, latência e custo da chamada ORIGINAL (mesmo se veio do cache),
    e se veio do cache.
    """
    pedido = montar_pedido(mensagens, modelo)
    pasta = config.PASTA_CACHE / NOME
    arquivo = pasta / f"{_chave_do_cache(pedido)}.json"
    if arquivo.exists():
        guardado = json.loads(arquivo.read_text(encoding="utf-8"))
        return {**guardado["resultado"], "do_cache": True}

    for tentativa in range(1, config.LLM_TENTATIVAS_REDE + 1):
        inicio = time.perf_counter()
        try:
            bruto = _post(pedido, chave)
            break
        except (ConnectionError, TimeoutError, OSError) as erro:
            if tentativa == config.LLM_TENTATIVAS_REDE:
                raise ErroDeAPI(f"a API não respondeu após {tentativa} tentativas: {erro}") from erro
            espera = 2 ** tentativa
            print(f"  (erro de rede: {erro}. Tentando de novo em {espera}s...)")
            time.sleep(espera)
    latencia_ms = (time.perf_counter() - inicio) * 1000

    try:
        texto = bruto["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        texto = None
    uso = bruto.get("usage") or {}
    resultado = {
        "texto": texto,
        "latencia_ms": round(latencia_ms, 1),
        # usage.cost: "total amount charged to your account", em créditos do OpenRouter.
        "custo_usd": uso.get("cost"),
        "modelo_respondeu": bruto.get("model"),
        "provedor_respondeu": bruto.get("provider"),
        "tokens_raciocinio": ((uso.get("completion_tokens_details") or {}).get("reasoning_tokens")),
    }

    pasta.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps({
        "quando": datetime.now().isoformat(timespec="seconds"),
        "pedido": pedido,
        "resposta_bruta": bruto,
        "resultado": resultado,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**resultado, "do_cache": False}


def decidir(texto_estado: str, modelo: str, chave: str) -> dict:
    """
    Uma decisão completa: pergunta, valida e, se preciso, pede de novo UMA vez.
    Latência e custo somam as duas tentativas quando houver segunda.
    """
    mensagens = montar_mensagens(texto_estado)
    primeira = perguntar(mensagens, modelo, chave)
    tentativas = [primeira]
    prob = validar_resposta(primeira["texto"])

    if prob is None:
        mensagens_2 = mensagens + [
            {"role": "assistant", "content": primeira["texto"] or ""},
            {"role": "user", "content": CORRECAO},
        ]
        segunda = perguntar(mensagens_2, modelo, chave)
        tentativas.append(segunda)
        prob = validar_resposta(segunda["texto"])

    return {
        "prob_subir": prob,
        "latencia_ms": sum(t["latencia_ms"] or 0 for t in tentativas),
        "custo_usd": sum(t["custo_usd"] or 0 for t in tentativas),
        "respostas": [t["texto"] for t in tentativas],
        "chamadas_pagas": sum(not t["do_cache"] for t in tentativas),
        "custo_pago_agora": sum(t["custo_usd"] or 0 for t in tentativas if not t["do_cache"]),
        "provedores": [t.get("provedor_respondeu") for t in tentativas],
        "tokens_raciocinio": [t.get("tokens_raciocinio") for t in tentativas],
    }


# ---------------------------------------------------------------------------
# Falhas (resposta inválida mesmo na segunda tentativa)
# ---------------------------------------------------------------------------

def carregar_falhas() -> pd.DataFrame:
    if not config.ARQUIVO_FALHAS_LLM.exists():
        return pd.DataFrame(columns=["data", "ticker"])
    return pd.read_csv(config.ARQUIVO_FALHAS_LLM, parse_dates=["data"])


def registrar_falhas(novas: list[dict]) -> None:
    if not novas:
        return
    tabela = pd.concat([carregar_falhas(), pd.DataFrame(novas)], ignore_index=True)
    config.ARQUIVO_FALHAS_LLM.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(config.ARQUIVO_FALHAS_LLM, index=False, date_format="%Y-%m-%d")


# ---------------------------------------------------------------------------
# Etapas "python main.py llm" e "python main.py llm --teste"
# ---------------------------------------------------------------------------

def estados_da_janela() -> pd.DataFrame:
    df = dados.carregar_indicadores()
    df = df[(df["ticker"] != config.REFERENCIA) & (df["periodo"] == "teste_limpo")]
    return estado.gerar_estados(df)  # o MESMO texto que o Jev recebe


def rodar(refazer: bool = False) -> None:
    chave, modelo = carregar_configuracao()
    verificar_preregistro(modelo)

    estados = estados_da_janela()
    if refazer:
        decisoes.apagar(NOME)
        config.ARQUIVO_FALHAS_LLM.unlink(missing_ok=True)
    pendentes = decisoes.pares_pendentes(NOME, estados, ja_resolvidos=carregar_falhas())
    if pendentes.empty:
        print("LLM: todos os pares (dia, ação) da janela limpa já foram decididos. Nada a fazer.")
        return

    print(f"LLM ({modelo}): {len(pendentes)} decisões a tomar em {pendentes['data'].nunique()} dia(s). "
          "Respostas já guardadas no cache não são pagas de novo.")
    pagas, custo_total = 0, 0.0
    for dia, do_dia in pendentes.groupby("data"):
        linhas, falhas = [], []
        for _, linha in do_dia.iterrows():
            r = decidir(linha["estado"], modelo, chave)
            pagas += r["chamadas_pagas"]
            custo_total += r["custo_pago_agora"]
            if r["prob_subir"] is None:
                falhas.append({
                    "data": dia, "ticker": linha["ticker"], "modelo": modelo,
                    "prompt_versao": PROMPT_VERSAO, "respostas": json.dumps(r["respostas"], ensure_ascii=False),
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
        registrar_falhas(falhas)
        if falhas:
            print(f"  {dia:%d/%m/%Y}: {len(falhas)} resposta(s) inválida(s) mesmo após nova tentativa "
                  f"-> registradas em {config.ARQUIVO_FALHAS_LLM.relative_to(config.RAIZ)}")

    print(f"\nLLM: {pagas} chamada(s) paga(s) nesta execução, custo total {custo_total:.4f} "
          "(créditos do OpenRouter).")


def testar() -> None:
    """
    Uma única pergunta, num pregão ANTIGO (fora da janela limpa), só para conferir chave,
    modelo e formato da resposta. Não grava decisão nenhuma e não exige o pré-registro.
    """
    chave, modelo = carregar_configuracao()
    df = dados.carregar_indicadores()
    antigo = df[(df["ticker"] != config.REFERENCIA) & (df["periodo"] == "historico")].iloc[[0]]
    texto = estado.gerar_estados(antigo)["estado"].iloc[0]

    print(f"Modelo: {modelo}\nEstado enviado: {texto}\nPergunta: {PERGUNTA}\n")
    r = decidir(texto, modelo, chave)
    print(f"Resposta(s) do modelo: {r['respostas']}")
    if r["prob_subir"] is None:
        print("-> INVÁLIDA mesmo após nova tentativa. Este modelo não segue o formato; considere outro.")
    else:
        print(f"-> válida: prob_subir = {r['prob_subir']}")
    print(f"Latência: {r['latencia_ms']:.0f} ms | custo: {r['custo_usd']} | chamadas pagas: {r['chamadas_pagas']}")
    print(f"Provedor que respondeu: {r['provedores']} | tokens de raciocínio: {r['tokens_raciocinio']}")
