"""
decisoes.py
Ler, validar e gravar os arquivos resultados/decisoes/decisoes_<modelo>.csv.

Todo modelo usa este módulo, então todos os arquivos saem no MESMO formato:
    data, ticker, prob_subir, decisao   (+ opcionais: latencia_ms, custo_usd)

Os arquivos são "só de acréscimo": um dia que já tem decisão gravada não é
recalculado (a não ser que você peça com --refazer). Isso importa porque as
decisões vão para o git: se o commit de uma decisão foi feito antes do pregão
seguinte, ele prova que ela foi tomada sem conhecer o resultado. Reescrever
o passado depois apagaria essa prova.
"""

import pandas as pd

import config

COLUNAS_OBRIGATORIAS = ["data", "ticker", "prob_subir", "decisao"]
COLUNAS_OPCIONAIS = ["latencia_ms", "custo_usd"]
DECISOES_VALIDAS = {"comprar", "ficar_de_fora"}


def arquivo_de(nome: str):
    return config.PASTA_DECISOES / f"decisoes_{nome}.csv"


def validar(decisoes: pd.DataFrame, nome: str) -> None:
    """Confere a regra 3 do projeto. Para o programa com uma mensagem clara se algo estiver errado."""
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in decisoes.columns]
    if faltando:
        raise ValueError(f"[{nome}] faltam as colunas obrigatórias: {faltando}")

    extras = [c for c in decisoes.columns if c not in COLUNAS_OBRIGATORIAS + COLUNAS_OPCIONAIS]
    if extras:
        raise ValueError(f"[{nome}] colunas não permitidas: {extras} "
                         f"(só são aceitas {COLUNAS_OBRIGATORIAS + COLUNAS_OPCIONAIS})")

    invalidas = set(decisoes["decisao"]) - DECISOES_VALIDAS
    if invalidas:
        raise ValueError(f"[{nome}] valores inválidos em 'decisao': {invalidas}")

    prob = decisoes["prob_subir"]
    if prob.isna().any() or (prob < 0).any() or (prob > 1).any():
        raise ValueError(f"[{nome}] 'prob_subir' precisa estar entre 0 e 1, sem valores vazios")

    if decisoes.duplicated(["data", "ticker"]).any():
        raise ValueError(f"[{nome}] há mais de uma decisão para o mesmo dia e ação")


def carregar(nome: str) -> pd.DataFrame:
    """Lê as decisões já gravadas de um modelo (tabela vazia se ainda não existem)."""
    arquivo = arquivo_de(nome)
    if not arquivo.exists():
        return pd.DataFrame(columns=COLUNAS_OBRIGATORIAS)
    return pd.read_csv(arquivo, parse_dates=["data"])


def dias_pendentes(nome: str, dias: list, refazer: bool = False) -> list:
    """Dos dias pedidos, devolve os que ainda NÃO têm decisão gravada."""
    if refazer:
        return list(dias)
    ja_decididos = set(carregar(nome)["data"])
    return [d for d in dias if d not in ja_decididos]


def pares_pendentes(nome: str, pares: pd.DataFrame, ja_resolvidos: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Dos pares (data, ticker) pedidos, devolve os que ainda NÃO têm decisão gravada.
    Usado pelos modelos de API (Jev e LLM), que decidem ação por ação.
    'ja_resolvidos' permite pular também pares que falharam de vez (para não pagar de novo).
    """
    feitos = carregar(nome)[["data", "ticker"]]
    if ja_resolvidos is not None and not ja_resolvidos.empty:
        feitos = pd.concat([feitos, ja_resolvidos[["data", "ticker"]]], ignore_index=True)
    m = pares.merge(feitos.drop_duplicates(), on=["data", "ticker"], how="left", indicator=True)
    return m[m["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def apagar(nome: str) -> None:
    """Apaga as decisões gravadas de um modelo (usado só pelo --refazer)."""
    arquivo_de(nome).unlink(missing_ok=True)


def acrescentar(novas: pd.DataFrame, nome: str, refazer: bool = False) -> pd.DataFrame:
    """Junta as decisões novas às já gravadas, valida e salva. Devolve a tabela completa."""
    config.PASTA_DECISOES.mkdir(parents=True, exist_ok=True)
    antigas = carregar(nome)
    if refazer or antigas.empty:
        todas = novas
    else:
        todas = pd.concat([antigas, novas], ignore_index=True)

    validar(todas, nome)
    todas = todas.sort_values(["data", "ticker"]).reset_index(drop=True)
    todas.to_csv(arquivo_de(nome), index=False, date_format="%Y-%m-%d", float_format="%.6f")
    print(f"Salvo: {arquivo_de(nome).relative_to(config.RAIZ)} "
          f"({len(novas)} decisões novas, {len(todas)} no total)")
    return todas
