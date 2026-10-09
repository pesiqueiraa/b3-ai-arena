"""
b3.py
Séries históricas OFICIAIS da B3 (arquivos COTAHIST). Elas têm todas as ações negociadas,
inclusive as que deixaram de existir. Uso: montar o universo de cada mês sem viés de
sobrevivência. Os preços daqui NÃO são ajustados por dividendos; os retornos vêm do Yahoo.

Layout oficial (posições fixas, 245 caracteres por linha). Campos usados (posições a partir de 1):
    TIPREG 1-2 ("01" = cotação), DATA 3-10, CODBDI 11-12 ("02" = lote padrão),
    CODNEG 13-24 (ticker), TPMERC 25-27 ("010" = à vista), NOMRES 28-39 (nome),
    ESPECI 40-49 (ON, PN, UNT...), PREULT 109-121 (fechamento, 2 decimais),
    VOLTOT 171-188 (volume financeiro, 2 decimais).
"""

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

import config

COLUNAS = ["data", "ticker", "nome", "especie", "fechamento", "volume_financeiro"]


def _especie(especi: str) -> str | None:
    """'PN      N2' -> 'PN'. Devolve None se não for uma espécie aceita (BDR, fundo...)."""
    partes = especi.split()
    if partes and partes[0] in config.ESPECIES_ACEITAS:
        return partes[0]
    return None


def ler_linhas(linhas: Iterable[str]) -> pd.DataFrame:
    """Converte linhas do COTAHIST numa tabela, só com ações à vista no lote padrão."""
    registros = []
    for linha in linhas:
        if len(linha) < 188 or linha[0:2] != "01":
            continue
        if linha[10:12] != "02" or linha[24:27] != "010":
            continue
        especie = _especie(linha[39:49])
        if especie is None:
            continue
        registros.append({
            "data": linha[2:10],
            "ticker": linha[12:24].strip(),
            "nome": linha[27:39].strip(),
            "especie": especie,
            "fechamento": int(linha[108:121]) / 100,
            "volume_financeiro": int(linha[170:188]) / 100,
        })
    tabela = pd.DataFrame(registros, columns=COLUNAS)
    tabela["data"] = pd.to_datetime(tabela["data"], format="%Y%m%d")
    return tabela


def caminho_do_ano(ano: int) -> Path:
    return config.PASTA_B3 / f"COTAHIST_A{ano}.ZIP"


def baixar_ano(ano: int, forcar: bool = False) -> Path:
    """Baixa o arquivo do ano (se ainda não existir, ou se forcar=True) e devolve o caminho."""
    import requests

    destino = caminho_do_ano(ano)
    if destino.exists() and not forcar:
        return destino
    config.PASTA_B3.mkdir(parents=True, exist_ok=True)
    resposta = requests.get(config.URL_COTAHIST.format(ano=ano), timeout=600)
    resposta.raise_for_status()
    destino.write_bytes(resposta.content)
    return destino


def ler_ano(caminho: Path) -> pd.DataFrame:
    """Lê um arquivo COTAHIST_A{ano}.ZIP (texto em latin-1 dentro do zip)."""
    with zipfile.ZipFile(caminho) as arquivo:
        with arquivo.open(arquivo.namelist()[0]) as bruto:
            return ler_linhas(io.TextIOWrapper(bruto, encoding="latin-1"))


def tickers_com_troca_de_nome(b3: pd.DataFrame) -> pd.DataFrame:
    """
    Tickers que aparecem com mais de um nome de empresa. Pode ser só uma troca de nome,
    ou o ticker reaproveitado por outra empresa. A etapa 1 só relata os casos.
    """
    nomes = b3.groupby("ticker")["nome"].apply(lambda s: sorted(s.unique()))
    nomes = nomes[nomes.str.len() > 1]
    return pd.DataFrame({"ticker": nomes.index, "nomes": nomes.str.join(" | ").values}).reset_index(drop=True)
