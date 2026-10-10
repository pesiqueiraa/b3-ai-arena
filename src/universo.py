"""
universo.py
Quais ações estão "na lista" em cada mês: uma aproximação do IBrX-100 que usa só o que se
sabia na época (sem viés de sobrevivência).

Regra, no primeiro pregão de cada mês:
    1. volume financeiro médio diário dos 'janela' pregões ANTERIORES (dia sem negócio conta como zero);
    2. uma ação por empresa (mesma raiz de 4 letras): fica a mais líquida;
    3. ficam as 'tamanho' maiores.
"""

import pandas as pd

import config

COLUNAS = ["mes", "data_inicio", "ticker", "posicao", "volume_medio"]


def montar_universo(b3: pd.DataFrame, tamanho: int = config.TAMANHO_UNIVERSO,
                    janela: int = config.JANELA_LIQUIDEZ) -> pd.DataFrame:
    calendario = pd.DatetimeIndex(sorted(b3["data"].unique()))
    volumes = (b3.pivot_table(index="data", columns="ticker", values="volume_financeiro", aggfunc="sum")
                 .reindex(calendario).fillna(0.0))
    primeiros = pd.Series(calendario).groupby(calendario.to_period("M")).first()

    linhas = []
    for mes, dia in primeiros.items():
        posicao_do_dia = calendario.get_loc(dia)
        if posicao_do_dia < janela:
            continue  # ainda não há 'janela' pregões de histórico
        media = volumes.iloc[posicao_do_dia - janela:posicao_do_dia].sum() / janela
        media = media[media > 0]
        mais_liquida_por_empresa = media.groupby(media.index.str[:4]).idxmax()
        escolhidas = media.loc[mais_liquida_por_empresa.values].sort_values(ascending=False).head(tamanho)
        for posicao, (ticker, volume) in enumerate(escolhidas.items(), start=1):
            linhas.append({"mes": str(mes), "data_inicio": dia, "ticker": ticker,
                           "posicao": posicao, "volume_medio": float(volume)})
    return pd.DataFrame(linhas, columns=COLUNAS)


def membros_por_dia(universo: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Expande o universo mensal para pares (data, ticker): cada dia usa a lista do seu mês."""
    dias = pd.DataFrame({"data": calendario})
    dias["mes"] = dias["data"].dt.to_period("M").astype(str)
    membros = dias.merge(universo[["mes", "ticker"]], on="mes", how="inner")
    return membros[["data", "ticker"]].sort_values(["data", "ticker"]).reset_index(drop=True)


def calcular_cobertura(universo: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """Para cada mês: quantas ações do universo têm preço ajustado (Yahoo) naquele mês."""
    com_preco = precos.assign(mes=precos["data"].dt.to_period("M").astype(str))[["mes", "ticker"]].drop_duplicates()
    marcado = universo.merge(com_preco, on=["mes", "ticker"], how="left", indicator=True)
    marcado["tem_preco"] = marcado["_merge"] == "both"
    resumo = marcado.groupby("mes").agg(no_universo=("ticker", "count"), com_preco=("tem_preco", "sum")).reset_index()
    resumo["cobertura"] = resumo["com_preco"] / resumo["no_universo"]
    return resumo
