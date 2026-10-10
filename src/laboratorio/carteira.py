"""
carteira.py
Simula alguém seguindo os sinais de COMPRA.

    - O capital é dividido em 'horizonte' fatias iguais (10).
    - No fechamento de cada pregão, a fatia da vez vende o que tinha (pagando custo) e compra,
      com pesos iguais, as ações sinalizadas naquele dia (pagando custo). Sem sinal, fica em caixa.
    - Cada compra fica, portanto, exatamente 10 pregões na carteira.
    - Custo: CUSTO_POR_LADO (0,10%) sobre o valor comprado e sobre o valor vendido.
    - Ação que some: o retorno vira 0 (a posição fica parada no último preço).
"""

import pandas as pd

import config


def simular_compras(sinais: pd.DataFrame, retornos: pd.DataFrame, inicio, fim,
                    horizonte: int = config.HORIZONTE, custo: float = config.CUSTO_POR_LADO) -> pd.Series:
    dias = retornos.index[(retornos.index >= pd.Timestamp(inicio)) & (retornos.index <= pd.Timestamp(fim))]
    compras = sinais[sinais["sinal"] == "compra"].groupby("data")["ticker"].apply(list).to_dict()
    fatias = [{"caixa": 1.0 / horizonte, "posicoes": {}} for _ in range(horizonte)]

    valores = []
    for j, dia in enumerate(dias):
        retorno_do_dia = retornos.loc[dia]
        for fatia in fatias:                                   # 1) o dia passa para quem está comprado
            for ticker in fatia["posicoes"]:
                fatia["posicoes"][ticker] *= 1 + retorno_do_dia.get(ticker, 0.0)

        fatia = fatias[j % horizonte]                          # 2) a fatia da vez troca de posição
        total = fatia["caixa"] + sum(fatia["posicoes"].values()) * (1 - custo)
        fatia["posicoes"] = {}
        novas = compras.get(dia, [])
        if novas:
            total *= 1 - custo
            fatia["posicoes"] = {ticker: total / len(novas) for ticker in novas}
            fatia["caixa"] = 0.0
        else:
            fatia["caixa"] = total

        valores.append(sum(f["caixa"] + sum(f["posicoes"].values()) for f in fatias))

    serie = pd.Series(valores, index=dias, dtype=float)
    return serie / serie.shift(1, fill_value=1.0) - 1
