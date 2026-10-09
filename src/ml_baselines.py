"""
ml_e_baselines.py
Passo 2 do experimento: gera as decisões do modelo de ML e do baseline "sempre comprar".

Lê:   dados/indicadores_b3.csv   (criado pelo baixar_acoes.py)
Gera: dados/decisoes_ml.csv
      dados/decisoes_sempre_comprar.csv

Todo arquivo dados/decisoes_<nome>.csv segue o MESMO formato, uma linha por ação por dia:
    data, ticker, prob_subir, decisao
    - prob_subir: chance (0 a 1) que o modelo dá de a ação subir no próximo pregão
    - decisao:    "comprar" ou "ficar_de_fora"
O Jev e o LLM vão gerar arquivos nesse mesmo formato, e o avaliar.py compara todos.

Como rodar:
    pip install pandas scikit-learn
    python ml_e_baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PASTA = Path("dados")
ARQUIVO_INDICADORES = PASTA / "indicadores_b3.csv"

# Indicadores que os modelos podem ver. O gabarito NUNCA entra aqui.
FEATURES = [
    "retorno_1d", "retorno_5d", "retorno_20d",
    "distancia_media_20d", "volatilidade_20d", "volume_relativo_20d",
]

# BOVA11 é a régua de comparação, não uma ação para o modelo escolher.
REFERENCIA = "BOVA11"

# Compra quando a probabilidade de subir passa deste limiar.
LIMIAR_COMPRA = 0.5


def carregar() -> pd.DataFrame:
    if not ARQUIVO_INDICADORES.exists():
        raise SystemExit(f"Não achei {ARQUIVO_INDICADORES}. Rode antes o baixar_acoes.py.")
    df = pd.read_csv(ARQUIVO_INDICADORES, parse_dates=["data"])
    df = df[df["ticker"] != REFERENCIA].copy()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def decisoes_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treino "walk-forward": para decidir no dia D, o modelo treina SÓ com linhas de antes
    de D. A linha de D-1 pode entrar, porque o gabarito dela (retorno de D-1 para D)
    já é conhecido no fechamento de D, que é quando a decisão é tomada.
    """
    dias_teste = sorted(df.loc[df["periodo"] == "teste_limpo", "data"].unique())
    resultados = []

    for dia in dias_teste:
        treino = df[df["data"] < dia].dropna(subset=FEATURES + ["gabarito_retorno_amanha"])
        hoje = df[df["data"] == dia].dropna(subset=FEATURES)
        if treino.empty or hoje.empty:
            continue

        alvo = (treino["gabarito_retorno_amanha"] > 0).astype(int)  # 1 = subiu no dia seguinte

        # Regressão logística: um modelo simples e honesto como primeira referência.
        # O StandardScaler fica dentro do pipeline, então a normalização também
        # só aprende com os dados de treino.
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        modelo.fit(treino[FEATURES], alvo)

        prob = modelo.predict_proba(hoje[FEATURES])[:, 1]
        resultados.append(pd.DataFrame({
            "data": hoje["data"].values,
            "ticker": hoje["ticker"].values,
            "prob_subir": prob,
        }))

    saida = pd.concat(resultados, ignore_index=True)
    saida["decisao"] = np.where(saida["prob_subir"] > LIMIAR_COMPRA, "comprar", "ficar_de_fora")
    return saida


def decisoes_sempre_comprar(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: compra todas as ações todo dia (carteira igualitária)."""
    hoje = df[df["periodo"] == "teste_limpo"][["data", "ticker"]].copy()
    hoje["prob_subir"] = 1.0
    hoje["decisao"] = "comprar"
    return hoje


def salvar(decisoes: pd.DataFrame, nome: str) -> None:
    arquivo = PASTA / f"decisoes_{nome}.csv"
    decisoes.sort_values(["data", "ticker"]).to_csv(
        arquivo, index=False, date_format="%Y-%m-%d", float_format="%.4f"
    )
    print(f"Salvo: {arquivo} ({len(decisoes)} decisões)")


def main() -> None:
    df = carregar()

    ml = decisoes_ml(df)
    salvar(ml, "ml")
    salvar(decisoes_sempre_comprar(df), "sempre_comprar")

    # Decisões para o próximo pregão (o dia mais recente, que ainda não tem gabarito).
    ultimo_dia = ml["data"].max()
    ultimas = ml[ml["data"] == ultimo_dia].sort_values("prob_subir", ascending=False)
    compras = ultimas[ultimas["decisao"] == "comprar"]
    print(f"\nCom os dados de {ultimo_dia:%d/%m/%Y}, o ML compraria para o próximo pregão:")
    if compras.empty:
        print("  nenhuma ação")
    for _, linha in compras.iterrows():
        print(f"  {linha['ticker']:<7} chance de subir: {linha['prob_subir']:.0%}")


if __name__ == "__main__":
    main()