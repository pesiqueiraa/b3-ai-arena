"""
Modelo de ML clássico (regressão logística com treino walk-forward).

Arquivos desta pasta:
    features.py      -> quais indicadores o modelo pode ver (o gabarito nunca)
    modelo.py        -> a regressão logística (imputação + normalização + logística)
    walk_forward.py  -> treina de novo a cada dia, usando só o passado
    __init__.py      -> a etapa "python main.py ml" (este arquivo)
"""

import config
from src import dados, decisoes
from src.modelos.ml.walk_forward import gerar_decisoes

NOME = "ml"


def rodar(refazer: bool = False) -> None:
    df = dados.carregar_indicadores()
    df = df[df["ticker"] != config.REFERENCIA].copy()  # BOVA11 é régua, não escolha

    dias_teste = sorted(df.loc[df["periodo"] == "teste_limpo", "data"].unique())
    pendentes = decisoes.dias_pendentes(NOME, dias_teste, refazer)
    if not pendentes:
        print("ML: todos os dias da janela limpa já têm decisão gravada. Nada a fazer.")
        print("    (para recalcular tudo do zero: python main.py ml --refazer)")
        return

    print(f"ML: treinando e decidindo {len(pendentes)} dia(s) "
          f"({pendentes[0]:%d/%m/%Y} a {pendentes[-1]:%d/%m/%Y})...")
    novas = gerar_decisoes(df, pendentes)
    todas = decisoes.acrescentar(novas, NOME, refazer)

    # Decisões para o próximo pregão (o dia mais recente, que ainda não tem gabarito).
    ultimo_dia = todas["data"].max()
    ultimas = todas[todas["data"] == ultimo_dia].sort_values("prob_subir", ascending=False)
    compras = ultimas[ultimas["decisao"] == "comprar"]
    print(f"\nCom os dados de {ultimo_dia:%d/%m/%Y}, o ML compraria para o próximo pregão:")
    if compras.empty:
        print("  nenhuma ação")
    for _, linha in compras.iterrows():
        print(f"  {linha['ticker']:<7} chance de subir: {linha['prob_subir']:.0%}")
