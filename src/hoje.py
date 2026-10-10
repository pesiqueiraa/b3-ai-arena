"""
hoje.py
Etapa "python main.py sinais": compras e vendas para o PRÓXIMO pregão.

Treina com tudo o que já era conhecido no último pregão (com embargo) e aplica a configuração
escolhida pelo laboratório. Se o laboratório ainda não rodou, usa o padrão do config.py.

AVISO: os sinais são SÓ PARA LEITURA até a prova final ter resultado positivo.
"""

import json

import pandas as pd

import config
from src.base import Base, carregar_base
from src.features.preco import COLUNAS_MODELO
from src.laboratorio.walk_forward import limite_de_treino
from src.modelos import combinador
from src.sinais import gerar_sinais


def escolha_atual() -> dict:
    if config.ARQUIVO_ESCOLHA.exists():
        e = json.loads(config.ARQUIVO_ESCOLHA.read_text(encoding="utf-8"))
        return {"tipo": e["tipo"], "k": int(e["k"]), "p_min": float(e["p_min"])}
    return {"tipo": "logistica", "k": config.K_PADRAO, "p_min": config.P_MIN_PADRAO}


def calcular(base: Base, escolha: dict, treinar=combinador.treinar) -> tuple[pd.DataFrame, pd.Timestamp]:
    dia = base.tabela["data"].max()
    limite = limite_de_treino(base.calendario, dia)
    treino = base.tabela[(base.tabela["data"] <= limite) & base.tabela["rotulo"].notna()]
    modelo = treinar(treino[COLUNAS_MODELO], treino["rotulo"].astype(int), escolha["tipo"])
    do_dia = base.tabela[base.tabela["data"] == dia]
    probs = pd.DataFrame({"data": do_dia["data"].values, "ticker": do_dia["ticker"].values,
                          "prob": combinador.prob_superar(modelo, do_dia[COLUNAS_MODELO])})
    return gerar_sinais(probs, escolha["k"], escolha["p_min"]), dia


def rodar() -> None:
    escolha = escolha_atual()
    if not config.ARQUIVO_ESCOLHA.exists():
        print("Aviso: o laboratório ainda não rodou; usando a configuração padrão do config.py.")
    sinais, dia = calcular(carregar_base(), escolha)

    print("=" * 72)
    print("AVISO: sinais SÓ PARA LEITURA até a prova final ter resultado positivo.")
    if config.ARQUIVO_PROVA_FINAL.exists():
        aprovado = json.loads(config.ARQUIVO_PROVA_FINAL.read_text(encoding="utf-8"))["aprovado"]
        print(f"Prova final: {'APROVADA (use com cautela)' if aprovado else 'NÃO aprovada'}.")
    print("=" * 72)
    print(f"Com os dados de {dia:%d/%m/%Y} (modelo '{escolha['tipo']}', k = {escolha['k']}, "
          f"p_min = {escolha['p_min']:.2f}). Pergunta: vai superar o Ibovespa nos próximos 10 pregões?\n")
    for tipo, titulo in (("compra", "COMPRAR (mais chance de superar o Ibovespa)"),
                         ("venda", "VENDER se tiver (mais chance de ficar para trás)")):
        do_tipo = sinais[sinais["sinal"] == tipo]
        print(titulo + ":")
        if do_tipo.empty:
            print("  nenhuma ação com convicção suficiente hoje")
        for _, linha in do_tipo.iterrows():
            print(f"  {linha['ticker']:<8} chance de superar o Ibovespa: {linha['prob']:.0%}")
        print()
