"""
jev.py
Decisões do Jev (TypeSafe AI) via API oficial, com versão FIXA (config.JEV_VERSAO = "jev-1.13.0").

STATUS: ainda não implementado (próxima etapa do projeto).

Plano:
    - Pergunta do tipo Noul: "A ação vai fechar o próximo pregão acima do fechamento de hoje?"
      A probabilidade de "sim" vira prob_subir.
    - Recebe o texto de src/estado.py (o mesmo do LLM).
    - Cache em disco em resultados/cache/, latência e custo de cada chamada,
      só nos dias com periodo == "teste_limpo".
    - Grava com src/decisoes.py (nome "jev").

TODO: consultar https://docs.typesafe.ai ANTES de implementar. Não inventar nomes de
campos, endpoints ou pacotes. A chave vem do .env (TYPESAFE_API_KEY).
"""

NOME = "jev"


def rodar(refazer: bool = False) -> None:
    raise NotImplementedError(
        "jev.py ainda não foi implementado: falta consultar a documentação oficial "
        "(https://docs.typesafe.ai) e escrever a chamada à API."
    )
