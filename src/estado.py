"""
estado.py
Transforma uma linha de indicadores no "estado" em texto enviado ao Jev e ao LLM.
Os dois recebem EXATAMENTE o mesmo texto, gerado aqui.

STATUS: ainda não implementado (próxima etapa do projeto).

O que vai fazer (conforme o plano):
    - Texto curto, com números já calculados e em porcentagem. Nada de tabela crua,
      porque o Jev é fraco em aritmética e em comparar datas. Exemplo:
      "PETR4. Retorno em 5 pregões: +3,8%. Preço 6,8% acima da média de 20 pregões.
       Volume 30% acima do normal."
    - anonimizar=True: remove ticker, nome da empresa e datas (vira "Ação A"),
      para o teste de memorização.
    - Nunca pode usar a coluna gabarito_retorno_amanha.
"""

import pandas as pd


def gerar_estado(linha: pd.Series, anonimizar: bool = False, rotulo_anonimo: str = "Ação A") -> str:
    # TODO (próxima etapa): montar o texto a partir de retorno_1d, retorno_5d, retorno_20d,
    # distancia_media_20d, volatilidade_20d e volume_relativo_20d.
    raise NotImplementedError("estado.py ainda não foi implementado (próxima etapa).")
