"""
llm.py
Decisões de um LLM via API compatível com OpenRouter.

STATUS: ainda não implementado (próxima etapa do projeto).

Plano:
    - Provedor e modelo configuráveis pelo .env (LLM_API_KEY, LLM_MODELO).
    - Temperatura 0 (config.LLM_TEMPERATURA) e nenhuma ferramenta de busca na web.
    - Pede só o JSON {"prob_subir": <número de 0 a 1>} e valida. Se vier inválido,
      tenta de novo uma vez e registra a falha.
    - Recebe o texto de src/estado.py (o mesmo do Jev).
    - Cache em disco em resultados/cache/, latência e custo de cada chamada,
      só nos dias com periodo == "teste_limpo".
    - Grava com src/decisoes.py (nome "llm").
"""

NOME = "llm"


def rodar(refazer: bool = False) -> None:
    raise NotImplementedError("llm.py ainda não foi implementado (próxima etapa).")
