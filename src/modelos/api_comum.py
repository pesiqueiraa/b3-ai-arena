"""
api_comum.py
Peças compartilhadas pelos modelos que chamam APIs pagas (Jev e LLM):
    - cache em disco: a mesma pergunta nunca é paga duas vezes;
    - registro de falhas (respostas inválidas mesmo após nova tentativa);
    - trava do pré-registro: a etapa só roda se a configuração estiver registrada.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config


class ErroDeConfiguracao(Exception):
    """Algo no .env ou no pré-registro impede a etapa de rodar."""


class ErroDeAPI(Exception):
    """A API não respondeu direito mesmo depois das novas tentativas de rede."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def chave_do_cache(pedido: dict) -> str:
    """Impressão digital do pedido: qualquer mudança no pedido gera outra chave."""
    texto = json.dumps(pedido, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def arquivo_do_cache(pasta: str, pedido: dict, sufixo: str = "") -> Path:
    return config.PASTA_CACHE / pasta / f"{chave_do_cache(pedido)}{sufixo}.json"


def ler_cache(arquivo: Path) -> dict | None:
    if not arquivo.exists():
        return None
    return json.loads(arquivo.read_text(encoding="utf-8"))["resultado"]


def gravar_cache(arquivo: Path, pedido: dict, bruto: dict, resultado: dict) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps({
        "quando": datetime.now().isoformat(timespec="seconds"),
        "pedido": pedido,
        "resposta_bruta": bruto,
        "resultado": resultado,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Falhas
# ---------------------------------------------------------------------------

def carregar_falhas(arquivo: Path) -> pd.DataFrame:
    if not arquivo.exists():
        return pd.DataFrame(columns=["data", "ticker"])
    return pd.read_csv(arquivo, parse_dates=["data"])


def registrar_falhas(arquivo: Path, novas: list[dict]) -> None:
    if not novas:
        return
    tabela = pd.concat([carregar_falhas(arquivo), pd.DataFrame(novas)], ignore_index=True)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(arquivo, index=False, date_format="%Y-%m-%d")


# ---------------------------------------------------------------------------
# Pré-registro
# ---------------------------------------------------------------------------

def exigir_no_preregistro(itens: list[str], comando_teste: str) -> None:
    """
    O PREREGISTRO.md manda congelar a configuração ANTES da primeira chamada.
    Cada item precisa aparecer lá entre crases; senão, a etapa não roda.
    """
    texto = config.ARQUIVO_PREREGISTRO.read_text(encoding="utf-8")
    faltando = [item for item in itens if f"`{item}`" not in texto]
    if faltando:
        raise ErroDeConfiguracao(
            "o PREREGISTRO.md ainda não registra " + " e ".join(f"`{f}`" for f in faltando)
            + ". Escreva o adendo, faça o commit e só então rode. "
            f"Para só testar a chave e o modelo: {comando_teste}"
        )
