"""
estado.py
Transforma uma linha de indicadores no "estado" em texto enviado ao Jev e ao LLM.
Os dois recebem EXATAMENTE o mesmo texto, gerado aqui (regra 4 do projeto).

Por que texto e não tabela: o Jev é fraco em aritmética e em comparar datas. Então
todos os números já chegam calculados, em porcentagem, com a direção escrita por
extenso ("acima", "abaixo"). Exemplo:

    PETR4 (Petrobras), pregão de 08/10/2026. Retorno no último pregão: +2,1%.
    Retorno em 5 pregões: +11,2%. Retorno em 20 pregões: +12,9%. Preço 10,5% acima
    da média de 20 pregões. Oscilação diária típica nos últimos 20 pregões: 2,5%.
    Volume 13% acima do normal.

Com anonimizar=True, ticker, nome da empresa e data somem (vira "Ação A"). Isso serve
para o teste de memorização: se o modelo decide diferente só por saber QUAL é a ação
ou QUANDO foi, ele está usando lembrança, e não os números.

O gabarito NUNCA é lido aqui: o texto é montado só com as colunas de ESTADO_COLUNAS.
"""

import math

import pandas as pd

import config

# As MESMAS seis informações que o ML recebe (src/modelos/ml/features.py).
ESTADO_COLUNAS = [
    "retorno_1d", "retorno_5d", "retorno_20d",
    "distancia_media_20d", "volatilidade_20d", "volume_relativo_20d",
]
assert not any("gabarito" in c for c in ESTADO_COLUNAS), "O gabarito não pode entrar no estado!"

ROTULO_ANONIMO = "Ação A"


# ---------------------------------------------------------------------------
# Formatação de números (padrão brasileiro: vírgula decimal)
# ---------------------------------------------------------------------------

def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)))


def formatar_pct(fracao: float, casas: int = 1, sinal: bool = True) -> str:
    """0.038 -> '+3,8%'. Com sinal=False: 0.038 -> '3,8%'. Nunca mostra '-0,0%'."""
    valor = round(fracao * 100, casas)
    if valor == 0:
        valor = 0.0  # evita "-0,0"
    texto = f"{valor:{'+' if sinal and valor != 0 else ''}.{casas}f}".replace(".", ",")
    return f"{texto}%"


def _frase_retorno(rotulo: str, fracao) -> str:
    if _vazio(fracao):
        return f"{rotulo}: sem dado."
    return f"{rotulo}: {formatar_pct(fracao)}."


def _frase_media(distancia) -> str:
    if _vazio(distancia):
        return "Distância da média de 20 pregões: sem dado."
    if round(distancia * 100, 1) == 0:
        return "Preço igual à média de 20 pregões."
    lado = "acima" if distancia > 0 else "abaixo"
    return f"Preço {formatar_pct(abs(distancia), sinal=False)} {lado} da média de 20 pregões."


def _frase_volatilidade(vol) -> str:
    if _vazio(vol):
        return "Oscilação diária típica nos últimos 20 pregões: sem dado."
    return f"Oscilação diária típica nos últimos 20 pregões: {formatar_pct(vol, sinal=False)}."


def _frase_volume(relativo) -> str:
    """volume_relativo_20d = volume de hoje / média de 20 pregões (1,0 = normal)."""
    if _vazio(relativo):
        return "Volume: sem dado."
    if relativo >= 2:
        vezes = f"{relativo:.1f}".replace(".", ",")
        return f"Volume {vezes} vezes o normal."
    diferenca = round((relativo - 1) * 100)
    if diferenca == 0:
        return "Volume igual ao normal."
    lado = "acima" if diferenca > 0 else "abaixo"
    return f"Volume {abs(diferenca)}% {lado} do normal."


# ---------------------------------------------------------------------------
# O estado
# ---------------------------------------------------------------------------

def gerar_estado(linha: pd.Series, anonimizar: bool = False) -> str:
    """Monta o texto de UMA ação em UM dia. Só lê data, ticker e ESTADO_COLUNAS."""
    if anonimizar:
        cabecalho = f"{ROTULO_ANONIMO}."
    else:
        ticker = linha["ticker"]
        nome = config.NOMES_EMPRESAS.get(ticker)
        quem = f"{ticker} ({nome})" if nome else ticker
        cabecalho = f"{quem}, pregão de {pd.Timestamp(linha['data']):%d/%m/%Y}."

    v = {c: linha[c] for c in ESTADO_COLUNAS}
    frases = [
        cabecalho,
        _frase_retorno("Retorno no último pregão", v["retorno_1d"]),
        _frase_retorno("Retorno em 5 pregões", v["retorno_5d"]),
        _frase_retorno("Retorno em 20 pregões", v["retorno_20d"]),
        _frase_media(v["distancia_media_20d"]),
        _frase_volatilidade(v["volatilidade_20d"]),
        _frase_volume(v["volume_relativo_20d"]),
    ]
    return " ".join(frases)


def gerar_estados(df: pd.DataFrame, anonimizar: bool = False) -> pd.DataFrame:
    """
    Estado de várias linhas de uma vez: devolve data, ticker e estado.
    É ESTA função que o Jev e o LLM devem chamar, para receberem o mesmo texto.
    """
    entrada = df[["data", "ticker", *ESTADO_COLUNAS]]  # o gabarito fica de fora já aqui
    saida = entrada[["data", "ticker"]].copy()
    saida["estado"] = [gerar_estado(linha, anonimizar) for _, linha in entrada.iterrows()]
    return saida.reset_index(drop=True)


def rodar() -> None:
    """Etapa extra 'python main.py estado': mostra exemplos do último pregão, nos dois modos."""
    from src import dados

    df = dados.carregar_indicadores()
    df = df[df["ticker"] != config.REFERENCIA]
    ultimo = df[df["data"] == df["data"].max()].head(3)

    print(f"Exemplos de estado ({len(ultimo)} ações do pregão de {df['data'].max():%d/%m/%Y}):\n")
    for _, linha in ultimo.iterrows():
        print("Normal:     " + gerar_estado(linha))
        print("Anonimizado: " + gerar_estado(linha, anonimizar=True))
        print()
