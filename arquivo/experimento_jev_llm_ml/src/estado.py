"""
estado.py
Transforma uma linha de indicadores no "estado" em texto enviado ao Jev e ao LLM.
Os dois recebem EXATAMENTE o mesmo texto e a mesma pergunta, gerados aqui (regra 4).

Por que texto e não tabela: o Jev é fraco em aritmética e em comparar datas. Então
todos os números já chegam calculados, em porcentagem, com a direção escrita por
extenso ("above", "below").

Idioma: o experimento usa INGLÊS (config.ESTADO_IDIOMA = "en"), porque a documentação
do Jev diz que inglês é o idioma principal dele e que outros idiomas têm acurácia menor
(ver Desvio 1 no PREREGISTRO.md). A versão em português continua disponível. Exemplo:

    en: PETR4 (Petrobras), trading session of October 8, 2026. Return in the last session: +2.1%.
        Return over the last 5 sessions: +11.2%. Return over the last 20 sessions: +12.9%.
        Price 10.5% above its 20-session average. Typical daily move over the last
        20 sessions: 2.5%. Volume 13% above normal.

    pt: PETR4 (Petrobras), pregão de 08/10/2026. Retorno no último pregão: +2,1%.
        Retorno em 5 pregões: +11,2%. Retorno em 20 pregões: +12,9%. Preço 10,5% acima
        da média de 20 pregões. Oscilação diária típica nos últimos 20 pregões: 2,5%.
        Volume 13% acima do normal.

Com anonimizar=True, ticker, nome da empresa e data somem (vira "Stock A" / "Ação A").
Isso serve para o teste de memorização: se o modelo decide diferente só por saber QUAL é
a ação ou QUANDO foi, ele está usando lembrança, e não os números.

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

MESES_EN = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]

# Todos os textos fixos, por idioma. A versão "en" é tradução direta da "pt": mesma informação.
TEXTOS = {
    "pt": {
        "pergunta": "A ação vai fechar o próximo pregão acima do fechamento de hoje?",
        "anonimo": "Ação A",
        "sessao": "pregão de {data}",
        "decimal": ",",
        "sem_dado": "sem dado",
        "ret_1d": "Retorno no último pregão",
        "ret_5d": "Retorno em 5 pregões",
        "ret_20d": "Retorno em 20 pregões",
        "media_vazia": "Distância da média de 20 pregões: sem dado.",
        "media_igual": "Preço igual à média de 20 pregões.",
        "media": "Preço {pct} {lado} da média de 20 pregões.",
        "acima": "acima", "abaixo": "abaixo",
        "vol": "Oscilação diária típica nos últimos 20 pregões",
        "volume_vazio": "Volume: sem dado.",
        "volume_vezes": "Volume {vezes} vezes o normal.",
        "volume_igual": "Volume igual ao normal.",
        "volume": "Volume {pct}% {lado} do normal.",
    },
    "en": {
        "pergunta": "Will the stock close the next trading session above today's close?",
        "anonimo": "Stock A",
        "sessao": "trading session of {data}",
        "decimal": ".",
        "sem_dado": "no data",
        "ret_1d": "Return in the last session",
        "ret_5d": "Return over the last 5 sessions",
        "ret_20d": "Return over the last 20 sessions",
        "media_vazia": "Distance from the 20-session average: no data.",
        "media_igual": "Price equal to its 20-session average.",
        "media": "Price {pct} {lado} its 20-session average.",
        "acima": "above", "abaixo": "below",
        "vol": "Typical daily move over the last 20 sessions",
        "volume_vazio": "Volume: no data.",
        "volume_vezes": "Volume {vezes} times normal.",
        "volume_igual": "Volume equal to normal.",
        "volume": "Volume {pct}% {lado} normal.",
    },
}

# A pergunta feita ao Jev e ao LLM, no idioma do experimento (registrada no PREREGISTRO.md).
PERGUNTA = TEXTOS[config.ESTADO_IDIOMA]["pergunta"]
ROTULO_ANONIMO = TEXTOS[config.ESTADO_IDIOMA]["anonimo"]


# ---------------------------------------------------------------------------
# Formatação de números
# ---------------------------------------------------------------------------

def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)))


def formatar_pct(fracao: float, casas: int = 1, sinal: bool = True, idioma: str = "pt") -> str:
    """0.038 -> '+3,8%' (pt) ou '+3.8%' (en). Com sinal=False: '3,8%'. Nunca mostra '-0,0%'."""
    valor = round(fracao * 100, casas)
    if valor == 0:
        valor = 0.0  # evita "-0,0"
    texto = f"{valor:{'+' if sinal and valor != 0 else ''}.{casas}f}".replace(".", TEXTOS[idioma]["decimal"])
    return f"{texto}%"


def _data(data, idioma: str) -> str:
    data = pd.Timestamp(data)
    if idioma == "en":
        return f"{MESES_EN[data.month - 1]} {data.day}, {data.year}"  # sem ambiguidade dia/mês
    return f"{data:%d/%m/%Y}"


def _frase_retorno(chave: str, fracao, t: dict, idioma: str) -> str:
    if _vazio(fracao):
        return f"{t[chave]}: {t['sem_dado']}."
    return f"{t[chave]}: {formatar_pct(fracao, idioma=idioma)}."


def _frase_media(distancia, t: dict, idioma: str) -> str:
    if _vazio(distancia):
        return t["media_vazia"]
    if round(distancia * 100, 1) == 0:
        return t["media_igual"]
    lado = t["acima"] if distancia > 0 else t["abaixo"]
    return t["media"].format(pct=formatar_pct(abs(distancia), sinal=False, idioma=idioma), lado=lado)


def _frase_volatilidade(vol, t: dict, idioma: str) -> str:
    if _vazio(vol):
        return f"{t['vol']}: {t['sem_dado']}."
    return f"{t['vol']}: {formatar_pct(vol, sinal=False, idioma=idioma)}."


def _frase_volume(relativo, t: dict) -> str:
    """volume_relativo_20d = volume de hoje / média de 20 pregões (1,0 = normal)."""
    if _vazio(relativo):
        return t["volume_vazio"]
    if relativo >= 2:
        return t["volume_vezes"].format(vezes=f"{relativo:.1f}".replace(".", t["decimal"]))
    diferenca = round((relativo - 1) * 100)
    if diferenca == 0:
        return t["volume_igual"]
    lado = t["acima"] if diferenca > 0 else t["abaixo"]
    return t["volume"].format(pct=abs(diferenca), lado=lado)


# ---------------------------------------------------------------------------
# O estado
# ---------------------------------------------------------------------------

def gerar_estado(linha: pd.Series, anonimizar: bool = False, idioma: str | None = None) -> str:
    """Monta o texto de UMA ação em UM dia. Só lê data, ticker e ESTADO_COLUNAS."""
    idioma = idioma or config.ESTADO_IDIOMA
    t = TEXTOS[idioma]
    if anonimizar:
        cabecalho = f"{t['anonimo']}."
    else:
        ticker = linha["ticker"]
        nome = config.NOMES_EMPRESAS.get(ticker)
        quem = f"{ticker} ({nome})" if nome else ticker
        cabecalho = f"{quem}, {t['sessao'].format(data=_data(linha['data'], idioma))}."

    v = {c: linha[c] for c in ESTADO_COLUNAS}
    frases = [
        cabecalho,
        _frase_retorno("ret_1d", v["retorno_1d"], t, idioma),
        _frase_retorno("ret_5d", v["retorno_5d"], t, idioma),
        _frase_retorno("ret_20d", v["retorno_20d"], t, idioma),
        _frase_media(v["distancia_media_20d"], t, idioma),
        _frase_volatilidade(v["volatilidade_20d"], t, idioma),
        _frase_volume(v["volume_relativo_20d"], t),
    ]
    return " ".join(frases)


def gerar_estados(df: pd.DataFrame, anonimizar: bool = False, idioma: str | None = None) -> pd.DataFrame:
    """
    Estado de várias linhas de uma vez: devolve data, ticker e estado.
    É ESTA função que o Jev e o LLM devem chamar, para receberem o mesmo texto.
    """
    entrada = df[["data", "ticker", *ESTADO_COLUNAS]]  # o gabarito fica de fora já aqui
    saida = entrada[["data", "ticker"]].copy()
    saida["estado"] = [gerar_estado(linha, anonimizar, idioma) for _, linha in entrada.iterrows()]
    return saida.reset_index(drop=True)


def rodar() -> None:
    """Etapa extra 'python main.py estado': mostra exemplos do último pregão, nos dois modos."""
    from src import dados

    df = dados.carregar_indicadores()
    df = df[df["ticker"] != config.REFERENCIA]
    ultimo = df[df["data"] == df["data"].max()].head(3)

    print(f"Idioma do experimento: {config.ESTADO_IDIOMA} | Pergunta: {PERGUNTA}")
    print(f"Exemplos de estado ({len(ultimo)} ações do pregão de {df['data'].max():%d/%m/%Y}):\n")
    for _, linha in ultimo.iterrows():
        print("Normal:      " + gerar_estado(linha))
        print("Anonimizado: " + gerar_estado(linha, anonimizar=True))
        print()
