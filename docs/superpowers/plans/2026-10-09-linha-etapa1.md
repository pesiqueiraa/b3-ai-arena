# Linha de produção de sinais (etapa 1): plano de implementação

> **Para agentes:** SUB-SKILL OBRIGATÓRIA: use superpowers:subagent-driven-development (recomendado) ou superpowers:executing-plans para implementar este plano tarefa por tarefa. Os passos usam caixas (`- [ ]`) para acompanhamento.

**Objetivo:** gerar sinais de compra e venda para swing trade nas ~100 ações mais líquidas da B3,
só com preço e volume, e testá-los num laboratório sem vazamento (2016–2024), com uma prova final
trancada em 2025+.

**Arquitetura:**
- **Coleta:** COTAHIST da B3 (todas as ações, inclusive as que deixaram de existir) define o
  universo de cada mês; o Yahoo fornece os preços ajustados.
- **Base:** as features de preço e o alvo ("superou o BOVA11 em 10 pregões?") formam uma tabela
  com colunas-chave `data, ticker`.
- **Modelo:** um combinador de ML é retreinado todo mês, com embargo de 10 pregões.
- **Saída:** os sinais viram uma carteira simulada com custos, comparada com o BOVA11 e com
  agentes aleatórios.

**Tecnologia:** Python 3.12, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, yfinance 1.7.0,
requests 2.34.2, pytest 9.1.1. Nenhuma dependência nova.

**Especificação:** `docs/superpowers/specs/2026-10-09-linha-etapa1-design.md`

## Restrições globais

- **Código e testes:** comentários e mensagens no terminal em português, pensados para um iniciante.
- **Dependências:** nenhuma nova; tudo já está em `requirements.txt`. Os testes rodam sem internet.
- **Rodar os testes:** `.venv/Scripts/python -m pytest` (na pasta do projeto, Windows).
- **Momento conhecido:** as features de D usam só dados até o fechamento de D.
- **Embargo:** um treino feito no dia T usa só linhas com data ≤ `calendario[pos(T) − 10]`.
- **Períodos:** `LAB_INICIO = "2016-01-01"`, `LAB_FIM = "2024-12-31"` e `RESERVA_INICIO = "2025-01-01"`.
  Nada no laboratório pode ler preço ou rótulo de 2025+.
- **Parâmetros:**
  - `HORIZONTE = 10`, `TAMANHO_UNIVERSO = 100`, `JANELA_LIQUIDEZ = 63`;
  - `CUSTO_POR_LADO = 0.001`, `N_ALEATORIOS = 1000`;
  - `GRADE_K = (3, 5, 10)`, `GRADE_P_MIN = (0.50, 0.55, 0.60)`;
  - `K_PADRAO = 5`, `P_MIN_PADRAO = 0.55`.
- **Prova final, aprovada só se cumprir os três:** excesso líquido total > 0,
  `bateu_aleatorios ≥ 0.90` e `acerto_compras > 0.50`.
- **Constantes:** toda constante vem de `config.py`. Dentro das funções, leia `config.X` na hora
  da chamada (não como valor padrão de argumento) quando o teste precisar trocá-la por
  `monkeypatch`: `RESERVA_INICIO`, `N_ALEATORIOS`, `GRADE_K`, `GRADE_P_MIN`, caminhos de arquivo.
- **Commits:** toda mensagem termina com a linha
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Foco da revisão

1. **Arquivo anual da B3 do ano corrente atrasado ou indisponível:** o calendário é a união das
   datas da B3 com as do BOVA11 no Yahoo, e o `atualizar` usa a cópia local se o download falhar.
   Testado na Tarefa 7.
2. **Yahoo devolve datas repetidas ou o dia de hoje incompleto:** as duas coisas são removidas
   antes de qualquer cálculo. Testado na Tarefa 3.
3. **Dia com menos de 2k ações ou com probabilidades empatadas:** uma ação nunca aparece como
   compra e venda no mesmo dia. Testado na Tarefa 9.
4. **Ação do universo sem nenhum preço no Yahoo** (deixou de existir): a base é montada sem
   quebrar e a cobertura mostra a falta. Testado na Tarefa 7.
5. **Rodar `laboratorio`, `sinais` ou `prova-final` antes de `atualizar`:** mensagem clara
   mandando rodar `atualizar`, e nada de traceback. Testado na Tarefa 7.

## Mapa de arquivos

```
config.py                       (novo) todas as constantes
main.py                         (novo) atualizar | laboratorio | sinais | prova-final
pytest.ini                      (novo) pythonpath=. testpaths=tests
src/__init__.py
src/coleta/__init__.py
src/coleta/b3.py                COTAHIST: download, leitura, tickers com troca de nome
src/coleta/yahoo.py             preços ajustados, remoção de dia incompleto e datas repetidas
src/universo.py                 universo mensal, membros por dia, cobertura
src/alvo.py                     alvo (excesso sobre o BOVA11 em 10 pregões) e retornos diários
src/features/__init__.py
src/features/preco.py           features de preço e ranks dentro do universo
src/base.py                     junta tudo: Base(tabela, calendario, retornos)
src/atualizar.py                comando "atualizar"
src/modelos/__init__.py
src/modelos/combinador.py       regressão logística e gradient boosting
src/sinais.py                   probabilidades -> compra / venda
src/hoje.py                     comando "sinais" (próximo pregão)
src/laboratorio/__init__.py
src/laboratorio/walk_forward.py retreino mensal com embargo
src/laboratorio/carteira.py     simulação da carteira de compras com custos
src/laboratorio/metricas.py     Brier, referência, acertos, resumo, agentes aleatórios
src/laboratorio/executar.py     comando "laboratorio" (grade, tentativas, escolha, relatório)
src/laboratorio/prova_final.py  comando "prova-final" (uma vez só)
tests/conftest.py               dados sintéticos (passeio aleatório)
tests/test_*.py                 um arquivo por módulo
arquivo/experimento_jev_llm_ml/ o experimento antigo inteiro
```

---

### Tarefa 1: Arquivar o experimento e criar o esqueleto novo

**Arquivos:**
- Mover (git mv): `main.py`, `config.py`, `pytest.ini`, `PREREGISTRO.md`, `src/` e `tests/` para
  `arquivo/experimento_jev_llm_ml/`; `resultados/decisoes/`, `resultados/extra/`,
  `resultados/resultado.csv` e `resultados/comparacoes.csv` para
  `arquivo/experimento_jev_llm_ml/resultados/`
- Mover (git mv): `resultados/cache/` para `arquivo/experimento_jev_llm_ml/resultados/cache/` (o `.gitkeep` dela é versionado)
- Criar: `arquivo/experimento_jev_llm_ml/README.md`, `config.py`, `pytest.ini`, `src/__init__.py`,
  `src/coleta/__init__.py`, `src/features/__init__.py`, `src/modelos/__init__.py`,
  `src/laboratorio/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`
- Modificar: `.gitignore`, `arquivo/experimento_jev_llm_ml/PREREGISTRO.md` (Desvio 2)

**Interfaces:**
- Produz: o módulo `config` (constantes abaixo); em `tests/conftest.py`,
  `gerar_precos(n_acoes=20, inicio="2015-01-01", fim="2017-12-31", semente=7) -> tuple[pd.DataFrame, pd.DatetimeIndex]`
  (colunas `data, ticker, fechamento_ajustado, volume`, com BOVA11),
  `universo_completo(precos, calendario) -> pd.DataFrame` (colunas `mes, data_inicio, ticker, posicao, volume_medio`)
  e as fixtures `sinteticos` e `base_sintetica` (esta última usa `src.base.montar_base`, criado na Tarefa 7).

- [ ] **Passo 1: Mover o experimento para o arquivo**

```bash
cd "C:/Users/pesiq/OneDrive/Desktop/Ideias/b3-ai-arena"
mkdir -p arquivo/experimento_jev_llm_ml/resultados
git mv main.py config.py pytest.ini PREREGISTRO.md src tests arquivo/experimento_jev_llm_ml/
git mv resultados/decisoes resultados/extra resultados/resultado.csv resultados/comparacoes.csv arquivo/experimento_jev_llm_ml/resultados/
# a pasta cache tem um .gitkeep versionado; o git mv leva junto as respostas (fora do git) que estão nela
git mv resultados/cache arquivo/experimento_jev_llm_ml/resultados/cache
git status --short | head -20   # conferir: só renomeações (R) para arquivo/
```

- [ ] **Passo 2: Registrar o Desvio 2 no pré-registro arquivado**

Acrescentar ao final de `arquivo/experimento_jev_llm_ml/PREREGISTRO.md`:

```markdown

### Desvio 2 (09/10/2026): experimento encerrado

O experimento foi encerrado por decisão do autor em 09/10/2026, com 17 pregões avaliados de 250
previstos. **Não há conclusões**: todos os resultados ficaram "parciais", e os intervalos de
confiança de todas as comparações cruzaram o zero. O projeto passou a ser uma linha de produção
em que ML, LLM e Jev trabalham juntos (ver `docs/superpowers/specs/2026-10-09-linha-etapa1-design.md`).
O código, as decisões e os resultados ficam preservados nesta pasta.
```

- [ ] **Passo 3: README do arquivo**

Criar `arquivo/experimento_jev_llm_ml/README.md`:

```markdown
# Experimento encerrado: Jev vs LLM vs ML (15/09 a 09/10/2026)

Este é o primeiro formato do projeto: comparar, nas mesmas condições, três modelos decidindo todo
dia se uma ação da B3 subiria no pregão seguinte. Ele foi **encerrado em 09/10/2026 com 17
pregões e sem conclusões** (ver "Desvio 2" no `PREREGISTRO.md`).

O que fica aqui:
- `PREREGISTRO.md`: o plano de análise, os adendos e os desvios;
- `resultados/decisoes/`: as decisões de cada modelo, versionadas no git;
- `src/` e `tests/`: o código do experimento.

O código continua aqui como registro. Para rodar, entre nesta pasta, recrie a pasta `dados/`
com `python main.py baixar` e use os comandos do `main.py`.

As peças `src/modelos/jev.py`, `src/modelos/llm.py`, `src/modelos/api_comum.py` e
`src/estado.py` serão reaproveitadas na etapa 3 da linha de produção.
```

- [ ] **Passo 4: `.gitignore`**

Acrescentar ao `.gitignore`:

```
# Respostas de API guardadas pelo experimento arquivado (podem conter dados das contas)
arquivo/**/resultados/cache/
```

- [ ] **Passo 5: `config.py` novo**

```python
"""
config.py
Todas as configurações da linha de produção de sinais num lugar só.
Se quiser mudar uma data, um custo ou um parâmetro, é aqui.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pastas e arquivos
# ---------------------------------------------------------------------------

PASTA_DADOS = RAIZ / "dados"                         # tudo aqui fica fora do git
PASTA_B3 = PASTA_DADOS / "b3"                        # arquivos COTAHIST baixados
PASTA_PROCESSADOS = PASTA_DADOS / "processados"
ARQUIVO_B3 = PASTA_PROCESSADOS / "b3_acoes.csv.gz"
ARQUIVO_PRECOS = PASTA_PROCESSADOS / "precos_ajustados.csv.gz"
ARQUIVO_CALENDARIO = PASTA_PROCESSADOS / "calendario.csv"

PASTA_RESULTADOS = RAIZ / "resultados"               # vai para o git
PASTA_UNIVERSO = PASTA_RESULTADOS / "universo"
ARQUIVO_UNIVERSO = PASTA_UNIVERSO / "universo_mensal.csv"
ARQUIVO_COBERTURA = PASTA_UNIVERSO / "cobertura.csv"
ARQUIVO_TROCA_DE_NOME = PASTA_UNIVERSO / "tickers_com_troca_de_nome.csv"
PASTA_LABORATORIO = PASTA_RESULTADOS / "laboratorio"
ARQUIVO_TENTATIVAS = PASTA_LABORATORIO / "tentativas.csv"
ARQUIVO_ESCOLHA = PASTA_LABORATORIO / "escolha.json"
ARQUIVO_RELATORIO_LAB = PASTA_LABORATORIO / "relatorio.md"
ARQUIVO_PROVA_FINAL = PASTA_RESULTADOS / "prova_final.json"
ARQUIVO_RELATORIO_PROVA = PASTA_RESULTADOS / "prova_final.md"

# ---------------------------------------------------------------------------
# Datas
# ---------------------------------------------------------------------------

DATA_DOWNLOAD_INICIO = "2015-01-01"   # 120 pregões de histórico + 3 meses de liquidez antes de 2016
LAB_INICIO = "2016-01-01"             # laboratório: onde ajustamos tudo
LAB_FIM = "2024-12-31"
RESERVA_INICIO = "2025-01-01"         # reserva TRANCADA: só a prova final olha daqui em diante

# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------

URL_COTAHIST = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"

# ---------------------------------------------------------------------------
# Universo (aproximação do IBrX-100 sem viés de sobrevivência)
# ---------------------------------------------------------------------------

TAMANHO_UNIVERSO = 100
JANELA_LIQUIDEZ = 63                  # pregões usados para medir o volume médio (~3 meses)
ESPECIES_ACEITAS = ("ON", "PN", "PNA", "PNB", "PNC", "PND", "UNT")
REFERENCIA = "BOVA11"                 # ETF do Ibovespa: a régua do alvo

# ---------------------------------------------------------------------------
# Alvo, modelo e sinais
# ---------------------------------------------------------------------------

HORIZONTE = 10                        # "vai superar o Ibovespa nos próximos 10 pregões?"
TIPOS_MODELO = ("logistica", "boosting")
SEMENTE = 42

K_PADRAO = 5                          # quantas compras / vendas por dia, no máximo
P_MIN_PADRAO = 0.55                   # probabilidade mínima para um sinal de compra
GRADE_K = (3, 5, 10)                  # grade FIXA e pequena do laboratório (ver especificação 8.2)
GRADE_P_MIN = (0.50, 0.55, 0.60)

# ---------------------------------------------------------------------------
# Carteira simulada e teste contra a sorte
# ---------------------------------------------------------------------------

CUSTO_POR_LADO = 0.001                # 0,05% de custos + 0,05% de margem para slippage
N_ALEATORIOS = 1000

# ---------------------------------------------------------------------------
# Prova final: critérios definidos ANTES de qualquer resultado
# ---------------------------------------------------------------------------

CRITERIO_EXCESSO_MINIMO = 0.0         # excesso líquido total sobre o BOVA11 precisa ser > isto
CRITERIO_ALEATORIOS = 0.90            # superar pelo menos 90% dos agentes aleatórios
CRITERIO_ACERTO_COMPRAS = 0.50        # mais da metade das compras supera o BOVA11
```

- [ ] **Passo 6: `pytest.ini` e pacotes vazios**

`pytest.ini`:

```ini
[pytest]
# Permite "import config" e "from src import ..." de dentro dos testes.
pythonpath = .
testpaths = tests
```

Criar `src/__init__.py`, `src/coleta/__init__.py`, `src/features/__init__.py`,
`src/modelos/__init__.py`, `src/laboratorio/__init__.py` e `tests/__init__.py`, cada um com uma
única linha de docstring, por exemplo `"""Linha de produção de sinais (B3, swing trade)."""`.

- [ ] **Passo 7: Dados sintéticos dos testes**

`tests/conftest.py`:

```python
"""
Dados inventados para os testes, sem internet.
Os preços são um passeio aleatório: o futuro é imprevisível por construção, então nenhum modelo
honesto deveria achar padrão neles.
"""

import numpy as np
import pandas as pd
import pytest

import config


def gerar_precos(n_acoes: int = 20, inicio: str = "2015-01-01", fim: str = "2017-12-31",
                 semente: int = 7) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Devolve (precos, calendario). Colunas: data, ticker, fechamento_ajustado, volume (com BOVA11)."""
    rng = np.random.default_rng(semente)
    calendario = pd.DatetimeIndex(pd.bdate_range(inicio, fim))
    tickers = [f"AC{i:02d}3" for i in range(n_acoes)] + [config.REFERENCIA]
    partes = []
    for ticker in tickers:
        preco = 20 * np.exp(np.cumsum(rng.normal(0, 0.015, len(calendario))))
        partes.append(pd.DataFrame({
            "data": calendario,
            "ticker": ticker,
            "fechamento_ajustado": preco,
            "volume": rng.lognormal(15, 0.4, len(calendario)).round(),
        }))
    return pd.concat(partes, ignore_index=True), calendario


def universo_completo(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Universo em que todas as ações (menos o BOVA11) estão na lista em todos os meses."""
    primeiros = pd.Series(calendario).groupby(calendario.to_period("M")).first()
    acoes = sorted(t for t in precos["ticker"].unique() if t != config.REFERENCIA)
    linhas = [
        {"mes": str(mes), "data_inicio": dia, "ticker": t, "posicao": i + 1, "volume_medio": 1.0}
        for mes, dia in primeiros.items() for i, t in enumerate(acoes)
    ]
    return pd.DataFrame(linhas)


@pytest.fixture(scope="session")
def sinteticos():
    return gerar_precos()


@pytest.fixture(scope="session")
def base_sintetica(sinteticos):
    from src.base import montar_base  # criado na Tarefa 7

    precos, calendario = sinteticos
    return montar_base(precos, universo_completo(precos, calendario), calendario)
```

- [ ] **Passo 8: Teste das regras de configuração**

`tests/test_config.py`:

```python
"""As regras da especificação que dependem só da configuração."""

import pandas as pd

import config


def test_laboratorio_termina_antes_da_reserva():
    assert pd.Timestamp(config.LAB_INICIO) < pd.Timestamp(config.LAB_FIM) < pd.Timestamp(config.RESERVA_INICIO)


def test_download_comeca_antes_do_laboratorio():
    assert pd.Timestamp(config.DATA_DOWNLOAD_INICIO) < pd.Timestamp(config.LAB_INICIO)


def test_grade_tem_no_maximo_9_combinacoes_por_modelo():
    assert len(config.GRADE_K) * len(config.GRADE_P_MIN) <= 9
    assert len(config.TIPOS_MODELO) == 2


def test_criterios_da_prova_final():
    assert config.CRITERIO_EXCESSO_MINIMO == 0.0
    assert config.CRITERIO_ALEATORIOS == 0.90
    assert config.CRITERIO_ACERTO_COMPRAS == 0.50
```

- [ ] **Passo 9: Rodar os testes**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `4 passed`. O `conftest.py` importa sem erro, porque `src.base` só é importado dentro da fixture.

- [ ] **Passo 10: Commit**

```bash
git add -A
git commit -m "Arquiva o experimento Jev/LLM/ML e cria o esqueleto da linha de produção

Experimento encerrado (Desvio 2) e movido inteiro para arquivo/. Novo
config.py, pytest.ini e dados sintéticos para os testes da etapa 1.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 2: Leitura do COTAHIST da B3

**Arquivos:**
- Criar: `src/coleta/b3.py`
- Teste: `tests/test_b3.py`

**Interfaces:**
- Consome: `config.PASTA_B3`, `config.URL_COTAHIST`, `config.ESPECIES_ACEITAS`.
- Produz:
  - `ler_linhas(linhas: Iterable[str]) -> pd.DataFrame`, com colunas
    `data (datetime), ticker, nome, especie, fechamento (float, R$), volume_financeiro (float, R$)`;
  - `caminho_do_ano(ano: int) -> Path` e `baixar_ano(ano: int, forcar: bool = False) -> Path`;
  - `ler_ano(caminho: Path) -> pd.DataFrame`;
  - `tickers_com_troca_de_nome(b3: pd.DataFrame) -> pd.DataFrame` (colunas `ticker, nomes`).

- [ ] **Passo 1: Escrever os testes**

`tests/test_b3.py`:

```python
"""Leitura do COTAHIST (layout oficial de posições fixas da B3)."""

import zipfile

import pandas as pd
import pytest

from src.coleta import b3


def linha(tipreg="01", data="20240102", codbdi="02", ticker="PETR4", tpmerc="010",
          nome="PETROBRAS", especi="PN      N2", preult=3812, voltot=150000000) -> str:
    """Monta uma linha de 245 caracteres no layout do COTAHIST (posições começando em 1)."""
    caracteres = [" "] * 245

    def texto(inicio, fim, valor):
        tamanho = fim - inicio + 1
        caracteres[inicio - 1:fim] = list(str(valor).ljust(tamanho)[:tamanho])

    def numero(inicio, fim, valor):
        tamanho = fim - inicio + 1
        caracteres[inicio - 1:fim] = list(str(valor).zfill(tamanho))

    texto(1, 2, tipreg)
    texto(3, 10, data)
    texto(11, 12, codbdi)
    texto(13, 24, ticker)
    texto(25, 27, tpmerc)
    texto(28, 39, nome)
    texto(40, 49, especi)
    numero(109, 121, preult)
    numero(171, 188, voltot)
    return "".join(caracteres)


def test_le_os_campos_certos():
    tabela = b3.ler_linhas([linha()])
    assert tabela.to_dict("records") == [{
        "data": pd.Timestamp("2024-01-02"), "ticker": "PETR4", "nome": "PETROBRAS",
        "especie": "PN", "fechamento": 38.12, "volume_financeiro": 1500000.0,
    }]


@pytest.mark.parametrize("mudanca", [
    {"tipreg": "00"},              # cabeçalho do arquivo
    {"codbdi": "12"},              # fundo imobiliário
    {"codbdi": "14"},              # ETF
    {"tpmerc": "070"},             # opção, não é mercado à vista
    {"especi": "DRN     ED"},      # BDR
    {"especi": "CI"},              # fundo
])
def test_descarta_o_que_nao_e_acao_a_vista(mudanca):
    assert b3.ler_linhas([linha(**mudanca)]).empty


def test_aceita_units_e_preferenciais_de_classe():
    tabela = b3.ler_linhas([
        linha(ticker="TAEE11", especi="UNT     N2"),
        linha(ticker="USIM5", especi="PNA     N1"),
        linha(ticker="VALE3", especi="ON  EJ  NM"),   # marcação de ex-direito continua sendo ON
    ])
    assert list(tabela["especie"]) == ["UNT", "PNA", "ON"]


def test_linha_curta_e_ignorada():
    assert b3.ler_linhas(["01" + "x" * 50]).empty


def test_tabela_vazia_tem_as_colunas_certas():
    assert list(b3.ler_linhas([]).columns) == ["data", "ticker", "nome", "especie", "fechamento", "volume_financeiro"]


def test_le_arquivo_zip(tmp_path):
    caminho = tmp_path / "COTAHIST_A2024.ZIP"
    with zipfile.ZipFile(caminho, "w") as arquivo:
        arquivo.writestr("COTAHIST_A2024.TXT", "00COTAHIST.2024\n" + linha() + "\n99COTAHIST.2024\n")
    assert len(b3.ler_ano(caminho)) == 1


def test_detecta_ticker_com_troca_de_nome():
    tabela = pd.DataFrame({
        "ticker": ["ABCD3", "ABCD3", "EFGH3"],
        "nome": ["VELHA SA", "NOVA SA", "OUTRA"],
    })
    assert b3.tickers_com_troca_de_nome(tabela).to_dict("records") == [
        {"ticker": "ABCD3", "nomes": "NOVA SA | VELHA SA"}
    ]
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_b3.py -q`
Expected: FAIL com `ImportError` / `cannot import name 'b3'`.

- [ ] **Passo 3: Implementar**

`src/coleta/b3.py`:

```python
"""
b3.py
Séries históricas OFICIAIS da B3 (arquivos COTAHIST). Elas têm todas as ações negociadas,
inclusive as que deixaram de existir. Uso: montar o universo de cada mês sem viés de
sobrevivência. Os preços daqui NÃO são ajustados por dividendos; os retornos vêm do Yahoo.

Layout oficial (posições fixas, 245 caracteres por linha). Campos usados (posições a partir de 1):
    TIPREG 1-2 ("01" = cotação), DATA 3-10, CODBDI 11-12 ("02" = lote padrão),
    CODNEG 13-24 (ticker), TPMERC 25-27 ("010" = à vista), NOMRES 28-39 (nome),
    ESPECI 40-49 (ON, PN, UNT...), PREULT 109-121 (fechamento, 2 decimais),
    VOLTOT 171-188 (volume financeiro, 2 decimais).
"""

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

import config

COLUNAS = ["data", "ticker", "nome", "especie", "fechamento", "volume_financeiro"]


def _especie(especi: str) -> str | None:
    """'PN      N2' -> 'PN'. Devolve None se não for uma espécie aceita (BDR, fundo...)."""
    partes = especi.split()
    if partes and partes[0] in config.ESPECIES_ACEITAS:
        return partes[0]
    return None


def ler_linhas(linhas: Iterable[str]) -> pd.DataFrame:
    """Converte linhas do COTAHIST numa tabela, só com ações à vista no lote padrão."""
    registros = []
    for linha in linhas:
        if len(linha) < 188 or linha[0:2] != "01":
            continue
        if linha[10:12] != "02" or linha[24:27] != "010":
            continue
        especie = _especie(linha[39:49])
        if especie is None:
            continue
        registros.append({
            "data": linha[2:10],
            "ticker": linha[12:24].strip(),
            "nome": linha[27:39].strip(),
            "especie": especie,
            "fechamento": int(linha[108:121]) / 100,
            "volume_financeiro": int(linha[170:188]) / 100,
        })
    tabela = pd.DataFrame(registros, columns=COLUNAS)
    tabela["data"] = pd.to_datetime(tabela["data"], format="%Y%m%d")
    return tabela


def caminho_do_ano(ano: int) -> Path:
    return config.PASTA_B3 / f"COTAHIST_A{ano}.ZIP"


def baixar_ano(ano: int, forcar: bool = False) -> Path:
    """Baixa o arquivo do ano (se ainda não existir, ou se forcar=True) e devolve o caminho."""
    import requests

    destino = caminho_do_ano(ano)
    if destino.exists() and not forcar:
        return destino
    config.PASTA_B3.mkdir(parents=True, exist_ok=True)
    resposta = requests.get(config.URL_COTAHIST.format(ano=ano), timeout=600)
    resposta.raise_for_status()
    destino.write_bytes(resposta.content)
    return destino


def ler_ano(caminho: Path) -> pd.DataFrame:
    """Lê um arquivo COTAHIST_A{ano}.ZIP (texto em latin-1 dentro do zip)."""
    with zipfile.ZipFile(caminho) as arquivo:
        with arquivo.open(arquivo.namelist()[0]) as bruto:
            return ler_linhas(io.TextIOWrapper(bruto, encoding="latin-1"))


def tickers_com_troca_de_nome(b3: pd.DataFrame) -> pd.DataFrame:
    """
    Tickers que aparecem com mais de um nome de empresa. Pode ser só uma troca de nome,
    ou o ticker reaproveitado por outra empresa. A etapa 1 só relata os casos.
    """
    nomes = b3.groupby("ticker")["nome"].apply(lambda s: sorted(s.unique()))
    nomes = nomes[nomes.str.len() > 1]
    return pd.DataFrame({"ticker": nomes.index, "nomes": nomes.str.join(" | ").values}).reset_index(drop=True)
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_b3.py -q`
Expected: `12 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/coleta/b3.py tests/test_b3.py
git commit -m "Leitura do COTAHIST da B3 (universo sem viés de sobrevivência)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 3: Preços ajustados do Yahoo

**Arquivos:**
- Criar: `src/coleta/yahoo.py`
- Teste: `tests/test_yahoo.py`

**Interfaces:**
- Produz:
  - `baixar_ticker(ticker: str, inicio: str) -> pd.DataFrame`, com colunas
    `data, ticker, abertura, maxima, minima, fechamento_ajustado, volume`;
  - `remover_pregoes_incompletos(precos) -> pd.DataFrame`;
  - `remover_datas_repetidas(precos) -> pd.DataFrame`;
  - `baixar_varios(tickers: list[str], inicio: str, pausa: float = 0.3, baixar=baixar_ticker) -> tuple[pd.DataFrame, list[str]]`,
    que devolve os preços (colunas `data, ticker, fechamento_ajustado, volume`) e os tickers que falharam.

- [ ] **Passo 1: Escrever os testes**

`tests/test_yahoo.py`:

```python
"""Preços ajustados do Yahoo: limpeza e download de vários tickers (com um 'baixar' falso)."""

import pandas as pd

from src.coleta import yahoo


def _tabela(ticker, datas, abertura=10.0):
    return pd.DataFrame({
        "data": pd.to_datetime(datas), "ticker": ticker, "abertura": abertura, "maxima": abertura,
        "minima": abertura, "fechamento_ajustado": 10.0, "volume": 1000.0,
    })


def test_remove_pregao_incompleto_mas_mantem_dia_sem_negocio():
    precos = pd.concat([
        _tabela("AAAA3", ["2026-10-07"]),
        _tabela("AAAA3", ["2026-10-08"], abertura=0.0),                       # dia de hoje pela metade
        _tabela("BBBB3", ["2024-04-26"]).assign(volume=0.0),                  # sem negócio, preço ok
    ])
    limpo = yahoo.remover_pregoes_incompletos(precos)
    assert list(limpo["data"].dt.strftime("%Y-%m-%d")) == ["2026-10-07", "2024-04-26"]


def test_remove_datas_repetidas_ficando_com_a_ultima():
    precos = pd.concat([_tabela("AAAA3", ["2024-01-02"]), _tabela("AAAA3", ["2024-01-02"]).assign(fechamento_ajustado=11.0)])
    limpo = yahoo.remover_datas_repetidas(precos)
    assert len(limpo) == 1 and limpo["fechamento_ajustado"].iloc[0] == 11.0


def test_baixar_varios_junta_e_lista_falhas():
    def baixar_falso(ticker, inicio):
        if ticker == "ERRO3":
            raise ConnectionError("sem internet")
        if ticker == "VAZI3":
            return pd.DataFrame()
        return _tabela(ticker, ["2024-01-02", "2024-01-03"])

    precos, falharam = yahoo.baixar_varios(["AAAA3", "ERRO3", "VAZI3", "BBBB3"], "2024-01-01",
                                           pausa=0, baixar=baixar_falso)
    assert sorted(falharam) == ["ERRO3", "VAZI3"]
    assert list(precos.columns) == ["data", "ticker", "fechamento_ajustado", "volume"]
    assert sorted(precos["ticker"].unique()) == ["AAAA3", "BBBB3"]


def test_baixar_varios_sem_nada_devolve_tabela_vazia():
    precos, falharam = yahoo.baixar_varios(["ERRO3"], "2024-01-01", pausa=0,
                                           baixar=lambda t, i: pd.DataFrame())
    assert precos.empty and falharam == ["ERRO3"]
    assert list(precos.columns) == ["data", "ticker", "fechamento_ajustado", "volume"]
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_yahoo.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/coleta/yahoo.py`:

```python
"""
yahoo.py
Preços AJUSTADOS por dividendos e desdobramentos (Yahoo Finance). São usados nos retornos, nas
features e no alvo. No Yahoo, ações da B3 têm ".SA" no final (o código adiciona sozinho).
"""

import time

import pandas as pd

COLUNAS = ["data", "ticker", "fechamento_ajustado", "volume"]


def baixar_ticker(ticker: str, inicio: str) -> pd.DataFrame:
    """Baixa um ticker. Devolve tabela vazia se o Yahoo não tiver dados (ex.: ação que deixou de existir)."""
    import yfinance as yf  # importado aqui para os testes rodarem sem internet

    bruto = yf.download(f"{ticker}.SA", start=inicio, auto_adjust=False, progress=False)
    if bruto is None or bruto.empty:
        return pd.DataFrame()
    if isinstance(bruto.columns, pd.MultiIndex):           # versões novas: ("Close", "PETR4.SA")
        bruto.columns = bruto.columns.get_level_values(0)
    if "Adj Close" not in bruto.columns:
        bruto["Adj Close"] = bruto["Close"]
    tabela = bruto.rename(columns={
        "Open": "abertura", "High": "maxima", "Low": "minima",
        "Adj Close": "fechamento_ajustado", "Volume": "volume",
    })[["abertura", "maxima", "minima", "fechamento_ajustado", "volume"]].copy()
    tabela.index.name = "data"
    tabela = tabela.reset_index()
    tabela["data"] = pd.to_datetime(tabela["data"])
    if tabela["data"].dt.tz is not None:
        tabela["data"] = tabela["data"].dt.tz_localize(None)
    tabela["data"] = tabela["data"].dt.normalize()
    tabela.insert(1, "ticker", ticker)
    return tabela.dropna(subset=["fechamento_ajustado"])


def remover_pregoes_incompletos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Quando o download acontece antes de o Yahoo consolidar o dia, a linha de hoje vem com abertura,
    máxima e mínima iguais a zero. Essas linhas são removidas. Dias antigos com volume zero mas com
    preços preenchidos (dia sem negócio) ficam.
    """
    incompleta = (precos["abertura"] == 0) & (precos["maxima"] == 0) & (precos["minima"] == 0)
    if incompleta.any():
        datas = sorted(precos.loc[incompleta, "data"].dt.strftime("%d/%m/%Y").unique())
        print(f"Aviso: {int(incompleta.sum())} linha(s) de pregão incompleto removidas "
              f"(datas: {', '.join(datas)}). Baixe de novo depois do fechamento.")
    return precos[~incompleta].copy()


def remover_datas_repetidas(precos: pd.DataFrame) -> pd.DataFrame:
    """Se o Yahoo devolver a mesma data duas vezes para um ticker, fica a última."""
    return precos.drop_duplicates(subset=["data", "ticker"], keep="last").copy()


def baixar_varios(tickers: list[str], inicio: str, pausa: float = 0.3,
                  baixar=baixar_ticker) -> tuple[pd.DataFrame, list[str]]:
    """Baixa vários tickers, limpa e junta. Devolve (precos, tickers_que_falharam)."""
    tabelas, falharam = [], []
    for numero, ticker in enumerate(tickers, start=1):
        print(f"  Yahoo {numero}/{len(tickers)}: {ticker:<8}", end=" ", flush=True)
        try:
            tabela = baixar(ticker, inicio)
        except Exception as erro:  # sem internet, ticker sem dados, limite do Yahoo...
            print(f"-> erro: {erro}")
            falharam.append(ticker)
            continue
        if tabela.empty:
            print("-> sem dados")
            falharam.append(ticker)
            continue
        print(f"-> {len(tabela)} pregões")
        tabelas.append(tabela)
        if pausa:
            time.sleep(pausa)  # pausa curta para o Yahoo não bloquear

    if not tabelas:
        return pd.DataFrame(columns=COLUNAS), falharam
    precos = pd.concat(tabelas, ignore_index=True)
    precos = remover_datas_repetidas(remover_pregoes_incompletos(precos))
    return precos[COLUNAS].sort_values(["ticker", "data"]).reset_index(drop=True), falharam
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_yahoo.py -q`
Expected: `4 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/coleta/yahoo.py tests/test_yahoo.py
git commit -m "Preços ajustados do Yahoo com limpeza de dias incompletos e datas repetidas

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 4: Universo mensal

**Arquivos:**
- Criar: `src/universo.py`
- Teste: `tests/test_universo.py`

**Interfaces:**
- Consome: a tabela da B3 de `b3.ler_ano` (colunas `data, ticker, volume_financeiro`).
- Produz:
  - `montar_universo(b3, tamanho=config.TAMANHO_UNIVERSO, janela=config.JANELA_LIQUIDEZ) -> pd.DataFrame`,
    com colunas `mes (str "AAAA-MM"), data_inicio (datetime), ticker, posicao (1..n), volume_medio`;
  - `membros_por_dia(universo, calendario: pd.DatetimeIndex) -> pd.DataFrame` (colunas `data, ticker`);
  - `calcular_cobertura(universo, precos) -> pd.DataFrame` (colunas `mes, no_universo, com_preco, cobertura`).

- [ ] **Passo 1: Escrever os testes**

`tests/test_universo.py`:

```python
"""Universo mensal: as mais líquidas pelo volume PASSADO, uma ação por empresa."""

import pandas as pd

from src import universo


def b3_falso(volumes: dict[str, float], n_dias: int = 70, inicio: str = "2020-01-01") -> pd.DataFrame:
    dias = pd.bdate_range(inicio, periods=n_dias)
    return pd.DataFrame([
        {"data": d, "ticker": t, "volume_financeiro": v} for d in dias for t, v in volumes.items()
    ])


def test_ordena_pelo_volume_e_limita_o_tamanho():
    b3 = b3_falso({"AAAA3": 5e6, "BBBB3": 4e6, "CCCC3": 3e6, "DDDD3": 2e6, "EEEE3": 1e6})
    u = universo.montar_universo(b3, tamanho=3, janela=20)
    primeiro = u[u["mes"] == u["mes"].min()]
    assert list(primeiro["ticker"]) == ["AAAA3", "BBBB3", "CCCC3"]
    assert list(primeiro["posicao"]) == [1, 2, 3]


def test_uma_acao_por_empresa_fica_a_mais_liquida():
    b3 = b3_falso({"PETR3": 1e6, "PETR4": 5e6, "VALE3": 3e6})
    u = universo.montar_universo(b3, tamanho=10, janela=20)
    assert set(u["ticker"]) == {"PETR4", "VALE3"}


def test_volume_futuro_nao_muda_o_mes():
    b3 = b3_falso({"AAAA3": 5e6, "BBBB3": 4e6, "CCCC3": 3e6})
    antes = universo.montar_universo(b3, tamanho=2, janela=20)
    mes = sorted(antes["mes"].unique())[0]
    inicio_do_mes = antes.loc[antes["mes"] == mes, "data_inicio"].iloc[0]
    alterado = b3.copy()
    alterado.loc[(alterado["data"] >= inicio_do_mes) & (alterado["ticker"] == "CCCC3"), "volume_financeiro"] = 1e12
    depois = universo.montar_universo(alterado, tamanho=2, janela=20)
    pd.testing.assert_frame_equal(antes[antes["mes"] == mes].reset_index(drop=True),
                                  depois[depois["mes"] == mes].reset_index(drop=True))


def test_dia_sem_negocio_conta_como_zero_na_media():
    dias = pd.bdate_range("2020-01-01", periods=70)
    linhas = [{"data": d, "ticker": "CONS3", "volume_financeiro": 1e6} for d in dias]
    linhas += [{"data": d, "ticker": "RARA3", "volume_financeiro": 3e6} for d in dias[::4]]  # 1 a cada 4 dias
    u = universo.montar_universo(pd.DataFrame(linhas), tamanho=1, janela=20)
    assert set(u["ticker"]) == {"CONS3"}  # média do RARA3 ~ 0,75 mi < 1 mi


def test_meses_sem_historico_suficiente_ficam_de_fora():
    b3 = b3_falso({"AAAA3": 1e6}, n_dias=70, inicio="2020-01-01")
    u = universo.montar_universo(b3, tamanho=5, janela=63)
    assert u["data_inicio"].min() >= pd.bdate_range("2020-01-01", periods=70)[63]


def test_membros_por_dia_usa_a_lista_do_proprio_mes():
    u = pd.DataFrame({"mes": ["2020-01", "2020-02"], "data_inicio": pd.to_datetime(["2020-01-02", "2020-02-03"]),
                      "ticker": ["AAAA3", "BBBB3"], "posicao": [1, 1], "volume_medio": [1.0, 1.0]})
    cal = pd.DatetimeIndex(pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"]))
    m = universo.membros_por_dia(u, cal)
    assert list(zip(m["data"].dt.strftime("%m-%d"), m["ticker"])) == [
        ("01-30", "AAAA3"), ("01-31", "AAAA3"), ("02-03", "BBBB3")]


def test_cobertura_conta_preco_no_proprio_mes():
    u = pd.DataFrame({"mes": ["2020-01", "2020-01", "2020-02"], "ticker": ["AAAA3", "MORT3", "MORT3"],
                      "data_inicio": pd.to_datetime(["2020-01-02"] * 2 + ["2020-02-03"]),
                      "posicao": [1, 2, 1], "volume_medio": 1.0})
    precos = pd.DataFrame({"data": pd.to_datetime(["2020-01-15", "2020-01-15"]), "ticker": ["AAAA3", "MORT3"],
                           "fechamento_ajustado": 1.0, "volume": 1.0})
    c = universo.calcular_cobertura(u, precos).set_index("mes")
    assert c.loc["2020-01", "cobertura"] == 1.0
    assert c.loc["2020-02", "cobertura"] == 0.0     # MORT3 sem preço em fevereiro
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_universo.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/universo.py`:

```python
"""
universo.py
Quais ações estão "na lista" em cada mês: uma aproximação do IBrX-100 que usa só o que se
sabia na época (sem viés de sobrevivência).

Regra, no primeiro pregão de cada mês:
    1. volume financeiro médio diário dos 'janela' pregões ANTERIORES (dia sem negócio conta como zero);
    2. uma ação por empresa (mesma raiz de 4 letras): fica a mais líquida;
    3. ficam as 'tamanho' maiores.
"""

import pandas as pd

import config

COLUNAS = ["mes", "data_inicio", "ticker", "posicao", "volume_medio"]


def montar_universo(b3: pd.DataFrame, tamanho: int = config.TAMANHO_UNIVERSO,
                    janela: int = config.JANELA_LIQUIDEZ) -> pd.DataFrame:
    calendario = pd.DatetimeIndex(sorted(b3["data"].unique()))
    volumes = (b3.pivot_table(index="data", columns="ticker", values="volume_financeiro", aggfunc="sum")
                 .reindex(calendario).fillna(0.0))
    primeiros = pd.Series(calendario).groupby(calendario.to_period("M")).first()

    linhas = []
    for mes, dia in primeiros.items():
        posicao_do_dia = calendario.get_loc(dia)
        if posicao_do_dia < janela:
            continue  # ainda não há 'janela' pregões de histórico
        media = volumes.iloc[posicao_do_dia - janela:posicao_do_dia].sum() / janela
        media = media[media > 0]
        mais_liquida_por_empresa = media.groupby(media.index.str[:4]).idxmax()
        escolhidas = media.loc[mais_liquida_por_empresa.values].sort_values(ascending=False).head(tamanho)
        for posicao, (ticker, volume) in enumerate(escolhidas.items(), start=1):
            linhas.append({"mes": str(mes), "data_inicio": dia, "ticker": ticker,
                           "posicao": posicao, "volume_medio": float(volume)})
    return pd.DataFrame(linhas, columns=COLUNAS)


def membros_por_dia(universo: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Expande o universo mensal para pares (data, ticker): cada dia usa a lista do seu mês."""
    dias = pd.DataFrame({"data": calendario})
    dias["mes"] = dias["data"].dt.to_period("M").astype(str)
    membros = dias.merge(universo[["mes", "ticker"]], on="mes", how="inner")
    return membros[["data", "ticker"]].sort_values(["data", "ticker"]).reset_index(drop=True)


def calcular_cobertura(universo: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """Para cada mês: quantas ações do universo têm preço ajustado (Yahoo) naquele mês."""
    com_preco = precos.assign(mes=precos["data"].dt.to_period("M").astype(str))[["mes", "ticker"]].drop_duplicates()
    marcado = universo.merge(com_preco, on=["mes", "ticker"], how="left", indicator=True)
    marcado["tem_preco"] = marcado["_merge"] == "both"
    resumo = marcado.groupby("mes").agg(no_universo=("ticker", "count"), com_preco=("tem_preco", "sum")).reset_index()
    resumo["cobertura"] = resumo["com_preco"] / resumo["no_universo"]
    return resumo
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_universo.py -q`
Expected: `7 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/universo.py tests/test_universo.py
git commit -m "Universo mensal pela liquidez passada, uma ação por empresa, e cobertura

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 5: Alvo e retornos diários

**Arquivos:**
- Criar: `src/alvo.py`
- Teste: `tests/test_alvo.py`

**Interfaces:**
- Consome: preços (`data, ticker, fechamento_ajustado, volume`, com BOVA11) e `calendario: pd.DatetimeIndex`.
- Produz:
  - `calcular_alvo(precos, calendario, horizonte=config.HORIZONTE) -> pd.DataFrame`, com colunas
    `data, ticker, excesso, rotulo (0/1)`. Só tem as linhas em que D e D+10 têm preço;
  - `retornos_diarios(precos, calendario) -> pd.DataFrame`: índice = calendário, colunas = tickers
    (inclui BOVA11). Usa o último preço disponível; depois que a ação some, o retorno é 0.

- [ ] **Passo 1: Escrever os testes**

`tests/test_alvo.py`:

```python
"""Alvo: 'a ação superou o BOVA11 nos próximos 10 pregões?' e retornos para a simulação."""

import numpy as np
import pandas as pd
import pytest

from src import alvo


def _precos(series: dict[str, list[float]], inicio="2024-01-01"):
    n = len(next(iter(series.values())))
    cal = pd.DatetimeIndex(pd.bdate_range(inicio, periods=n))
    linhas = [{"data": d, "ticker": t, "fechamento_ajustado": p, "volume": 1.0}
              for t, ps in series.items() for d, p in zip(cal, ps) if not np.isnan(p)]
    return pd.DataFrame(linhas), cal


def test_excesso_sobre_o_bova11_em_10_pregoes():
    precos, cal = _precos({"AAAA3": [100 * 1.01 ** i for i in range(15)], "BOVA11": [50.0] * 15})
    a = alvo.calcular_alvo(precos, cal, horizonte=10)
    assert len(a) == 5                                     # só os 5 dias com D+10 conhecido
    assert a["excesso"].iloc[0] == pytest.approx(1.01 ** 10 - 1)
    assert (a["rotulo"] == 1).all()
    assert list(a.columns) == ["data", "ticker", "excesso", "rotulo"]


def test_sem_preco_em_d_mais_10_nao_ha_rotulo():
    precos_a = [100.0] * 15
    precos_a[10] = np.nan                                  # suspensa no dia D+10 de D=0
    precos, cal = _precos({"AAAA3": precos_a, "BOVA11": [50.0] * 15})
    a = alvo.calcular_alvo(precos, cal, horizonte=10)
    assert cal[0] not in set(a["data"])


def test_rotulo_zero_quando_fica_atras():
    precos, cal = _precos({"AAAA3": [100.0] * 12, "BOVA11": [50 * 1.01 ** i for i in range(12)]})
    assert (alvo.calcular_alvo(precos, cal, horizonte=10)["rotulo"] == 0).all()


def test_sem_bova11_da_erro_claro():
    precos, cal = _precos({"AAAA3": [100.0] * 12})
    with pytest.raises(ValueError, match="BOVA11"):
        alvo.calcular_alvo(precos, cal)


def test_retornos_diarios_apos_a_acao_sumir_sao_zero():
    precos, cal = _precos({"AAAA3": [10.0, 11.0, np.nan, np.nan], "BOVA11": [1.0, 1.0, 1.0, 1.0]})
    r = alvo.retornos_diarios(precos, cal)
    assert list(r.index) == list(cal)
    assert r["AAAA3"].tolist() == pytest.approx([0.0, 0.1, 0.0, 0.0])
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_alvo.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/alvo.py`:

```python
"""
alvo.py
O que a linha tenta prever e quanto a carteira ganhou de fato.

    excesso(i, D) = retorno ajustado de i entre D e D+10 − retorno ajustado do BOVA11 entre D e D+10
    rotulo        = 1 se excesso > 0, senão 0

O rótulo de D só fica conhecido no fechamento de D+10. Por isso o treino usa EMBARGO
(ver src/laboratorio/walk_forward.py). Ele nunca entra nas features.
"""

import pandas as pd

import config


def _larga(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Tabela larga: uma linha por pregão do calendário, uma coluna por ticker."""
    return (precos.pivot_table(index="data", columns="ticker", values="fechamento_ajustado", aggfunc="last")
                  .reindex(calendario))


def calcular_alvo(precos: pd.DataFrame, calendario: pd.DatetimeIndex,
                  horizonte: int = config.HORIZONTE) -> pd.DataFrame:
    larga = _larga(precos, calendario)
    if config.REFERENCIA not in larga.columns:
        raise ValueError(f"faltam os preços do {config.REFERENCIA} (a referência do alvo).")
    futuro = larga.shift(-horizonte) / larga - 1          # NaN se faltar preço em D ou em D+h
    excesso = futuro.drop(columns=config.REFERENCIA).sub(futuro[config.REFERENCIA], axis=0)
    longo = (excesso.rename_axis(index="data", columns="ticker").reset_index()
                    .melt(id_vars="data", var_name="ticker", value_name="excesso")
                    .dropna(subset=["excesso"]))
    longo["rotulo"] = (longo["excesso"] > 0).astype(int)
    return longo.sort_values(["data", "ticker"]).reset_index(drop=True)


def retornos_diarios(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Retorno de cada pregão para a simulação da carteira. Repete o último preço conhecido; depois
    que a ação some, o retorno fica 0 (a posição é encerrada no último preço disponível).
    """
    larga = _larga(precos, calendario).ffill()
    return (larga / larga.shift(1) - 1).fillna(0.0)
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_alvo.py -q`
Expected: `5 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/alvo.py tests/test_alvo.py
git commit -m "Alvo (excesso sobre o BOVA11 em 10 pregões) e retornos diários

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 6: Features de preço e ranks

**Arquivos:**
- Criar: `src/features/preco.py`
- Teste: `tests/test_features.py`

**Interfaces:**
- Consome: preços e calendário; `membros_por_dia` da Tarefa 4 (as colunas `data, ticker`).
- Produz:
  - `COLUNAS: list[str]`, com as 11 features abaixo;
  - `COLUNAS_MODELO: list[str]`, igual a `COLUNAS` mais `"{c}_rank"` para cada uma (22 colunas);
  - `calcular_features(precos, calendario) -> pd.DataFrame`, com colunas `data, ticker, *COLUNAS`.
    Só tem as linhas com 120 pregões de histórico, e o BOVA11 não aparece como linha;
  - `adicionar_ranks(feat, membros) -> pd.DataFrame`: mantém só os membros do universo do dia e
    acrescenta os ranks, que vão de 0 a 1 dentro de cada dia.

- [ ] **Passo 1: Escrever os testes**

`tests/test_features.py`:

```python
"""Features de preço: só passado, 120 pregões de histórico e ranks dentro do universo do dia."""

import numpy as np
import pandas as pd
import pytest

from src.features import preco


def test_nenhuma_feature_olha_para_frente(sinteticos):
    precos, cal = sinteticos
    dia = cal[400]
    original = preco.calcular_features(precos, cal)
    alterados = precos.copy()
    futuro = alterados["data"] > dia
    alterados.loc[futuro, "fechamento_ajustado"] *= 3.0
    alterados.loc[futuro, "volume"] *= 10
    novo = preco.calcular_features(alterados, cal)
    ate_hoje = lambda f: f[f["data"] <= dia].sort_values(["data", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ate_hoje(original), ate_hoje(novo))


def test_precisa_de_120_pregoes_e_bova_nao_e_linha(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal)
    assert f["data"].min() == cal[120]
    assert "BOVA11" not in set(f["ticker"])
    assert f[preco.COLUNAS].notna().all().all()


def test_valores_conferem_com_a_conta_na_mao(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal).set_index(["data", "ticker"])
    p = precos.pivot(index="data", columns="ticker", values="fechamento_ajustado")
    d, t = cal[300], "AC003"
    assert f.loc[(d, t), "p_ret_5"] == pytest.approx(p.loc[d, t] / p[t].iloc[295] - 1)
    rel = (p.loc[d, t] / p[t].iloc[280] - 1) - (p.loc[d, "BOVA11"] / p["BOVA11"].iloc[280] - 1)
    assert f.loc[(d, t), "p_rel_bova_20"] == pytest.approx(rel)


def test_ranks_so_entre_os_membros_do_dia(sinteticos):
    precos, cal = sinteticos
    f = preco.calcular_features(precos, cal)
    dia = cal[200]
    membros = pd.DataFrame({"data": [dia] * 4, "ticker": ["AC003", "AC013", "AC023", "AC033"]})
    r = preco.adicionar_ranks(f, membros)
    assert set(r["ticker"]) == {"AC003", "AC013", "AC023", "AC033"}
    assert set(r["data"]) == {dia}
    assert sorted(r["p_ret_20_rank"]) == [0.25, 0.5, 0.75, 1.0]
    assert len(preco.COLUNAS_MODELO) == 22
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/features/preco.py`:

```python
"""
preco.py
Features de swing trade calculadas SÓ com preços e volumes até o fechamento de D.
Toda coluna começa com "p_" (de preço); na etapa 3 virão as "t_" (de textos).

    p_ret_N        retorno em N pregões (5, 20, 60, 120)
    p_dist_mmN     distância do preço até a média de N pregões (20, 50)
    p_vol_N        volatilidade diária em N pregões (20, 60)
    p_volume_rel_20  volume de hoje / média de 20 pregões
    p_rel_bova_N   retorno em N pregões menos o do BOVA11 (20, 60)

A ação precisa de 120 pregões de histórico (a p_ret_120 precisa deles); antes disso, fica de fora.
Os "_rank" dão a posição da ação entre as do universo NAQUELE DIA (0 a 1), o que deixa
comparáveis anos calmos e anos agitados.
"""

import numpy as np
import pandas as pd

import config

COLUNAS = [
    "p_ret_5", "p_ret_20", "p_ret_60", "p_ret_120",
    "p_dist_mm20", "p_dist_mm50",
    "p_vol_20", "p_vol_60",
    "p_volume_rel_20",
    "p_rel_bova_20", "p_rel_bova_60",
]
COLUNAS_MODELO = COLUNAS + [f"{c}_rank" for c in COLUNAS]


def _de_uma_acao(g: pd.DataFrame) -> pd.DataFrame:
    """g: uma ação ordenada por data, com fechamento_ajustado, volume e bova (preço do BOVA11 no dia)."""
    preco = g["fechamento_ajustado"]
    bova = g["bova"]
    volume = g["volume"].replace(0, np.nan)
    retorno = preco / preco.shift(1) - 1

    saida = g[["data", "ticker"]].copy()
    for n in (5, 20, 60, 120):
        saida[f"p_ret_{n}"] = preco / preco.shift(n) - 1
    saida["p_dist_mm20"] = preco / preco.rolling(20).mean() - 1
    saida["p_dist_mm50"] = preco / preco.rolling(50).mean() - 1
    saida["p_vol_20"] = retorno.rolling(20).std()
    saida["p_vol_60"] = retorno.rolling(60).std()
    saida["p_volume_rel_20"] = volume / volume.rolling(20, min_periods=10).mean()
    for n in (20, 60):
        saida[f"p_rel_bova_{n}"] = saida[f"p_ret_{n}"] - (bova / bova.shift(n) - 1)
    return saida


def calcular_features(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    precos = precos[precos["data"].isin(calendario)]
    bova = (precos.loc[precos["ticker"] == config.REFERENCIA, ["data", "fechamento_ajustado"]]
                  .rename(columns={"fechamento_ajustado": "bova"}))
    acoes = precos[precos["ticker"] != config.REFERENCIA]
    partes = [_de_uma_acao(g.sort_values("data").merge(bova, on="data", how="left"))
              for _, g in acoes.groupby("ticker")]
    if not partes:
        return pd.DataFrame(columns=["data", "ticker", *COLUNAS])
    feat = pd.concat(partes, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=["p_ret_120"])  # 120 pregões de histórico
    return feat.sort_values(["data", "ticker"]).reset_index(drop=True)


def adicionar_ranks(feat: pd.DataFrame, membros: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as ações do universo de cada dia e acrescenta a posição de cada feature (0 a 1)."""
    base = feat.merge(membros, on=["data", "ticker"], how="inner")
    for coluna in COLUNAS:
        base[f"{coluna}_rank"] = base.groupby("data")[coluna].rank(pct=True)
    return base.sort_values(["data", "ticker"]).reset_index(drop=True)
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -q`
Expected: `4 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/features/preco.py tests/test_features.py
git commit -m "Features de preço para swing trade, sem olhar o futuro, com ranks no universo

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 7: Base completa e comando "atualizar"

**Arquivos:**
- Criar: `src/base.py`, `src/atualizar.py`
- Teste: `tests/test_base.py`

**Interfaces:**
- Consome: Tarefas 2 a 6.
- Produz:
  - `Base` (dataclass) com os campos:
    - `tabela: pd.DataFrame`: colunas `data, ticker, *COLUNAS_MODELO, excesso, rotulo`, em que
      `rotulo` vale NaN quando ainda é desconhecido;
    - `calendario: pd.DatetimeIndex`;
    - `retornos: pd.DataFrame`, indexado pelo calendário;
  - `montar_calendario(datas_b3, datas_referencia) -> pd.DatetimeIndex`;
  - `montar_base(precos, universo, calendario) -> Base`;
  - `carregar_base() -> Base`. Ele levanta `SystemExit` com uma mensagem clara se faltar algum
    arquivo de `atualizar`;
  - `src/atualizar.py`: `rodar() -> None`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_base.py`:

```python
"""A base que alimenta o laboratório, os sinais e a prova final; e o comando 'atualizar'."""

import pandas as pd
import pytest

import config
from src import atualizar, base
from src.features.preco import COLUNAS_MODELO
from tests.conftest import gerar_precos, universo_completo


def test_base_tem_features_alvo_e_retornos(base_sintetica):
    t = base_sintetica.tabela
    assert list(t.columns[:2]) == ["data", "ticker"]
    assert set(COLUNAS_MODELO) <= set(t.columns) and {"excesso", "rotulo"} <= set(t.columns)
    ultimos_10 = base_sintetica.calendario[-10:]
    assert t[t["data"].isin(ultimos_10)]["rotulo"].isna().all()       # futuro ainda desconhecido
    assert t[~t["data"].isin(ultimos_10)]["rotulo"].notna().all()
    assert list(base_sintetica.retornos.index) == list(base_sintetica.calendario)


def test_acao_do_universo_sem_preco_nao_quebra():
    precos, cal = gerar_precos(n_acoes=5, fim="2016-06-30")
    u = universo_completo(precos, cal)
    u = pd.concat([u, u[u["ticker"] == "AC003"].assign(ticker="MORT3")])   # existe no universo, não no Yahoo
    b = base.montar_base(precos, u, cal)
    assert "MORT3" not in set(b.tabela["ticker"])


def test_calendario_e_a_uniao_da_b3_com_o_bova11():
    b3 = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bova = pd.to_datetime(["2024-01-03", "2024-01-04"])                  # B3 do ano atrasada
    cal = base.montar_calendario(pd.Series(b3), pd.Series(bova))
    assert list(cal.strftime("%m-%d")) == ["01-02", "01-03", "01-04"]


def test_sem_atualizar_mensagem_clara(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARQUIVO_PRECOS", tmp_path / "nao_existe.csv.gz")
    with pytest.raises(SystemExit, match="atualizar"):
        base.carregar_base()


def test_atualizar_de_ponta_a_ponta_com_fontes_falsas(tmp_path, monkeypatch):
    """Roda o 'atualizar' inteiro trocando a internet por dados inventados."""
    for nome in ["PASTA_B3", "PASTA_PROCESSADOS", "PASTA_UNIVERSO"]:
        monkeypatch.setattr(config, nome, tmp_path / nome.lower())
    monkeypatch.setattr(config, "ARQUIVO_B3", tmp_path / "pp" / "b3.csv.gz")
    monkeypatch.setattr(config, "ARQUIVO_PRECOS", tmp_path / "pp" / "precos.csv.gz")
    monkeypatch.setattr(config, "ARQUIVO_CALENDARIO", tmp_path / "pp" / "calendario.csv")
    monkeypatch.setattr(config, "ARQUIVO_UNIVERSO", tmp_path / "u" / "universo.csv")
    monkeypatch.setattr(config, "ARQUIVO_COBERTURA", tmp_path / "u" / "cobertura.csv")
    monkeypatch.setattr(config, "ARQUIVO_TROCA_DE_NOME", tmp_path / "u" / "troca.csv")
    monkeypatch.setattr(config, "DATA_DOWNLOAD_INICIO", "2024-01-01")

    precos, cal = gerar_precos(n_acoes=4, inicio="2024-01-01", fim="2024-12-31")
    b3_falso = precos[precos["ticker"] != "BOVA11"].rename(columns={"fechamento_ajustado": "fechamento"})
    b3_falso = b3_falso.assign(nome="X", especie="ON", volume_financeiro=b3_falso["volume"])

    def baixar_ano_falso(ano, forcar=False):
        if ano > 2024:
            raise ConnectionError("arquivo do ano ainda não existe")
        return tmp_path / f"{ano}.zip"

    monkeypatch.setattr(atualizar.b3, "baixar_ano", baixar_ano_falso)
    monkeypatch.setattr(atualizar.b3, "caminho_do_ano", lambda ano: tmp_path / f"{ano}.zip")
    monkeypatch.setattr(atualizar.b3, "ler_ano", lambda caminho: b3_falso)
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (precos[precos["ticker"].isin(tickers)], []))
    monkeypatch.setattr(atualizar, "ano_atual", lambda: 2025)            # 2025 falha e não tem cópia: é pulado

    atualizar.rodar()
    assert config.ARQUIVO_UNIVERSO.exists() and config.ARQUIVO_PRECOS.exists()
    assert len(pd.read_csv(config.ARQUIVO_CALENDARIO)) == len(cal)
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_base.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar a base**

`src/base.py`:

```python
"""
base.py
Junta as peças da etapa 1 numa única "base" usada pelo laboratório, pelos sinais e pela prova final.

    tabela:     uma linha por (dia, ação do universo) com as features e o alvo
                (rotulo = NaN quando o futuro ainda não aconteceu)
    calendario: os pregões
    retornos:   retorno diário de cada ação (para simular a carteira)
"""

from dataclasses import dataclass

import pandas as pd

import config
from src.alvo import calcular_alvo, retornos_diarios
from src.features.preco import adicionar_ranks, calcular_features
from src.universo import membros_por_dia


@dataclass
class Base:
    tabela: pd.DataFrame
    calendario: pd.DatetimeIndex
    retornos: pd.DataFrame


def montar_calendario(datas_b3: pd.Series, datas_referencia: pd.Series) -> pd.DatetimeIndex:
    """
    Pregões = datas da B3 + datas do BOVA11 no Yahoo. A união cobre o caso de o arquivo anual
    da B3 do ano corrente ainda não ter os últimos dias.
    """
    datas = pd.concat([pd.Series(datas_b3), pd.Series(datas_referencia)]).drop_duplicates().sort_values()
    datas = datas[datas >= pd.Timestamp(config.DATA_DOWNLOAD_INICIO)]
    return pd.DatetimeIndex(datas.values)


def montar_base(precos: pd.DataFrame, universo: pd.DataFrame, calendario: pd.DatetimeIndex) -> Base:
    feat = calcular_features(precos, calendario)
    feat = adicionar_ranks(feat, membros_por_dia(universo, calendario))
    alvo = calcular_alvo(precos, calendario)
    tabela = feat.merge(alvo, on=["data", "ticker"], how="left")
    return Base(tabela=tabela, calendario=calendario, retornos=retornos_diarios(precos, calendario))


def carregar_base() -> Base:
    """Lê os arquivos gerados por 'python main.py atualizar' e monta a base."""
    for arquivo in (config.ARQUIVO_PRECOS, config.ARQUIVO_UNIVERSO, config.ARQUIVO_CALENDARIO):
        if not arquivo.exists():
            raise SystemExit(f"Não achei {arquivo.name}. Rode antes: python main.py atualizar")
    precos = pd.read_csv(config.ARQUIVO_PRECOS, parse_dates=["data"])
    universo = pd.read_csv(config.ARQUIVO_UNIVERSO, parse_dates=["data_inicio"], dtype={"mes": str})
    calendario = pd.DatetimeIndex(pd.read_csv(config.ARQUIVO_CALENDARIO, parse_dates=["data"])["data"])
    return montar_base(precos, universo, calendario)
```

- [ ] **Passo 4: Implementar o "atualizar"**

`src/atualizar.py`:

```python
"""
atualizar.py
Etapa "python main.py atualizar": baixa e prepara tudo o que a linha usa.

    1. B3 (COTAHIST) de DATA_DOWNLOAD_INICIO até hoje. O ano corrente é sempre baixado de novo.
    2. Universo de cada mês (as 100 mais líquidas, uma por empresa).
    3. Yahoo: preços ajustados de todas as ações que já passaram pelo universo, mais o BOVA11.
    4. Calendário de pregões e relatório de cobertura.

Na primeira vez são ~10 anos de dados da B3 (alguns minutos); depois, só o ano corrente.
"""

from datetime import date

import pandas as pd

import config
from src import universo as universo_mod
from src.base import montar_calendario
from src.coleta import b3, yahoo


def ano_atual() -> int:
    return date.today().year


def _baixar_b3() -> pd.DataFrame:
    partes = []
    for ano in range(int(config.DATA_DOWNLOAD_INICIO[:4]), ano_atual() + 1):
        print(f"B3 {ano}:", end=" ", flush=True)
        try:
            caminho = b3.baixar_ano(ano, forcar=(ano == ano_atual()))
        except Exception as erro:
            caminho = b3.caminho_do_ano(ano)
            if not caminho.exists():
                print(f"-> não consegui baixar ({erro}) e não há cópia local. Ano pulado.")
                continue
            print(f"-> não consegui baixar ({erro}); usando a cópia já baixada.", end=" ")
        tabela = b3.ler_ano(caminho)
        print(f"-> {len(tabela)} linhas de ações")
        partes.append(tabela)
    if not partes:
        raise SystemExit("Nenhum arquivo da B3 disponível. Confira a internet.")
    return pd.concat(partes, ignore_index=True)


def rodar() -> None:
    for pasta in (config.PASTA_PROCESSADOS, config.PASTA_UNIVERSO, config.ARQUIVO_B3.parent,
                  config.ARQUIVO_UNIVERSO.parent, config.ARQUIVO_CALENDARIO.parent):
        pasta.mkdir(parents=True, exist_ok=True)

    tabela_b3 = _baixar_b3()
    tabela_b3.to_csv(config.ARQUIVO_B3, index=False, date_format="%Y-%m-%d")

    print("\nMontando o universo de cada mês...")
    universo = universo_mod.montar_universo(tabela_b3)
    universo.to_csv(config.ARQUIVO_UNIVERSO, index=False, date_format="%Y-%m-%d")
    troca = b3.tickers_com_troca_de_nome(tabela_b3[tabela_b3["ticker"].isin(universo["ticker"])])
    troca.to_csv(config.ARQUIVO_TROCA_DE_NOME, index=False)

    tickers = sorted(set(universo["ticker"])) + [config.REFERENCIA]
    print(f"\nBaixando preços ajustados de {len(tickers)} tickers no Yahoo...")
    precos, falharam = yahoo.baixar_varios(tickers, config.DATA_DOWNLOAD_INICIO)
    if config.REFERENCIA not in set(precos["ticker"]):
        raise SystemExit(f"Não consegui os preços do {config.REFERENCIA}, que é a referência. Tente de novo.")
    precos.to_csv(config.ARQUIVO_PRECOS, index=False, date_format="%Y-%m-%d")

    calendario = montar_calendario(tabela_b3["data"], precos.loc[precos["ticker"] == config.REFERENCIA, "data"])
    pd.DataFrame({"data": calendario}).to_csv(config.ARQUIVO_CALENDARIO, index=False, date_format="%Y-%m-%d")

    cobertura = universo_mod.calcular_cobertura(universo, precos)
    cobertura.to_csv(config.ARQUIVO_COBERTURA, index=False, float_format="%.4f")
    media = cobertura["cobertura"].mean()

    print("\n===== Resumo =====")
    print(f"Meses no universo: {universo['mes'].nunique()} | ações diferentes: {universo['ticker'].nunique()}")
    print(f"Pregões: {calendario[0]:%d/%m/%Y} a {calendario[-1]:%d/%m/%Y} ({len(calendario)})")
    print(f"Sem dados no Yahoo: {len(falharam)} ticker(s)")
    print(f"Cobertura média do universo com preço ajustado: {media:.1%}")
    if media < 0.95:
        print("ATENÇÃO: cobertura abaixo de 95%. Parte do viés de sobrevivência continua "
              "(ver 'Riscos abertos' na especificação).")
    if not troca.empty:
        print(f"{len(troca)} ticker(s) com troca de nome: veja {config.ARQUIVO_TROCA_DE_NOME.name}")
```

- [ ] **Passo 5: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_base.py -q`
Expected: `5 passed`.

- [ ] **Passo 6: Commit**

```bash
git add src/base.py src/atualizar.py tests/test_base.py
git commit -m "Base da etapa 1 (features + alvo + retornos) e comando atualizar

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 8: Combinador e walk-forward com embargo

**Arquivos:**
- Criar: `src/modelos/combinador.py`, `src/laboratorio/walk_forward.py`
- Teste: `tests/test_walk_forward.py`

**Interfaces:**
- Consome: `Base` (Tarefa 7), `COLUNAS_MODELO` (Tarefa 6).
- Produz:
  - em `combinador`: `criar_modelo(tipo: str)`,
    `treinar(x: pd.DataFrame, y: pd.Series, tipo: str)` e
    `prob_superar(modelo, x: pd.DataFrame) -> np.ndarray`;
  - em `walk_forward`:
    - `datas_de_retreino(calendario, inicio, fim) -> list[pd.Timestamp]`;
    - `limite_de_treino(calendario, dia, horizonte=config.HORIZONTE) -> pd.Timestamp | None`;
    - `ultimo_dia_avaliavel(calendario, fim, horizonte=config.HORIZONTE) -> pd.Timestamp | None`;
    - `gerar_probabilidades(tabela, calendario, inicio, fim, tipo, colunas=COLUNAS_MODELO, horizonte=config.HORIZONTE, treinar=combinador.treinar, prob=combinador.prob_superar) -> pd.DataFrame`,
      com colunas `data, ticker, prob`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_walk_forward.py`:

```python
"""Walk-forward mensal com embargo: o modelo nunca vê rótulos que ainda não existiam."""

import numpy as np
import pandas as pd
import pytest

from src.features.preco import COLUNAS_MODELO
from src.laboratorio import walk_forward as wf
from src.modelos import combinador


def test_datas_de_retreino_sao_o_primeiro_pregao_de_cada_mes():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-31"))
    assert [d.strftime("%m-%d") for d in wf.datas_de_retreino(cal, "2024-01-01", "2024-03-31")] == ["01-01", "02-01", "03-01"]


def test_limite_e_ultimo_dia_avaliavel():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    assert wf.limite_de_treino(cal, cal[15], horizonte=10) == cal[5]
    assert wf.limite_de_treino(cal, cal[5], horizonte=10) is None
    assert wf.ultimo_dia_avaliavel(cal, cal[-1], horizonte=10) == cal[-11]


def test_embargo_o_treino_so_ve_rotulos_ja_conhecidos(base_sintetica):
    vistos = []

    def treinar_espiao(x, y, tipo):
        vistos.append(len(x))
        return combinador.treinar(x, y, tipo)

    b = base_sintetica
    inicio, fim = b.calendario[300], b.calendario[420]
    wf.gerar_probabilidades(b.tabela, b.calendario, inicio, fim, "logistica", treinar=treinar_espiao)
    for dia, n in zip(wf.datas_de_retreino(b.calendario, inicio, fim), vistos):
        limite = wf.limite_de_treino(b.calendario, dia)
        esperado = b.tabela[(b.tabela["data"] <= limite) & b.tabela["rotulo"].notna()]
        assert n == len(esperado)


def test_decisao_nao_muda_se_o_futuro_mudar(base_sintetica):
    """
    Bagunça (1) as features de depois de 'fim' e (2) os rótulos que nenhum retreino podia conhecer
    (depois do limite do ÚLTIMO retreino da janela). As probabilidades têm de ficar idênticas.
    """
    b = base_sintetica
    inicio, fim = b.calendario[300], b.calendario[330]
    original = wf.gerar_probabilidades(b.tabela, b.calendario, inicio, fim, "logistica")

    ultimo_retreino = wf.datas_de_retreino(b.calendario, inicio, fim)[-1]
    rotulo_desconhecido = b.tabela["data"] > wf.limite_de_treino(b.calendario, ultimo_retreino)
    depois_do_fim = b.tabela["data"] > fim
    rng = np.random.default_rng(3)
    bagunca = b.tabela.copy()
    bagunca.loc[depois_do_fim, COLUNAS_MODELO] = rng.normal(size=(depois_do_fim.sum(), len(COLUNAS_MODELO)))
    bagunca.loc[rotulo_desconhecido, "rotulo"] = rng.integers(0, 2, rotulo_desconhecido.sum())

    pd.testing.assert_frame_equal(original, wf.gerar_probabilidades(bagunca, b.calendario, inicio, fim, "logistica"))


def test_passeio_aleatorio_acerto_perto_de_50(base_sintetica):
    b = base_sintetica
    probs = wf.gerar_probabilidades(b.tabela, b.calendario, b.calendario[250], b.calendario[-11], "logistica")
    m = probs.merge(b.tabela[["data", "ticker", "rotulo"]], on=["data", "ticker"]).dropna()
    acerto = ((m["prob"] > 0.5) == (m["rotulo"] == 1)).mean()
    assert len(m) > 5000
    assert 0.40 <= acerto <= 0.60


def test_boosting_tambem_funciona(base_sintetica):
    b = base_sintetica
    probs = wf.gerar_probabilidades(b.tabela, b.calendario, b.calendario[400], b.calendario[430], "boosting")
    assert probs["prob"].between(0, 1).all() and len(probs) > 0


def test_tipo_desconhecido_da_erro():
    with pytest.raises(ValueError, match="tipo"):
        combinador.criar_modelo("rede-neural")
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_walk_forward.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar o combinador**

`src/modelos/combinador.py`:

```python
"""
combinador.py
O ML que junta todos os sinais e aprende no histórico quanto cada um vale.

    "logistica": regressão logística (a régua simples e transparente)
    "boosting":  gradient boosting de histogramas (mais flexível; aceita valores vazios)

As funções recebem SÓ features (e, no treino, o rótulo dos dias passados). Datas e embargo
ficam em src/laboratorio/walk_forward.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def criar_modelo(tipo: str):
    if tipo == "logistica":
        # Imputação pela mediana do TREINO, normalização e logística: tudo aprende só com o treino.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    if tipo == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, random_state=config.SEMENTE)
    raise ValueError(f"tipo de modelo desconhecido: '{tipo}' (use um de {config.TIPOS_MODELO})")


def treinar(x: pd.DataFrame, y: pd.Series, tipo: str):
    modelo = criar_modelo(tipo)
    modelo.fit(x, y)
    return modelo


def prob_superar(modelo, x: pd.DataFrame) -> np.ndarray:
    """Probabilidade (0 a 1) de cada ação superar o BOVA11 nos próximos 10 pregões."""
    return modelo.predict_proba(x)[:, 1]
```

- [ ] **Passo 4: Implementar o walk-forward**

`src/laboratorio/walk_forward.py`:

```python
"""
walk_forward.py
Treina de novo no primeiro pregão de cada mês, usando só o passado, com EMBARGO.

Por que o embargo: o rótulo do dia D só fica conhecido no fechamento de D+10. Um treino feito
no dia T só pode usar linhas com D ≤ calendario[pos(T) − 10]. Sem isso, o modelo aprenderia
com resultados que ainda não tinham acontecido.
"""

import pandas as pd

import config
from src.features.preco import COLUNAS_MODELO
from src.modelos import combinador


def datas_de_retreino(calendario: pd.DatetimeIndex, inicio, fim) -> list[pd.Timestamp]:
    dias = calendario[(calendario >= pd.Timestamp(inicio)) & (calendario <= pd.Timestamp(fim))]
    return list(pd.Series(dias).groupby(dias.to_period("M")).first())


def limite_de_treino(calendario: pd.DatetimeIndex, dia, horizonte: int = config.HORIZONTE):
    """Último dia cujo rótulo já era conhecido no fechamento de 'dia'. None se não houver."""
    posicao = calendario.get_loc(pd.Timestamp(dia)) - horizonte
    return calendario[posicao] if posicao >= 0 else None


def ultimo_dia_avaliavel(calendario: pd.DatetimeIndex, fim, horizonte: int = config.HORIZONTE):
    """Último dia D cujo resultado (D + horizonte) cai até 'fim'. None se não houver."""
    dias = calendario[calendario <= pd.Timestamp(fim)]
    posicao = len(dias) - 1 - horizonte
    return dias[posicao] if posicao >= 0 else None


def gerar_probabilidades(tabela: pd.DataFrame, calendario: pd.DatetimeIndex, inicio, fim, tipo: str,
                         colunas: list[str] = COLUNAS_MODELO, horizonte: int = config.HORIZONTE,
                         treinar=combinador.treinar, prob=combinador.prob_superar) -> pd.DataFrame:
    """Probabilidade de cada (dia, ação) entre 'inicio' e 'fim', com retreino mensal."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    retreinos = datas_de_retreino(calendario, inicio, fim)
    partes = []
    for i, dia in enumerate(retreinos):
        proximo = retreinos[i + 1] if i + 1 < len(retreinos) else fim + pd.Timedelta(days=1)
        limite = limite_de_treino(calendario, dia, horizonte)
        if limite is None:
            continue
        treino = tabela[(tabela["data"] <= limite) & tabela["rotulo"].notna()]
        alvo = tabela[(tabela["data"] >= dia) & (tabela["data"] < proximo) & (tabela["data"] <= fim)]
        if treino.empty or alvo.empty or treino["rotulo"].nunique() < 2:
            continue
        modelo = treinar(treino[colunas], treino["rotulo"].astype(int), tipo)
        partes.append(pd.DataFrame({"data": alvo["data"].values, "ticker": alvo["ticker"].values,
                                    "prob": prob(modelo, alvo[colunas])}))
    if not partes:
        return pd.DataFrame(columns=["data", "ticker", "prob"])
    return pd.concat(partes, ignore_index=True)
```

- [ ] **Passo 5: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_walk_forward.py -q`
Expected: `7 passed`. O teste de passeio aleatório leva alguns segundos.

- [ ] **Passo 6: Commit**

```bash
git add src/modelos/combinador.py src/laboratorio/walk_forward.py tests/test_walk_forward.py
git commit -m "Combinador (logística e boosting) e walk-forward mensal com embargo de 10 pregões

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 9: Regra dos sinais

**Arquivos:**
- Criar: `src/sinais.py`
- Teste: `tests/test_sinais.py`

**Interfaces:**
- Produz: `gerar_sinais(probs: pd.DataFrame, k: int, p_min: float) -> pd.DataFrame`, com
  colunas `data, ticker, prob, sinal ("compra" | "venda")`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_sinais.py`:

```python
"""Sinais: top-k compras acima do mínimo, top-k vendas abaixo de 1 − mínimo, sem conflito."""

import pandas as pd

from src.sinais import gerar_sinais


def _probs(valores: dict[str, float], data="2024-01-02"):
    return pd.DataFrame({"data": pd.Timestamp(data), "ticker": list(valores), "prob": list(valores.values())})


def test_compra_e_venda_respeitam_k_e_minimo():
    p = _probs({"A": 0.70, "B": 0.60, "C": 0.54, "D": 0.50, "E": 0.44, "F": 0.30})
    s = gerar_sinais(p, k=2, p_min=0.55)
    assert list(s.loc[s["sinal"] == "compra", "ticker"]) == ["A", "B"]
    assert list(s.loc[s["sinal"] == "venda", "ticker"]) == ["F", "E"]


def test_sem_conviccao_nao_ha_sinal():
    s = gerar_sinais(_probs({"A": 0.52, "B": 0.50, "C": 0.48}), k=3, p_min=0.55)
    assert s.empty and list(s.columns) == ["data", "ticker", "prob", "sinal"]


def test_mesma_acao_nunca_e_compra_e_venda_no_mesmo_dia():
    s = gerar_sinais(_probs({"A": 0.50, "B": 0.50}), k=5, p_min=0.50)   # empate exato no limite
    assert not s.duplicated(["data", "ticker"]).any()


def test_cada_dia_tem_seus_proprios_sinais():
    p = pd.concat([_probs({"A": 0.9, "B": 0.1}, "2024-01-02"), _probs({"A": 0.1, "B": 0.9}, "2024-01-03")])
    s = gerar_sinais(p, k=1, p_min=0.55).set_index(["data", "sinal"])["ticker"]
    assert s[(pd.Timestamp("2024-01-02"), "compra")] == "A"
    assert s[(pd.Timestamp("2024-01-03"), "compra")] == "B"
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_sinais.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/sinais.py`:

```python
"""
sinais.py
Transforma probabilidades em sinais.

    compra: as k ações de MAIOR probabilidade de superar o BOVA11, se prob ≥ p_min
    venda:  as k ações de MENOR probabilidade, se prob ≤ 1 − p_min
            ("deve ficar para trás"; não é recomendação de vender a descoberto)

Uma ação nunca é compra e venda no mesmo dia (pode acontecer num empate exato com p_min = 0,5).
"""

import pandas as pd

COLUNAS = ["data", "ticker", "prob", "sinal"]


def gerar_sinais(probs: pd.DataFrame, k: int, p_min: float) -> pd.DataFrame:
    if probs.empty:
        return pd.DataFrame(columns=COLUNAS)
    topo = probs.sort_values(["data", "prob"], ascending=[True, False]).groupby("data").head(k)
    compras = topo[topo["prob"] >= p_min].assign(sinal="compra")
    fundo = probs.sort_values(["data", "prob"], ascending=[True, True]).groupby("data").head(k)
    vendas = fundo[fundo["prob"] <= 1 - p_min].assign(sinal="venda")
    vendas = vendas.merge(compras[["data", "ticker"]], on=["data", "ticker"], how="left", indicator=True)
    vendas = vendas[vendas["_merge"] == "left_only"].drop(columns="_merge")
    sinais = pd.concat([compras, vendas], ignore_index=True)[COLUNAS]
    return sinais.reset_index(drop=True)
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_sinais.py -q`
Expected: `4 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/sinais.py tests/test_sinais.py
git commit -m "Regra dos sinais de compra e venda (top-k com probabilidade mínima)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 10: Simulação da carteira de compras

**Arquivos:**
- Criar: `src/laboratorio/carteira.py`
- Teste: `tests/test_carteira.py`

**Interfaces:**
- Consome: os sinais (Tarefa 9) e `Base.retornos` (Tarefa 7).
- Produz: `simular_compras(sinais, retornos, inicio, fim, horizonte=config.HORIZONTE, custo=config.CUSTO_POR_LADO) -> pd.Series`,
  com o retorno diário líquido da carteira, indexado pelos pregões entre `inicio` e `fim`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_carteira.py`:

```python
"""Carteira em 10 fatias: cada pregão uma fatia compra os sinais do dia e segura 10 pregões."""

import pandas as pd
import pytest

from src.laboratorio.carteira import simular_compras


def _retornos(n=4, r=0.01):
    dias = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"A": [r] * n, "B": [0.0] * n}, index=dias)


def test_conta_na_mao_com_custos():
    ret = _retornos()
    sinais = pd.DataFrame({"data": [ret.index[0]], "ticker": ["A"], "prob": [0.7], "sinal": ["compra"]})
    serie = simular_compras(sinais, ret, ret.index[0], ret.index[-1], horizonte=2, custo=0.01)
    valor = (1 + serie).cumprod()
    # dia 0: fatia 0 (0,5) compra A pagando 1% -> 0,495; fatia 1 em caixa (0,5)
    assert valor.iloc[0] == pytest.approx(0.995)
    # dia 1: A rende 1% -> 0,49995
    assert valor.iloc[1] == pytest.approx(0.99995)
    # dia 2: A rende 1% -> 0,5049495; fatia 0 vende pagando 1% -> 0,49990001
    assert valor.iloc[2] == pytest.approx(0.5049495 * 0.99 + 0.5)
    assert valor.iloc[3] == pytest.approx(valor.iloc[2])     # tudo em caixa


def test_sem_sinais_carteira_fica_parada():
    ret = _retornos()
    vazio = pd.DataFrame(columns=["data", "ticker", "prob", "sinal"])
    assert (simular_compras(vazio, ret, ret.index[0], ret.index[-1]) == 0).all()


def test_sinal_de_venda_nao_vira_compra():
    ret = _retornos()
    vendas = pd.DataFrame({"data": [ret.index[0]], "ticker": ["A"], "prob": [0.2], "sinal": ["venda"]})
    assert (simular_compras(vendas, ret, ret.index[0], ret.index[-1]) == 0).all()


def test_ticker_sem_retorno_conta_como_zero():
    ret = _retornos()
    sinais = pd.DataFrame({"data": [ret.index[0]], "ticker": ["SUMIU3"], "prob": [0.9], "sinal": ["compra"]})
    valor = (1 + simular_compras(sinais, ret, ret.index[0], ret.index[-1], horizonte=2, custo=0.0)).cumprod()
    assert valor.iloc[-1] == pytest.approx(1.0)
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_carteira.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/laboratorio/carteira.py`:

```python
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
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_carteira.py -q`
Expected: `4 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/laboratorio/carteira.py tests/test_carteira.py
git commit -m "Simulação da carteira de compras em 10 fatias com custo por lado

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 11: Métricas e agentes aleatórios

**Arquivos:**
- Criar: `src/laboratorio/metricas.py`
- Teste: `tests/test_metricas.py`

**Interfaces:**
- Consome: `Base.tabela` (colunas `data, ticker, excesso, rotulo`) e os sinais.
- Produz:
  - `brier(prob, rotulo) -> float`;
  - `prob_referencia(tabela, calendario, horizonte=config.HORIZONTE) -> pd.Series` (indexada pelo calendário);
  - `acerto_sinais(sinais, tabela, sinal: str) -> float`;
  - `acumulado(ret: pd.Series) -> float` e `pior_queda(ret) -> float`;
  - `resumo_carteira(ret, ret_bova) -> dict`, com as chaves
    `retorno_total, retorno_bova11, excesso_total, excesso_medio_anual, fracao_anos_bons, pior_queda`;
  - `bateu_aleatorios(sinais, tabela, n_agentes: int, rng) -> float`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_metricas.py`:

```python
"""Métricas do relatório e o teste contra a sorte."""

import numpy as np
import pandas as pd
import pytest

from src.laboratorio import metricas


def test_brier_casos_conhecidos():
    assert metricas.brier([1.0, 0.0], [1, 0]) == 0.0
    assert metricas.brier([0.5, 0.5], [1, 0]) == 0.25


def test_referencia_usa_so_rotulos_ja_conhecidos():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    tabela = pd.DataFrame({"data": np.repeat(cal, 2), "ticker": ["A", "B"] * 30, "rotulo": [1, 0] * 30})
    ref = metricas.prob_referencia(tabela, cal, horizonte=10)
    assert np.isnan(ref[cal[5]])                                  # nada conhecido ainda
    assert ref[cal[20]] == pytest.approx(0.5)
    bagunca = tabela.copy()
    bagunca.loc[bagunca["data"] > cal[10], "rotulo"] = 1          # muda rótulos ainda não conhecidos em cal[20]
    assert metricas.prob_referencia(bagunca, cal, horizonte=10)[cal[20]] == pytest.approx(0.5)


def test_acerto_de_compras_e_vendas():
    tabela = pd.DataFrame({"data": pd.Timestamp("2024-01-02"), "ticker": ["A", "B", "C"], "excesso": [0.02, -0.01, np.nan]})
    sinais = pd.DataFrame({"data": pd.Timestamp("2024-01-02"), "ticker": ["A", "B", "C"], "prob": 0.5,
                           "sinal": ["compra", "venda", "compra"]})
    assert metricas.acerto_sinais(sinais, tabela, "compra") == 1.0     # C sem resultado fica fora
    assert metricas.acerto_sinais(sinais, tabela, "venda") == 1.0


def test_resumo_da_carteira():
    dias = pd.to_datetime(["2023-06-01", "2023-06-02", "2024-06-03"])
    ret = pd.Series([0.10, -0.10, 0.05], index=dias)
    bova = pd.Series([0.0, 0.0, 0.10], index=dias)
    r = metricas.resumo_carteira(ret, bova)
    assert r["retorno_total"] == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert r["pior_queda"] == pytest.approx(-0.10)
    assert r["fracao_anos_bons"] == 0.0                           # 2023: -1% vs 0%; 2024: 5% vs 10%
    assert r["excesso_medio_anual"] == pytest.approx(((1.1 * 0.9 - 1) + (0.05 - 0.10)) / 2)


def _tabela_aleatoria(n_dias=60, n_acoes=30, semente=1):
    rng = np.random.default_rng(semente)
    dias = pd.bdate_range("2024-01-01", periods=n_dias)
    return pd.DataFrame([{"data": d, "ticker": f"T{i:02d}", "excesso": rng.normal(0, 0.05)}
                         for d in dias for i in range(n_acoes)])


def test_trapaceiro_bate_todos_os_aleatorios():
    tabela = _tabela_aleatoria()
    melhores = tabela.sort_values(["data", "excesso"], ascending=[True, False]).groupby("data").head(5)
    sinais = melhores.assign(prob=0.9, sinal="compra")[["data", "ticker", "prob", "sinal"]]
    assert metricas.bateu_aleatorios(sinais, tabela, 500, np.random.default_rng(0)) == 1.0


def test_escolha_ao_acaso_nao_parece_habilidade():
    """Uma escolha ao acaso supera, em média, ~metade dos agentes (média de 10 sorteios, para não depender de sorte)."""
    tabela = _tabela_aleatoria()
    fracoes = []
    for semente in range(10):
        sorteadas = tabela.groupby("data").sample(5, random_state=semente)
        sinais = sorteadas.assign(prob=0.9, sinal="compra")[["data", "ticker", "prob", "sinal"]]
        fracoes.append(metricas.bateu_aleatorios(sinais, tabela, 300, np.random.default_rng(100 + semente)))
    assert 0.2 < np.mean(fracoes) < 0.8


def test_sem_compras_nao_ha_teste():
    tabela = _tabela_aleatoria(n_dias=5)
    vazio = pd.DataFrame(columns=["data", "ticker", "prob", "sinal"])
    assert np.isnan(metricas.bateu_aleatorios(vazio, tabela, 100, np.random.default_rng(0)))
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_metricas.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/laboratorio/metricas.py`:

```python
"""
metricas.py
As métricas do relatório do laboratório e da prova final.

    brier             erro médio da probabilidade (menor é melhor; dizer sempre 50% dá 0,25)
    prob_referencia   probabilidade "boba": fração de rótulos 1 até D − 10 (só o que era conhecido)
    acerto_sinais     % das compras que superaram o BOVA11 (ou das vendas que ficaram para trás)
    resumo_carteira   retorno, excesso sobre o BOVA11, anos bons e pior queda
    bateu_aleatorios  % de agentes que, comprando o MESMO número de ações por dia sorteadas
                      do universo do dia, a linha superou (pelo excesso médio em 10 pregões)
"""

import numpy as np
import pandas as pd

import config


def brier(prob, rotulo) -> float:
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(rotulo, dtype=float)) ** 2))


def prob_referencia(tabela: pd.DataFrame, calendario: pd.DatetimeIndex,
                    horizonte: int = config.HORIZONTE) -> pd.Series:
    conhecidos = tabela.dropna(subset=["rotulo"])
    por_dia = (conhecidos.groupby("data")["rotulo"].agg(["sum", "count"])
                         .reindex(calendario, fill_value=0).cumsum())
    taxa = por_dia["sum"] / por_dia["count"].replace(0, np.nan)
    return taxa.shift(horizonte)  # no dia D, só vale o que era conhecido até D − horizonte


def acerto_sinais(sinais: pd.DataFrame, tabela: pd.DataFrame, sinal: str) -> float:
    m = (sinais[sinais["sinal"] == sinal]
         .merge(tabela[["data", "ticker", "excesso"]], on=["data", "ticker"])
         .dropna(subset=["excesso"]))
    if m.empty:
        return float("nan")
    certos = m["excesso"] > 0 if sinal == "compra" else m["excesso"] < 0
    return float(certos.mean())


def acumulado(ret: pd.Series) -> float:
    return float((1 + ret).prod() - 1)


def pior_queda(ret: pd.Series) -> float:
    curva = (1 + ret).cumprod()
    return float((curva / curva.cummax() - 1).min()) if len(curva) else float("nan")


def resumo_carteira(ret: pd.Series, ret_bova: pd.Series) -> dict:
    anos = ret.groupby(ret.index.year).apply(acumulado)
    anos_bova = ret_bova.groupby(ret_bova.index.year).apply(acumulado)
    excesso_anos = anos - anos_bova
    return {
        "retorno_total": acumulado(ret),
        "retorno_bova11": acumulado(ret_bova),
        "excesso_total": acumulado(ret) - acumulado(ret_bova),
        "excesso_medio_anual": float(excesso_anos.mean()) if len(excesso_anos) else float("nan"),
        "fracao_anos_bons": float((excesso_anos > 0).mean()) if len(excesso_anos) else float("nan"),
        "pior_queda": pior_queda(ret),
    }


def bateu_aleatorios(sinais: pd.DataFrame, tabela: pd.DataFrame, n_agentes: int, rng) -> float:
    conhecidos = tabela.dropna(subset=["excesso"])
    compras = (sinais[sinais["sinal"] == "compra"]
               .merge(conhecidos[["data", "ticker", "excesso"]], on=["data", "ticker"]))
    if compras.empty:
        return float("nan")
    excessos_do_dia = conhecidos.groupby("data")["excesso"].apply(np.asarray)
    nota_da_linha = compras.groupby("data")["excesso"].mean().mean()

    soma = np.zeros(n_agentes)
    dias = 0
    for dia, k in compras.groupby("data").size().items():
        opcoes = excessos_do_dia[dia]
        k = min(k, len(opcoes))
        sorteio = np.argsort(rng.random((n_agentes, len(opcoes))), axis=1)[:, :k]
        soma += opcoes[sorteio].mean(axis=1)
        dias += 1
    return float(np.mean(nota_da_linha > soma / dias))
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_metricas.py -q`
Expected: `7 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/laboratorio/metricas.py tests/test_metricas.py
git commit -m "Métricas do laboratório: Brier, referência sem vazamento, acertos, carteira e aleatórios

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 12: Laboratório (grade, tentativas, escolha e relatório)

**Arquivos:**
- Criar: `src/laboratorio/executar.py`
- Teste: `tests/test_laboratorio.py`

**Interfaces:**
- Consome: Tarefas 7 a 11.
- Produz:
  - `recortar_laboratorio(base: Base) -> Base`: tira tudo de `RESERVA_INICIO` em diante e apaga o
    rótulo e o excesso dos dias cujo resultado cairia na reserva;
  - `avaliar_probabilidades(probs, base) -> dict`, com as chaves `brier, brier_referencia, previsoes`;
  - `avaliar_configuracao(base, probs, tipo, k, p_min, inicio, fim_sinais, fim_carteira, rng) -> dict`,
    com as chaves `tipo, k, p_min`, as de `resumo_carteira` e mais
    `acerto_compras, acerto_vendas, bateu_aleatorios, compras_por_dia`;
  - `executar_laboratorio(base: Base) -> tuple[list[dict], dict, dict]`, que devolve
    `(linhas, escolha, resumo_probs)`;
  - `escolher(brier_por_tipo: dict, linhas: list[dict]) -> dict`, com as chaves `tipo, k, p_min`;
  - `registrar_tentativa(linha: dict) -> None` e `rodar() -> None`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_laboratorio.py`:

```python
"""O laboratório: nunca lê a reserva, registra cada tentativa e escolhe pela regra da especificação."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.base import montar_base
from src.laboratorio import executar
from tests.conftest import gerar_precos, universo_completo


@pytest.fixture
def lab_rapido(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LAB_INICIO", "2016-01-01")
    monkeypatch.setattr(config, "RESERVA_INICIO", "2017-01-01")
    monkeypatch.setattr(config, "N_ALEATORIOS", 50)
    monkeypatch.setattr(config, "GRADE_K", (3,))
    monkeypatch.setattr(config, "GRADE_P_MIN", (0.50, 0.55))
    monkeypatch.setattr(config, "ARQUIVO_TENTATIVAS", tmp_path / "tentativas.csv")


def _base(precos, cal):
    return montar_base(precos, universo_completo(precos, cal), cal)


def test_recorte_tira_a_reserva_e_os_rotulos_que_dependem_dela(lab_rapido, base_sintetica):
    b = executar.recortar_laboratorio(base_sintetica)
    corte = pd.Timestamp("2017-01-01")
    assert b.tabela["data"].max() < corte and b.calendario[-1] < corte and b.retornos.index[-1] < corte
    assert b.tabela.loc[b.tabela["data"] > b.calendario[-11], "rotulo"].isna().all()


def test_mudar_a_reserva_nao_muda_nada_no_laboratorio(lab_rapido):
    precos, cal = gerar_precos(n_acoes=12, fim="2017-06-30")
    alterados = precos.copy()
    na_reserva = alterados["data"] >= pd.Timestamp("2017-01-01")
    alterados.loc[na_reserva, "fechamento_ajustado"] *= np.random.default_rng(5).uniform(0.5, 2.0, na_reserva.sum())

    linhas_a, escolha_a, _ = executar.executar_laboratorio(_base(precos, cal))
    linhas_b, escolha_b, _ = executar.executar_laboratorio(_base(alterados, cal))
    assert escolha_a == escolha_b
    pd.testing.assert_frame_equal(pd.DataFrame(linhas_a), pd.DataFrame(linhas_b))


def test_cada_configuracao_vira_uma_tentativa_registrada(lab_rapido):
    precos, cal = gerar_precos(n_acoes=12, fim="2017-03-31")
    linhas, _, _ = executar.executar_laboratorio(_base(precos, cal))
    for linha in linhas:
        executar.registrar_tentativa(linha)
    registradas = pd.read_csv(config.ARQUIVO_TENTATIVAS)
    assert len(registradas) == len(linhas) == 2 * 1 * 2       # 2 modelos x 1 k x 2 p_min
    assert {"registrado_em", "tipo", "k", "p_min", "brier", "excesso_medio_anual"} <= set(registradas.columns)


def test_escolhe_primeiro_pelo_brier_depois_pelo_excesso():
    linhas = [
        {"tipo": "logistica", "k": 3, "p_min": 0.5, "excesso_medio_anual": 0.01},
        {"tipo": "logistica", "k": 5, "p_min": 0.55, "excesso_medio_anual": 0.03},
        {"tipo": "boosting", "k": 3, "p_min": 0.5, "excesso_medio_anual": 0.50},   # melhor excesso, pior Brier
    ]
    escolha = executar.escolher({"logistica": 0.249, "boosting": 0.251}, linhas)
    assert escolha == {"tipo": "logistica", "k": 5, "p_min": 0.55}


def test_excesso_vazio_nunca_e_escolhido():
    linhas = [{"tipo": "logistica", "k": 3, "p_min": 0.6, "excesso_medio_anual": float("nan")},
              {"tipo": "logistica", "k": 5, "p_min": 0.5, "excesso_medio_anual": -0.02}]
    assert executar.escolher({"logistica": 0.25}, linhas)["k"] == 5
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_laboratorio.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar**

`src/laboratorio/executar.py`:

```python
"""
executar.py
Etapa "python main.py laboratorio": testa a linha de LAB_INICIO até o fim de LAB_FIM.

    1. Recorta a base: nada de RESERVA_INICIO em diante entra aqui (nem preço, nem rótulo).
    2. Para cada modelo: probabilidades com walk-forward mensal e embargo; Brier.
    3. Para cada k e p_min da grade FIXA: sinais -> carteira com custos -> métricas.
    4. Toda configuração testada vai para resultados/laboratorio/tentativas.csv.
    5. Escolha (especificação 7): menor Brier; dentro dele, maior excesso médio anual.
    6. Relatório em resultados/laboratorio/relatorio.md e escolha em escolha.json.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.base import Base, carregar_base
from src.laboratorio.carteira import simular_compras
from src.laboratorio.metricas import (acerto_sinais, bateu_aleatorios, brier, prob_referencia,
                                      resumo_carteira)
from src.laboratorio.walk_forward import gerar_probabilidades, ultimo_dia_avaliavel
from src.sinais import gerar_sinais


def recortar_laboratorio(base: Base) -> Base:
    corte = pd.Timestamp(config.RESERVA_INICIO)
    calendario = base.calendario[base.calendario < corte]
    tabela = base.tabela[base.tabela["data"] < corte].copy()
    ultimo = ultimo_dia_avaliavel(calendario, calendario[-1])
    depende_da_reserva = tabela["data"] > ultimo if ultimo is not None else slice(None)
    tabela.loc[depende_da_reserva, ["excesso", "rotulo"]] = np.nan
    return Base(tabela=tabela, calendario=calendario, retornos=base.retornos[base.retornos.index < corte])


def avaliar_probabilidades(probs: pd.DataFrame, base: Base) -> dict:
    m = probs.merge(base.tabela[["data", "ticker", "rotulo"]], on=["data", "ticker"]).dropna(subset=["rotulo"])
    referencia = m["data"].map(prob_referencia(base.tabela, base.calendario))
    ok = referencia.notna()
    return {
        "brier": brier(m.loc[ok, "prob"], m.loc[ok, "rotulo"]),
        "brier_referencia": brier(referencia[ok], m.loc[ok, "rotulo"]),
        "previsoes": int(ok.sum()),
    }


def avaliar_configuracao(base: Base, probs: pd.DataFrame, tipo: str, k: int, p_min: float,
                         inicio, fim_sinais, fim_carteira, rng) -> dict:
    sinais = gerar_sinais(probs[probs["data"] <= pd.Timestamp(fim_sinais)], k, p_min)
    retorno = simular_compras(sinais, base.retornos, inicio, fim_carteira)
    retorno_bova = base.retornos[config.REFERENCIA].reindex(retorno.index).fillna(0.0)
    dias = max(probs["data"].nunique(), 1)
    return {
        "tipo": tipo, "k": k, "p_min": p_min,
        **resumo_carteira(retorno, retorno_bova),
        "acerto_compras": acerto_sinais(sinais, base.tabela, "compra"),
        "acerto_vendas": acerto_sinais(sinais, base.tabela, "venda"),
        "bateu_aleatorios": bateu_aleatorios(sinais, base.tabela, config.N_ALEATORIOS, rng),
        "compras_por_dia": float((sinais["sinal"] == "compra").sum() / dias),
    }


def escolher(brier_por_tipo: dict, linhas: list[dict]) -> dict:
    tipo = min(brier_por_tipo, key=brier_por_tipo.get)

    def nota(linha):
        valor = linha["excesso_medio_anual"]
        return -math.inf if valor is None or (isinstance(valor, float) and math.isnan(valor)) else valor

    melhor = max((l for l in linhas if l["tipo"] == tipo), key=nota)
    return {"tipo": tipo, "k": int(melhor["k"]), "p_min": float(melhor["p_min"])}


def executar_laboratorio(base: Base) -> tuple[list[dict], dict, dict]:
    base = recortar_laboratorio(base)
    inicio = pd.Timestamp(config.LAB_INICIO)
    fim_carteira = base.calendario[-1]
    fim_sinais = ultimo_dia_avaliavel(base.calendario, fim_carteira)

    linhas, resumo_probs = [], {}
    for indice, tipo in enumerate(config.TIPOS_MODELO):
        print(f"Laboratório: modelo '{tipo}' com retreino mensal (pode levar alguns minutos)...")
        probs = gerar_probabilidades(base.tabela, base.calendario, inicio, fim_sinais, tipo)
        resumo_probs[tipo] = avaliar_probabilidades(probs, base)
        for k in config.GRADE_K:
            for p_min in config.GRADE_P_MIN:
                rng = np.random.default_rng([config.SEMENTE, indice, k, int(round(p_min * 100))])
                linha = avaliar_configuracao(base, probs, tipo, k, p_min, inicio, fim_sinais, fim_carteira, rng)
                linhas.append({**linha, **resumo_probs[tipo]})
    escolha = escolher({t: r["brier"] for t, r in resumo_probs.items()}, linhas)
    return linhas, escolha, resumo_probs


def registrar_tentativa(linha: dict) -> None:
    arquivo = config.ARQUIVO_TENTATIVAS
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    nova = pd.DataFrame([{"registrado_em": datetime.now().isoformat(timespec="seconds"), **linha}])
    tabela = pd.concat([pd.read_csv(arquivo), nova], ignore_index=True) if arquivo.exists() else nova
    tabela.to_csv(arquivo, index=False, float_format="%.6f")


def _pct(valor) -> str:
    return "-" if valor is None or (isinstance(valor, float) and math.isnan(valor)) else f"{valor:+.1%}"


def _relatorio(linhas: list[dict], escolha: dict, resumo_probs: dict) -> str:
    texto = ["# Relatório do laboratório", "",
             f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. Período: {config.LAB_INICIO} a {config.LAB_FIM} "
             "(a reserva de 2025+ não foi lida).", "",
             "## Probabilidades (Brier: menor é melhor)", "",
             "| modelo | Brier | Brier da taxa histórica | previsões |", "|---|---|---|---|"]
    for tipo, r in resumo_probs.items():
        texto.append(f"| {tipo} | {r['brier']:.4f} | {r['brier_referencia']:.4f} | {r['previsoes']} |")
    texto += ["", "## Carteira de compras (custo de 0,10% por lado)", "",
              "| modelo | k | p_min | retorno | BOVA11 | excesso total | excesso médio/ano | anos bons | "
              "pior queda | acerto compras | acerto vendas | bateu aleatórios | compras/dia |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in linhas:
        texto.append(
            f"| {l['tipo']} | {l['k']} | {l['p_min']:.2f} | {_pct(l['retorno_total'])} | {_pct(l['retorno_bova11'])} | "
            f"{_pct(l['excesso_total'])} | {_pct(l['excesso_medio_anual'])} | {l['fracao_anos_bons']:.0%} | "
            f"{_pct(l['pior_queda'])} | {_pct(l['acerto_compras'])} | {_pct(l['acerto_vendas'])} | "
            f"{_pct(l['bateu_aleatorios'])} | {l['compras_por_dia']:.1f} |")
    cobertura = "-"
    if config.ARQUIVO_COBERTURA.exists():
        cobertura = f"{pd.read_csv(config.ARQUIVO_COBERTURA)['cobertura'].mean():.1%}"
    texto += ["", f"**Escolha:** modelo `{escolha['tipo']}`, k = {escolha['k']}, p_min = {escolha['p_min']:.2f} "
              "(menor Brier; dentro dele, maior excesso médio anual).", "",
              f"**Cobertura média do universo com preço ajustado:** {cobertura}.", "",
              f"Tentativas registradas até agora: ver `{config.ARQUIVO_TENTATIVAS.name}`.", "",
              "> Os sinais continuam **só para leitura** até a prova final (`python main.py prova-final`)."]
    return "\n".join(texto) + "\n"


def rodar() -> None:
    linhas, escolha, resumo_probs = executar_laboratorio(carregar_base())
    for linha in linhas:
        registrar_tentativa(linha)
    config.PASTA_LABORATORIO.mkdir(parents=True, exist_ok=True)
    config.ARQUIVO_ESCOLHA.write_text(json.dumps(
        {**escolha, "escolhido_em": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    config.ARQUIVO_RELATORIO_LAB.write_text(_relatorio(linhas, escolha, resumo_probs), encoding="utf-8")
    print(f"\nEscolha: {escolha}")
    print(f"Relatório: {config.ARQUIVO_RELATORIO_LAB.relative_to(config.RAIZ)}")
```

- [ ] **Passo 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_laboratorio.py -q`
Expected: `5 passed`.

- [ ] **Passo 5: Commit**

```bash
git add src/laboratorio/executar.py tests/test_laboratorio.py
git commit -m "Laboratório: grade fixa, tentativas registradas, escolha e relatório sem ler a reserva

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 13: Prova final, sinais do dia e main.py

**Arquivos:**
- Criar: `src/laboratorio/prova_final.py`, `src/hoje.py`, `main.py`
- Teste: `tests/test_prova_e_sinais.py`

**Interfaces:**
- Consome: Tarefas 7 a 12.
- Produz:
  - `prova_final.ErroProvaFinal` (exceção);
  - `prova_final.criterios(linha: dict) -> dict[str, bool]`;
  - `prova_final.rodar(base: Base | None = None) -> dict`, que devolve o resultado gravado;
  - `hoje.escolha_atual() -> dict`;
  - `hoje.calcular(base, escolha, treinar=combinador.treinar) -> tuple[pd.DataFrame, pd.Timestamp]`;
  - `hoje.rodar() -> None`;
  - `main.py`, com os comandos `atualizar`, `laboratorio`, `sinais` e `prova-final`.

- [ ] **Passo 1: Escrever os testes**

`tests/test_prova_e_sinais.py`:

```python
"""Prova final (uma vez só) e sinais do próximo pregão."""

import json

import pandas as pd
import pytest

import config
from src import hoje
from src.laboratorio import prova_final, walk_forward as wf
from src.modelos import combinador


@pytest.fixture
def arquivos(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQUIVO_PROVA_FINAL", tmp_path / "prova_final.json")
    monkeypatch.setattr(config, "ARQUIVO_RELATORIO_PROVA", tmp_path / "prova_final.md")
    monkeypatch.setattr(config, "ARQUIVO_ESCOLHA", tmp_path / "escolha.json")
    monkeypatch.setattr(config, "RESERVA_INICIO", "2017-07-01")
    monkeypatch.setattr(config, "N_ALEATORIOS", 50)
    return tmp_path


def _gravar_escolha():
    config.ARQUIVO_ESCOLHA.write_text(json.dumps({"tipo": "logistica", "k": 3, "p_min": 0.5}), encoding="utf-8")


def test_sem_escolha_do_laboratorio_recusa(arquivos, base_sintetica):
    with pytest.raises(prova_final.ErroProvaFinal, match="laboratorio"):
        prova_final.rodar(base_sintetica)


def test_roda_uma_vez_e_recusa_a_segunda(arquivos, base_sintetica):
    _gravar_escolha()
    resultado = prova_final.rodar(base_sintetica)
    assert config.ARQUIVO_PROVA_FINAL.exists() and config.ARQUIVO_RELATORIO_PROVA.exists()
    assert set(resultado["criterios"]) == {"excesso_total_positivo", "bateu_aleatorios", "acerto_compras"}
    assert resultado["aprovado"] == all(resultado["criterios"].values())
    assert resultado["periodo"][0] == "2017-07-03"                 # primeiro pregão da reserva
    with pytest.raises(prova_final.ErroProvaFinal, match="já foi rodada"):
        prova_final.rodar(base_sintetica)


def test_criterios_exigem_os_tres():
    ok = {"excesso_total": 0.01, "bateu_aleatorios": 0.95, "acerto_compras": 0.55}
    assert all(prova_final.criterios(ok).values())
    assert not prova_final.criterios({**ok, "bateu_aleatorios": 0.89})["bateu_aleatorios"]
    assert not prova_final.criterios({**ok, "acerto_compras": float("nan")})["acerto_compras"]


def test_sinais_do_dia_so_treinam_com_o_que_ja_era_conhecido(base_sintetica):
    vistos = []

    def treinar_espiao(x, y, tipo):
        vistos.append(len(x))
        return combinador.treinar(x, y, tipo)

    sinais, dia = hoje.calcular(base_sintetica, {"tipo": "logistica", "k": 3, "p_min": 0.5}, treinar=treinar_espiao)
    b = base_sintetica
    limite = wf.limite_de_treino(b.calendario, dia)
    assert dia == b.tabela["data"].max()
    assert vistos == [len(b.tabela[(b.tabela["data"] <= limite) & b.tabela["rotulo"].notna()])]
    assert set(sinais["data"]) <= {dia}


def test_sem_escolha_usa_o_padrao(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARQUIVO_ESCOLHA", tmp_path / "nao_existe.json")
    assert hoje.escolha_atual() == {"tipo": "logistica", "k": config.K_PADRAO, "p_min": config.P_MIN_PADRAO}
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_prova_e_sinais.py -q`
Expected: FAIL com `ImportError`.

- [ ] **Passo 3: Implementar a prova final**

`src/laboratorio/prova_final.py`:

```python
"""
prova_final.py
Etapa "python main.py prova-final": abre a reserva (RESERVA_INICIO em diante) UMA VEZ, com a
configuração escolhida pelo laboratório, e aplica os critérios definidos ANTES de qualquer resultado:

    1. excesso líquido total da carteira de compras sobre o BOVA11 > 0
    2. superar pelo menos 90% dos agentes aleatórios
    3. acerto das compras > 50%

Se resultados/prova_final.json já existir, a etapa se recusa a rodar. Rodar de novo exige
apagar o arquivo, e isso fica visível no git.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.base import Base, carregar_base
from src.laboratorio.executar import avaliar_configuracao, avaliar_probabilidades
from src.laboratorio.walk_forward import gerar_probabilidades, ultimo_dia_avaliavel


class ErroProvaFinal(Exception):
    """A prova final não pode rodar agora (já rodou, falta o laboratório, ou faltam pregões)."""


def _numero(valor):
    """NaN vira None no JSON (e conta como critério NÃO cumprido)."""
    return None if isinstance(valor, float) and math.isnan(valor) else valor


def criterios(linha: dict) -> dict[str, bool]:
    def maior(chave, limite, igual=False):
        valor = linha.get(chave)
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            return False
        return valor >= limite if igual else valor > limite

    return {
        "excesso_total_positivo": maior("excesso_total", config.CRITERIO_EXCESSO_MINIMO),
        "bateu_aleatorios": maior("bateu_aleatorios", config.CRITERIO_ALEATORIOS, igual=True),
        "acerto_compras": maior("acerto_compras", config.CRITERIO_ACERTO_COMPRAS),
    }


def rodar(base: Base | None = None) -> dict:
    if config.ARQUIVO_PROVA_FINAL.exists():
        raise ErroProvaFinal(f"a prova final já foi rodada ({config.ARQUIVO_PROVA_FINAL.name} existe). "
                             "Rodar de novo exige apagar o arquivo, e isso fica registrado no git.")
    if not config.ARQUIVO_ESCOLHA.exists():
        raise ErroProvaFinal("não há escolha do laboratório. Rode antes: python main.py laboratorio")
    escolha = json.loads(config.ARQUIVO_ESCOLHA.read_text(encoding="utf-8"))
    base = base or carregar_base()

    inicio_pedido = pd.Timestamp(config.RESERVA_INICIO)
    na_reserva = base.calendario[base.calendario >= inicio_pedido]
    fim_carteira = base.calendario[-1]
    fim_sinais = ultimo_dia_avaliavel(base.calendario, fim_carteira)
    if len(na_reserva) == 0 or fim_sinais is None or fim_sinais < na_reserva[0]:
        raise ErroProvaFinal("ainda não há pregões suficientes na reserva para avaliar.")
    inicio = na_reserva[0]

    print(f"Prova final: modelo '{escolha['tipo']}', k = {escolha['k']}, p_min = {escolha['p_min']} "
          f"| reserva {inicio:%d/%m/%Y} a {fim_sinais:%d/%m/%Y}")
    probs = gerar_probabilidades(base.tabela, base.calendario, inicio, fim_sinais, escolha["tipo"])
    rng = np.random.default_rng(config.SEMENTE)
    linha = {**avaliar_configuracao(base, probs, escolha["tipo"], escolha["k"], escolha["p_min"],
                                    inicio, fim_sinais, fim_carteira, rng),
             **avaliar_probabilidades(probs, base)}
    resultado_criterios = criterios(linha)
    resultado = {
        "rodado_em": datetime.now().isoformat(timespec="seconds"),
        "periodo": [f"{inicio:%Y-%m-%d}", f"{fim_sinais:%Y-%m-%d}"],
        "escolha": escolha,
        "metricas": {chave: _numero(valor) for chave, valor in linha.items()},
        "criterios": resultado_criterios,
        "aprovado": all(resultado_criterios.values()),
    }
    config.ARQUIVO_PROVA_FINAL.parent.mkdir(parents=True, exist_ok=True)
    config.ARQUIVO_PROVA_FINAL.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    config.ARQUIVO_RELATORIO_PROVA.write_text(_relatorio(resultado), encoding="utf-8")

    veredito = "APROVADA para uso com cautela" if resultado["aprovado"] else "NÃO aprovada"
    print(f"\nResultado: {veredito}")
    for nome, ok in resultado_criterios.items():
        print(f"  {'OK ' if ok else 'NÃO'}  {nome}")
    return resultado


def _relatorio(r: dict) -> str:
    m = r["metricas"]
    pct = lambda v: "-" if v is None else f"{v:+.1%}"
    linhas = [
        "# Prova final (reserva)", "",
        f"Rodada em {r['rodado_em']}. Período: {r['periodo'][0]} a {r['periodo'][1]}.",
        f"Configuração: modelo `{r['escolha']['tipo']}`, k = {r['escolha']['k']}, p_min = {r['escolha']['p_min']}.", "",
        f"- Excesso líquido total sobre o BOVA11: {pct(m.get('excesso_total'))} "
        f"({'OK' if r['criterios']['excesso_total_positivo'] else 'não'})",
        f"- Agentes aleatórios superados: {pct(m.get('bateu_aleatorios'))} "
        f"({'OK' if r['criterios']['bateu_aleatorios'] else 'não'})",
        f"- Acerto das compras: {pct(m.get('acerto_compras'))} ({'OK' if r['criterios']['acerto_compras'] else 'não'})",
        f"- Brier: {m.get('brier')} (taxa histórica: {m.get('brier_referencia')})", "",
        f"**Resultado: {'APROVADA para uso com cautela' if r['aprovado'] else 'NÃO aprovada'}.**", "",
        "> A reserva tem poucos meses: um resultado positivo é um sinal, não uma prova definitiva.",
    ]
    return "\n".join(linhas) + "\n"
```

- [ ] **Passo 4: Implementar os sinais do dia**

`src/hoje.py`:

```python
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
```

- [ ] **Passo 5: `main.py`**

```python
"""
main.py
Linha de produção de sinais para swing trade na B3 (etapa 1: só preço e volume).

    python main.py atualizar     # baixa B3 + Yahoo, monta o universo (precisa de internet)
    python main.py laboratorio   # testa a linha de 2016 a 2024 e escolhe a configuração
    python main.py sinais        # compras e vendas para o próximo pregão (SÓ PARA LEITURA)
    python main.py prova-final   # abre a reserva de 2025+ UMA vez e aplica os critérios
"""

import argparse
import sys

# Garante acentos corretos no terminal do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def comando_atualizar() -> None:
    from src import atualizar
    atualizar.rodar()


def comando_laboratorio() -> None:
    from src.laboratorio import executar
    executar.rodar()


def comando_sinais() -> None:
    from src import hoje
    hoje.rodar()


def comando_prova_final() -> None:
    from src.laboratorio import prova_final
    try:
        prova_final.rodar()
    except prova_final.ErroProvaFinal as erro:
        print(f"Prova final não rodou: {erro}")


COMANDOS = {
    "atualizar": comando_atualizar,
    "laboratorio": comando_laboratorio,
    "sinais": comando_sinais,
    "prova-final": comando_prova_final,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Linha de produção de sinais (B3, swing trade)")
    parser.add_argument("comando", choices=list(COMANDOS), help="o que rodar")
    args = parser.parse_args()
    COMANDOS[args.comando]()


if __name__ == "__main__":
    main()
```

- [ ] **Passo 6: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_prova_e_sinais.py -q`
Expected: `5 passed`.

Run: `.venv/Scripts/python main.py --help`
Expected: a ajuda lista `{atualizar,laboratorio,sinais,prova-final}`.

- [ ] **Passo 7: Commit**

```bash
git add src/laboratorio/prova_final.py src/hoje.py main.py tests/test_prova_e_sinais.py
git commit -m "Prova final (uma vez, critérios fixos), sinais do próximo pregão e main.py

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Tarefa 14: README e verificação com dados reais

**Arquivos:**
- Modificar: `README.md` (reescrever)
- Gerados (no git): `resultados/universo/*.csv`, `resultados/laboratorio/*`

- [ ] **Passo 1: Reescrever o README**

Substituir `README.md` por:

````markdown
# Linha de produção de sinais: B3, swing trade

> **In English (short):** a personal tool that generates buy/sell signals for swing trading the
> ~100 most liquid Brazilian stocks. Code, an LLM, Jev (TypeSafe's "System One" model) and a
> classic ML model work together, each where it is strongest. Stage 1 uses price and volume only,
> with a leakage-free lab (2016–2024) and a locked final test (2025+).

## O que é

Uma ferramenta pessoal que, todo dia após o fechamento, aponta ações da B3 com mais chance de
**superar o Ibovespa nos próximos 10 pregões** (compra) e com mais chance de **ficar para trás**
(venda, se você tiver).

A ideia é uma linha de produção, cada peça no seu melhor posto:

| Peça | Papel | Etapa |
|---|---|---|
| Código | coleta, datas, contas, custos | 1 |
| ML | combina os sinais e aprende no histórico quanto cada um vale | 1 |
| LLM | lê e organiza documentos longos (fatos relevantes da CVM) | 3 |
| Jev | responde perguntas sim/não estreitas sobre esses documentos, com probabilidade | 3 |

**Etapas:**
1. preços + ML + laboratório (**esta**);
2. documentos da CVM;
3. LLM + Jev viram features;
4. e-mail diário no GitHub Actions;
5. notícias.

## Aviso

**Os sinais são só para leitura** até a prova final (reserva de 2025+) ter resultado positivo.
Mesmo aprovada, a ferramenta não garante ganho: ela estima probabilidades a partir do passado.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                 # testes, sem internet

python main.py atualizar         # 1ª vez: alguns minutos (~10 anos de dados da B3)
python main.py laboratorio       # testa 2016–2024 e escolhe a configuração
python main.py sinais            # compras e vendas para o próximo pregão
python main.py prova-final       # UMA vez, quando o laboratório estiver revisado
```

## Como a linha evita se enganar

- **Universo sem viés de sobrevivência:** a cada mês, as 100 ações mais líquidas pelo volume dos
  3 meses **anteriores**, usando as cotações oficiais da B3 (que incluem empresas que deixaram de
  existir).
- **Nada do futuro:** as features usam só dados até o fechamento do dia. O treino tem **embargo de
  10 pregões**, porque o resultado de um dia só é conhecido 10 pregões depois.
- **Custos:** 0,10% por lado (custos + margem para slippage).
- **Teste contra a sorte:** 1.000 agentes que compram o mesmo número de ações sorteadas.
- **Ajuste limitado e registrado:** no máximo 18 configurações, todas em
  `resultados/laboratorio/tentativas.csv`.
- **Reserva trancada:** 2025 em diante só é lido uma vez, pela prova final, com critérios definidos
  antes (ver especificação).

## Documentos

- Especificação: `docs/superpowers/specs/2026-10-09-linha-etapa1-design.md`
- Plano de implementação: `docs/superpowers/plans/2026-10-09-linha-etapa1.md`
- Experimento anterior (Jev vs LLM vs ML), encerrado: `arquivo/experimento_jev_llm_ml/`

## Limitações

- Execução simplificada: compra e venda no preço de fechamento.
- Ações sem preço ajustado no Yahoo ficam de fora (ver cobertura em `resultados/universo/cobertura.csv`).
- Tickers reaproveitados por outra empresa são só relatados (`tickers_com_troca_de_nome.csv`).
- A reserva tem poucos meses: um resultado positivo é um sinal, não uma prova definitiva.
````

- [ ] **Passo 2: Rodar todos os testes**

Run: `.venv/Scripts/python -m pytest -q`
Expected: todos passam (cerca de 70 testes).

- [ ] **Passo 3: Verificação com dados reais (internet)**

Run: `.venv/Scripts/python main.py atualizar`
Expected:
- ~12 arquivos da B3 baixados ou lidos;
- universo com ~130 meses;
- cobertura média informada (anotar o valor);
- nenhum traceback.

Run: `.venv/Scripts/python main.py laboratorio`
Expected:
- 18 tentativas registradas;
- `resultados/laboratorio/relatorio.md` e `escolha.json` criados;
- leva de 5 a 20 minutos.

Run: `.venv/Scripts/python main.py sinais`
Expected: aviso de "só leitura" e até k compras e k vendas listadas.

**Não rodar `prova-final`**: ela fica para o usuário, depois de revisar o relatório.

- [ ] **Passo 4: Commit**

```bash
git add README.md resultados/universo resultados/laboratorio
git commit -m "README da linha de produção e primeira execução do laboratório (etapa 1)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
