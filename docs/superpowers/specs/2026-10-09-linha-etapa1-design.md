# Linha de produção de sinais (B3, swing trade): Etapa 1

**Data:** 09/10/2026 · **Status:** em revisão

## 1. Contexto e objetivo

O projeto deixa de ser um experimento que compara Jev, LLM e ML e passa a ser uma
**ferramenta pessoal de sinais de compra e venda** para swing trade na B3. Nela, as três
tecnologias trabalham juntas, cada uma no posto em que é melhor:

| Peça | Papel na linha | Por quê |
|---|---|---|
| Código | coleta, datas, contas, custos, regras | é determinístico e confiável |
| LLM | ler e organizar textos longos (etapa 3) | é bom com linguagem, mas ruim e confiante demais com números |
| Jev | responder perguntas estreitas sobre os textos, com probabilidade (etapa 3) | é rápido, barato, estável e calibrado em perguntas sim/não |
| ML | combinar todos os sinais e aprender no histórico quanto cada um vale | é o único que aprende com dados |

Se a ferramenta se mostrar útil, ela pode virar produto. Por isso as peças têm fronteiras
claras desde o início.

### Decisões tomadas com o usuário

| Tema | Decisão |
|---|---|
| Uso | ferramenta pessoal para decisões reais; possível produto no futuro |
| Horizonte | swing trade (dias a poucas semanas) |
| Ações | as ~100 maiores e mais líquidas (aproximação do IBrX-100) |
| Operação | só compra; "venda" quer dizer "deve ficar para trás", não short |
| Memória de carteira | não; o usuário cruza os sinais de venda com o que tem |
| Pergunta | "esta ação vai render mais que o Ibovespa nos próximos 10 pregões?" |
| Entrega (etapa 4) | e-mail diário após o fechamento |
| Onde roda (etapa 4) | GitHub Actions, repositório privado (cota grátis de 2.000 min/mês) |
| Experimento antigo | encerrado e arquivado (Desvio 2) |

### Etapas do projeto (cada uma tem sua própria especificação)

1. **Preços IBrX-100 + ML de swing + laboratório** (esta especificação)
2. Coleta de documentos da CVM (fatos relevantes, comunicados)
3. LLM + Jev transformando documentos em features, testadas no laboratório
4. E-mail diário + GitHub Actions
5. Notícias de portais, ao vivo

## 2. Arquitetura geral (vale para todas as etapas)

```
src/
  coleta/        precos.py (etapa 1) · cvm.py (etapa 2) · noticias.py (etapa 5)
  universo.py    quais ações estão na lista em cada mês
  features/      preco.py (etapa 1) · textos.py (etapa 3)
  modelos/       combinador.py (ML) · jev.py · llm.py · api_comum.py
  alvo.py        rótulo: superou o Ibovespa em 10 pregões?
  sinais.py      probabilidades -> compra / venda / nada
  laboratorio/   walk-forward, simulação de carteira, custos, agentes aleatórios, relatório
  entrega/       email.py (etapa 4)
main.py          python main.py atualizar | laboratorio | sinais | prova-final
```

**Regras que valem para a linha inteira**

1. **Momento em que a informação ficou conhecida.** Toda informação carrega a data e a hora
   em que ficou pública. O sinal do dia D só usa o que era conhecido até o fechamento de D
   (17h). Um documento publicado depois das 17h de D conta para D+1.
2. **Peças conversam por tabelas com colunas fixas.** As features entregam ao combinador uma
   tabela no formato longo (`data, ticker, feature, valor`). Trocar uma peça não quebra as outras.
3. **O que custa dinheiro é guardado e versionado** (respostas de LLM e Jev, a partir da etapa 3).
   O que é grátis (preços) é baixado de novo.
4. **Nada vira sinal sem passar pelo laboratório.** Uma peça nova só entra se melhorar o
   resultado no passado, com custos e contra os agentes aleatórios.

## 3. Etapa 1: escopo

**Entrega:**
- `python main.py atualizar`: baixa e atualiza os dados (B3 + Yahoo).
- `python main.py laboratorio`: testa a linha de 2016 a 2024 e gera um relatório.
- `python main.py sinais`: compras e vendas para o próximo pregão, no terminal.
- `python main.py prova-final`: abre a reserva (2025 em diante) uma única vez.

**Fora do escopo:** documentos e notícias, Jev e LLM, e-mail, GitHub Actions, memória de carteira e short.

**Aviso de uso:** os sinais da etapa 1 são **só para leitura** até a prova final ter resultado
positivo (seção 9).

## 4. Dados

### 4.1 Fontes
- **B3, séries históricas (COTAHIST):** `https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP`
  (download confirmado em 09/10/2026). Têm todas as ações negociadas, inclusive as que deixaram
  de existir, com volume financeiro. Os preços **não** são ajustados por proventos.
  Uso: definir o universo de cada mês.
- **Yahoo Finance (yfinance):** preços ajustados (`Adj Close`) das ações e do BOVA11.
  Uso: retornos, features e alvo.

### 4.2 Período
- Download a partir de **01/01/2015**, para ter 120 pregões de histórico e 3 meses de liquidez
  antes de 2016.
- **Laboratório:** 01/01/2016 a 31/12/2024.
- **Reserva (prova final):** 01/01/2025 até o último pregão completo. Nenhuma etapa de ajuste
  pode ler esse período.

### 4.3 Armazenamento
- `dados/b3/` e `dados/yahoo/`: arquivos brutos (fora do git).
- `dados/processados/`: tabelas prontas em CSV compactado (`.csv.gz`, fora do git).
- `resultados/universo/universo_mensal.csv`: a lista de cada mês (no git).
- `resultados/laboratorio/`: relatórios (no git).

### 4.4 Qualidade
- Linha de pregão incompleto (abertura, máxima e mínima zeradas) é descartada, como já acontece hoje.
- O relatório de cobertura informa, por mês, quantas ações do universo têm preço ajustado no
  Yahoo. Ações sem dados no Yahoo **ficam fora** da etapa 1 e são listadas.
- **Limite:** se a cobertura média ficar abaixo de 95% das ações-mês do universo, isso é
  tratado como risco aberto e investigado antes da prova final (seção 12).

## 5. Universo (aproximação do IBrX-100 sem viés de sobrevivência)

No **primeiro pregão de cada mês**, usando só dados da B3 até o pregão anterior:
1. **Considerar só ações à vista no lote padrão:** ON, PN, PNA, PNB e units. Ficam de fora
   BDRs, ETFs, FIIs, recibos e direitos.
2. **Uma por empresa:** quando a mesma empresa (mesma raiz de 4 letras do ticker) tem mais de
   uma classe, fica a mais líquida.
3. **Ordenar** pelo volume financeiro médio diário dos 63 pregões anteriores.
4. **Ficar com as 100 primeiras.**

O universo vale para o mês inteiro. Testes garantem que ele usa só volume passado.

## 6. Alvo e features

### 6.1 Alvo
Para a ação *i* no dia D:

`excesso(i, D) = retorno ajustado de i entre D e D+10 − retorno ajustado do BOVA11 entre D e D+10`

O rótulo é `1` se `excesso > 0`; senão, `0`. O BOVA11 ajustado é a referência porque, assim como
os preços ajustados das ações, ele inclui proventos.

### 6.2 Features de preço (todas calculadas só com dados até o fechamento de D)
- **Retornos:** 5, 20, 60 e 120 pregões.
- **Distância do preço até as médias** de 20 e 50 pregões.
- **Volatilidade diária:** 20 e 60 pregões.
- **Volume relativo:** volume de hoje dividido pela média de 20 pregões.
- **Retorno relativo ao BOVA11:** 20 e 60 pregões.
- **Versão em ranking** de cada feature: posição percentual da ação entre as do universo
  naquele dia (0 a 1). Isso deixa as features comparáveis entre anos com volatilidades
  diferentes.

Ações com menos de 120 pregões de histórico ficam fora naquele dia.

## 7. Modelo (combinador)

- **Base:** regressão logística (imputação pela mediana do treino + normalização), reaproveitando
  o padrão já testado.
- **Candidato:** `HistGradientBoostingClassifier` (scikit-learn, sem dependência nova).
- **Walk-forward:** retreino no **primeiro pregão de cada mês**, com janela que cresce desde 2016.
- **Embargo de 10 pregões:** o rótulo do dia D só fica conhecido em D+10. Por isso, um
  retreino feito no dia T usa só linhas com `D ≤ T − 10 pregões`. Sem isso, o treino veria o
  futuro. Haverá um teste específico para essa regra.
- **Escolha do modelo, em duas etapas e nesta ordem:**
  1. fica o modelo (base ou candidato) com **menor Brier médio** no laboratório;
  2. para esse modelo, ficam o *k* e o *p_min* (seção 8.2) com **maior excesso líquido médio por
     ano** da carteira de compras sobre o BOVA11.

  Essa configuração é a única levada à prova final.

## 8. Sinais e laboratório

### 8.1 Regra dos sinais
Para cada dia D:
- **Compra:** as *k* ações de maior probabilidade, se a probabilidade for ≥ *p_min*.
- **Venda:** as *k* ações de menor probabilidade, se a probabilidade for ≤ 1 − *p_min*.
- **Ponto de partida:** *k* = 5 e *p_min* = 0,55.

### 8.2 Ajuste no laboratório (limitado e registrado)
- A grade de ajuste é **fixa e pequena**: *k* ∈ {3, 5, 10} e *p_min* ∈ {0,50; 0,55; 0,60}, no
  máximo 9 combinações por modelo, e 2 modelos. No total são até 18 configurações.
- Toda configuração testada vai para `resultados/laboratorio/tentativas.csv`, com data, para
  que o número de tentativas fique visível ao julgar o resultado.

### 8.3 Simulação da carteira (só compras)
- **Capital dividido em 10 partes.** A cada pregão, uma parte compra, com pesos iguais, as
  ações sinalizadas naquele dia e segura por 10 pregões. Sem sinal, essa parte fica em caixa.
- **Custo:** 0,10% por lado (0,05% de custos + 0,05% de margem para slippage).
- **Execução:** no preço de fechamento do dia do sinal. Essa simplificação vai para as limitações.

### 8.4 Métricas do relatório
- **Brier** do modelo e da taxa histórica de "superar o Ibovespa" (referência).
- **Acerto das compras:** % que superou o BOVA11 em 10 pregões.
- **Acerto das vendas:** % que ficou atrás do BOVA11.
- **Carteira de compras:** retorno líquido acumulado e por ano, excesso sobre o BOVA11, pior
  queda e fração de anos com excesso positivo.
- **Agentes aleatórios:** 1.000 agentes que, a cada dia, compram o mesmo número de ações
  sorteadas do universo do dia. Medimos a % deles que a linha superou.
- **Cobertura:** ações e meses fora por falta de dados.

## 9. Prova final (reserva 2025+)

- `python main.py prova-final` roda a configuração escolhida no laboratório sobre 2025 em diante,
  grava `resultados/prova_final.json` e o relatório, e **se recusa a rodar de novo** se esse
  arquivo existir. Rodar de novo exige apagar o arquivo, o que fica visível no git.
- **Critérios definidos agora, antes de qualquer resultado.** A linha é **aprovada para uso com
  cautela** se, na reserva, cumprir os três:
  1. excesso líquido acumulado da carteira de compras sobre o BOVA11 **> 0**;
  2. superar **≥ 90%** dos agentes aleatórios;
  3. acerto das compras **> 50%**.
- **Limitação conhecida:** a reserva tem ~21 meses. Um resultado positivo é um sinal, não uma
  prova definitiva. A ferramenta continua sendo medida depois disso.

## 10. Migração do código atual

- `arquivo/experimento_jev_llm_ml/` recebe o experimento: decisões, resultados, `PREREGISTRO.md`
  e um README curto. O **Desvio 2** registra o encerramento em 09/10/2026, com 17 pregões e sem
  conclusões.
- **Migram para a estrutura nova, com os testes:**
  - `api_comum.py`, `jev.py`, `llm.py` e `estado.py`, para uso na etapa 3;
  - a lógica de walk-forward e imputação;
  - o teste dos agentes aleatórios;
  - `decisoes.py`, ou o equivalente para sinais.
- **Saem da pasta ativa** (ficam no arquivo): `baselines.py`, as comparações pré-registradas e
  a avaliação diária do experimento.

## 11. Erros e casos-limite

- **Falha de download (B3 ou Yahoo):** a etapa avisa e usa o último dia completo. Nenhum sinal
  é gerado com dados do dia faltando.
- **Ação nova no universo sem 120 pregões:** fica fora, e o relatório lista o caso.
- **Feriados:** só os pregões presentes nos dados contam (o calendário vem dos próprios arquivos da B3).
- **Ação que para de negociar no meio dos 10 pregões** (fechamento de capital, fusão): no
  treino, o rótulo fica vazio e a linha sai; na simulação, a posição é encerrada no último preço
  disponível.
- **Ação com o mesmo ticker reaproveitado por outra empresa:** detectada pela troca de nome do
  emissor no COTAHIST. Os dois períodos ficam como séries separadas.

## 12. Riscos abertos

- **Cobertura do Yahoo para ações que deixaram de existir:** se ficar abaixo de 95%, parte do
  viés de sobrevivência continua. Plano: medir na implementação e decidir se vale ajustar os
  preços da B3 por conta própria (desdobramentos e proventos).
- **Execução no fechamento:** o resultado real tende a ser um pouco pior.
- **Mudança de regime:** 2016–2024 inclui crises (2020) e eleições. A fração de anos bons
  mostra se o resultado depende de um período só.

## 13. Testes automáticos (pytest, sem internet)

- **Universo:** usa só volume dos 63 pregões anteriores; mudar volumes futuros não altera o mês.
- **Features:** mudar preços depois de D não altera nenhuma feature de D.
- **Alvo:** o rótulo de D é calculado com D+10 e nunca chega às features.
- **Embargo:** um retreino em T não contém linhas com D > T − 10 pregões.
- **Passeio aleatório:** em preços simulados sem padrão, o acerto fica entre 40% e 60%, e a
  carteira não supera de forma consistente os agentes aleatórios.
- **Trapaceiro:** um modelo que usa o rótulo supera 100% dos aleatórios. Isso prova que o
  laboratório detecta vazamento.
- **Custos:** a simulação cobra 0,10% por lado sobre o que mudou na carteira.
- **Prova final:** com `prova_final.json` presente, o comando recusa rodar.
- **Leitura do COTAHIST:** feita com um trecho fixo de exemplo (layout oficial de posições).

## 14. Critérios de pronto da etapa 1

- Os quatro comandos funcionam do zero, num ambiente novo, seguindo o README.
- Todos os testes passam.
- O relatório do laboratório traz todas as métricas da seção 8.4, mais a cobertura.
- O README explica a linha, a etapa 1 e o aviso de uso.
- A prova final **não** é rodada na etapa 1. Ela só roda quando o usuário decidir, depois de
  revisar o laboratório.
