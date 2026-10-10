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
