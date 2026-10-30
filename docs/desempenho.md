# Desempenho e resultados medidos

Medições com `time.perf_counter()` e `tracemalloc` (pico de memória alocada em Python). Máquina de desenvolvimento, Python 3.12; valores variam um pouco de execução para execução, os resultados numéricos (caixa, probabilidades) são reprodutíveis pela seed. Reproduza com `python questao_01/benchmark.py` e `python questao_02/benchmark.py`.

## Questão 1 (seed 1009)

### Brute Force × GA no subconjunto (Q1-002, Q1-007, Q1-008, Q1-014, Q1-016, Q1-030)
| Método | Caixa | Gap | Avaliações | Tempo (s) | Memória (MB) |
|---|---|---|---|---|---|
| Brute Force | R$ 92.153,65 | — | 343 | 0,004 | 0,001 |
| Algoritmo Genético | R$ 92.153,65 | **0,00 %** | 2.100 | 0,013 | 0,030 |

(Só 343 combinações, e não 7^6 = 117.649, porque o piso de margem elimina níveis de preço: o produto dos níveis viáveis dos 6 SKUs é 343.)

### GA no catálogo completo — duas configurações
| Config | População | Gerações (máx) | Mutação | Melhor na geração | Executadas | Caixa | Tempo (s) | Memória (MB) |
|---|---|---|---|---|---|---|---|---|
| A | 50 | 100 | 0,05 | 92 | 100 | R$ 466.012,29 | 0,033 | 0,091 |
| B | 100 | 200 | 0,025 | 71 | 131 (parou por estagnação) | R$ 466.047,37 | 0,046 | 0,162 |

Discussão: B converge ao **ótimo exato** (R$ 466.047,37, obtido por argmax por SKU, pois sem incentivos o problema é separável); A fica 0,0075 % abaixo e ainda melhorava perto da geração 100 (critério de parada por geração máxima), enquanto B parou por estagnação. Custo: ≈2× memória e ≈1,4× tempo — irrelevante em valor absoluto aqui. Gráfico: `questao_01/resultados/q1_fitness_por_geracao.png`.
Escolhas: torneio (k=3) dá pressão seletiva moderada; crossover de ponto único (0,8) preserva blocos de SKUs vizinhos; mutação por gene mantém diversidade; elitismo (2) garante que o melhor nunca se perde (testado); parada por gerações máximas **ou** 30/60 gerações sem melhora.

### Greedy antes/depois (`questao_01/resultados/q1_antes_depois_greedy.csv`, 15 SKUs)
| | Caixa |
|---|---|
| Política do GA (antes) | R$ 466.047,37 |
| Após incentivos, **líquido** do gasto (depois) | R$ 481.154,85 (+R$ 15.107,48) |
| Orçamento usado | R$ 20.134,06 de R$ 89.164,18 (22,6 %) |

O orçamento não é esgotado porque, além de certo ponto, cada real extra de incentivo custa mais do que gera (o Greedy só concede degraus com ganho líquido > 0).

### Resultado final Q1
Caixa R$ 481.154,85 · meta R$ 331.680,98 → **atingida, folga R$ 149.473,87** · margem global 21,48 % · estoque remanescente 1.035 un (25,0 %) · cobertura da dívida 1,56× · orçamento usado 22,6 %.

### Otimização demonstrada (antes/depois)
Fitness do GA: versão pandas por linha (`iloc`) → tabela NumPy pré-computada. 300 avaliações: **0,650 s → 0,0018 s (≈350×)**, resultados idênticos (verificado por `assert` e teste). Efeito no GA completo: o original levava ~26 s no catálogo (pop 50 × 100 gerações); agora 0,03 s.

### Escalabilidade Q1 (`q1_benchmark.md`)
| Algoritmo | Entrada | Trabalho | Tempo (s) | Memória (MB) |
|---|---|---|---|---|
| Brute Force | S=4 | 2.401 comb. | 0,024 | 0,001 |
| Brute Force | S=6 | 117.649 | 1,55 | 0,001 |
| Brute Force | S=8 | 5.764.801 | 99,3 | 0,001 |
| GA | N=30 | 130 mil avaliações-gene | 0,034 | 0,09 |
| GA | N=3.000 | 13 mi | 0,28 | 5,2 |
| GA | N=30.000 | 130 mi | 2,89 | 51,6 |
| Greedy | N=30 / 3.000 / 30.000 | — | 0,002 / 0,070 / 0,90 | 0,01 / 0,63 / 5,6 |

## Questão 2 (seed 1009; validação com seed 11009)

### Desempenho por etapa (24 SKUs, 2.000 + 5.000 cenários)
| Etapa | Tempo (s) | Pico de memória (MB) |
|---|---|---|
| Monte Carlo (estimação) | 0,15 | 33,6 |
| Programação Dinâmica | 0,28 | 30,7 |
| Greedy | 0,02 | 0,02 |
| Monte Carlo (validação) | 0,14 | 20,0 |

### Escalabilidade (3 tamanhos; catálogo replicado ×1, ×4, ×16 — `q2_benchmark.md`)
| N SKUs | MC estimação (s / MB) | DP (s / MB) | Greedy (s) | MC validação (s / MB) | Total (s) |
|---|---|---|---|---|---|
| 24 | 0,17 / 33,6 | 0,33 / 30,7 | 0,017 | 0,15 / 20,0 | 0,66 |
| 96 | 0,64 / 134,3 | 1,18 / 30,7 | 0,064 | 0,67 / 77,7 | 2,56 |
| 384 | 3,31 / 536,9 | 4,70 / 30,9 | 0,273 | 2,50 / 308,5 | 10,79 |
Tempo cresce ~linearmente com N (×4 SKUs ≈ ×3,9 no tempo); memória do Monte Carlo também linear em N (blocos de cenários × W × N); a da DP é constante (processa um SKU por vez).

### Probabilidade de pagamento (validação independente, 5.000 cenários)
| Política | P(S4) | P(S8) | P(S12) | P(todas) | Caixa final médio | P5 | P10 |
|---|---|---|---|---|---|---|---|
| Fixa (1,0) | 100 % | 100 % | 99,94 % | 99,94 % | 77.769,83 | 47.014,04 | 54.327,61 |
| DP | 100 % | 100 % | 99,98 % | 99,98 % | 236.724,34 | 143.905,57 | 170.680,86 |
| DP + Greedy | 100 % | 100 % | 99,98 % | 99,98 % | 238.091,76 | 142.314,22 | 169.851,62 |

Plano final: percentil de risco escolhido **P10** (alinhado ao `limite_risco_falha` = 10 %) = R$ 169.851,62; margem média 19,9 %; estoque final médio 460 un (17,2 %); **1 cenário de falha em 5.000** (semana 12); 68 destaques concedidos (R$ 13.326,13 de R$ 83.147,04 de orçamento). Critério de risco da instância (falha ≤ 10 % por vencimento): **atendido**.

Interpretação: no cenário base a instância tem folga (a dívida soma ≈ R$ 227 mil contra caixa de R$ 238 mil *após* pagá-la), então até a política fixa quase sempre paga; o ganho da DP é de **caixa** (≈3×), principalmente por praticar preço mais alto (nível 1,12 em 76 % das decisões SKU-semana) enquanto o estoque dura, e reduzi-lo no fim. O Greedy agrega pouco (+R$ 1,4 mil): destaque custa 5 % da receita nominal e só compensa em SKUs de alta margem e estoque folgado.

### Cenários de estresse (sem alterar código; 5.000 cenários)
| Cenário | Política fixa P(todas) | DP + Greedy P(todas) | Caixa final médio (DP+Greedy) |
|---|---|---|---|
| Base | 99,94 % | 99,98 % | R$ 238.092 |
| dívidas +30 %, concorrente −8 %, logística +15 % | **44,3 %** | **99,2 %** | R$ 157.065 |
| dívidas +10 %, concorrente −8 %, logística +15 %, orçamento −25 %, Q2-001 indisponível | 97,0 % | 99,9 % | R$ 192.491 |

## Q1 sob cenário-surpresa
`--meta-mult 1.10 --orcamento-mult 0.75 --custo-log-mult 1.15 --mutacao 0.2 --remover-sku Q1-001`: meta R$ 364.849,08; caixa R$ 445.552,96; meta atingida (folga R$ 80.703,88); cobertura da dívida 1,45×; orçamento usado 29,1 %.
