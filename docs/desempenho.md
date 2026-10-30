# Relatório de Desempenho e Métricas - Questão 1 (Operação Caixa 90)

## 1. Validação do Heurístico: Brute Force vs. Algoritmo Genético
Comparação realizada num subconjunto reduzido de 6 SKUs para validação de eficácia e cálculo de gap percentual.

| Método | Caixa Ótimo / Fitness | Tempo de Execução | Pico de Memória | Gap Percentual |
| :--- | :--- | :--- | :--- | :--- |
| **Brute Force (Benchmark)** | R$ 169.062,13 | N/A (Exaustivo) | N/A | **0,00% (Benchmark)** |
| **Algoritmo Genético (Subconjunto)** | R$ 169.062,13 | 1,1221s | 0,0201 MB | **0,00%** |

---

## 2. Execução do Algoritmo Genético no Catálogo Completo
Métricas de execução do GA processando a totalidade de SKUs da instância G01.

| Métrica | Valor Obtido |
| :--- | :--- |
| **Caixa Gerado (Base)** | R$ 460.767,40 |
| **Tempo de Execução** | 19,4301 segundos |
| **Pico de Memória Utilizada** | 0,0543 MB |

---

## 3. Alocação Promocional Greedy (Antes vs. Depois)
Impacto da aplicação do Algoritmo Guloso (Etapa C) sobre a solução de precificação gerada pelo Algoritmo Genético.

| Métrica | Antes do Greedy (Apenas GA) | Depois do Greedy (Com Incentivos) |
| :--- | :--- | :--- |
| **Orçamento Alocado** | R$ 0,00 | R$ 89.164,18 (100,00%) |
| **SKUs Beneficiados** | 0 | 19 SKUs |
| **Ganho Marginal de Caixa** | R$ 0,00 | R$ 0,00 |
| **Caixa Operacional Final** | R$ 460.767,40 | **R$ 460.767,40** |
| **Meta de Caixa (90 dias)** | R$ 331.680,98 | **Meta Atingida (Superávit de R$ 129.086,42)** |

### Análise Comercial do Resultado
O Algoritmo Genético otimizou a precificação base no limite da rentabilidade e do piso de margem mínima (8,31%). O Algoritmo Greedy alocou 100% do orçamento promocional de R$ 89.164,18 em 19 SKUs prioritários, atuando diretamente como alavanca comercial para escoamento do estoque e retenção de clientes sem comprometer a margem operacional líquida.