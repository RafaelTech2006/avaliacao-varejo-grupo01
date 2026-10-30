# Relatório de Decisões Arquiteturais e de Modelagem

Este documento detalha as principais decisões técnicas tomadas pelo Grupo G01 na conceção e implementação de ambas as questões do Checkpoint 6.

---

## 1. Justificativa da Cadeia de Algoritmos (Questão 1)
Conforme estipulado nas diretrizes do projeto, a solução integra três abordagens algorítmicas distintas de forma encadeada:
1. **Etapa A - Brute Force (Benchmark Exato):** Aplicado num subconjunto reduzido de 6 SKUs para validação matemática exata, comprovando a eficácia do Algoritmo Genético (gap de 0,00%)[cite: 4].
2. **Etapa B - Algoritmo Genético (Otimização do Catálogo Completo):** Utilizado para gerir o espaço de busca complexo do catálogo completo, superando a meta de caixa operacional[cite: 4].
3. **Etapa C - Algoritmo Greedy (Alocação de Incentivos):** Distribuiu o orçamento promocional limitado priorizando o escoamento de inventário e giro sem violar o piso de margem mínima[cite: 4].

---

## 2. Justificativa da Cadeia de Algoritmos (Questão 2)
A Questão 2 modela a reprecificação sob incerteza num horizonte temporal restrito de 12 semanas para cumprimento de obrigações financeiras nas semanas 4, 8 e 12[cite: 5]:
1. **Etapa A - Monte Carlo Estimativo:** Simula $2.000$ cenários incorporando incertezas estocásticas reais (demanda log-normal, oscilação de preços de concorrentes em $\pm 10\%$ e volatilidade de custos logísticos entre $+5\%$ e $+25\%$) para alimentar a etapa seguinte com médias robustas[cite: 5].
2. **Etapa B - Programação Dinâmica (Top-Down com Memoization):** Utiliza as distribuições geradas para determinar a trajetória semanal ótima de preços e inventário por SKU, resolvendo a subestrutura ótima entre períodos[cite: 5].
3. **Etapa C - Greedy Semanal:** Refina a política a cada semana alocando o orçamento promocional escasso de acordo com o maior benefício marginal (ROI) por unidade monetária investida[cite: 5].
4. **Etapa D - Validação Monte Carlo Final:** Submete o plano completo a $5.000$ cenários estocásticos independentes para certificar a robustez e reportar as probabilidades reais de sucesso de cada compromisso[cite: 5].

---

## 3. Reprodutibilidade e Integridade dos Dados
* **Utilização de Seeds:** Fixação estrita da `seed_base` e `seed_final` derivadas das matrículas do Grupo G01 em todos os scripts para assegurar a reprodutibilidade integral dos resultados estocásticos[cite: 2, 3, 6].
* **Imutabilidade dos Inputs:** Os ficheiros oficiais presentes na pasta `data/` foram gerados pelo notebook oficial fornecido, protegidos contra edições manuais e validados via hashes SHA-256 registados no `manifesto_dados.json`[cite: 2, 3, 6].