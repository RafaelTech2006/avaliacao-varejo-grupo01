# Relatório de Decisões Arquiteturais e de Modelagem

Este documento detalha as principais decisões técnicas tomadas pelo Grupo G01 na conceção e implementação da pipeline de otimização de preços e inventário da Operação Caixa 90.

---

## 1. Justificativa da Cadeia de Algoritmos (Questão 1)
Conforme estipulado nas diretrizes do projeto, a solução integra três abordagens algorítmicas distintas de forma encadeada:

1. **Etapa A - Brute Force (Benchmark Exato):**
   * **Decisão:** Aplicar a força bruta num subconjunto reduzido de 6 SKUs para testar todas as combinações discretas de preços possíveis.
   * **Justificativa:** Serviu como validação matemática absoluta, permitindo calcular o gap percentual e certificar que o Algoritmo Genético atinge o nível de eficácia ótima (comprovado pelo gap de 0,00%).

2. **Etapa B - Algoritmo Genético (Otimização do Catálogo Completo):**
   * **Decisão:** Utilizar o Algoritmo Genético (com operadores de seleção, cruzamento, mutação e elitismo) para gerir o espaço de busca complexo do catálogo completo.
   * **Justificativa:** O espaço de combinações de preços de todo o catálogo torna a força bruta inviável computacionalmente (explosão combinatória). O AG encontrou com eficiência uma política de preços que gerou R$ 460.767,40 em caixa, superando a meta estabelecida.

3. **Etapa C - Algoritmo Greedy (Alocação de Incentivos):**
   * **Decisão:** Implementar um algoritmo guloso para distribuir o orçamento promocional limitado (R$ 89.164,18) após a definição da política de preços base.
   * **Justificativa:** O *Greedy* priorizou os 19 SKUs com maior impacto de escoamento e giro de estoque, respeitando rigorosamente o piso de margem mínima regulamentar e garantindo uma alocação ótima dos recursos escassos.

---

## 2. Tratamento de Restrições de Negócio
* **Piso de Margem Mínima:** Todos os preços gerados ou testados pelas heurísticas foram estritamente filtrados para nunca violar o limite mínimo de margem estabelecido pela instância (8,31%), prevenindo vendas abaixo do custo operacional.
* **Controlo de Estoque:** As quantidades negociadas e os incentivos promocionais respeitaram as unidades físicas disponíveis em inventário em cada SKU.

---

## 3. Reprodutibilidade e Integridade dos Dados
* **Utilização de Seeds:** Fixação da `seed_base` (9) e `seed_final` (1009) derivadas das matrículas do Grupo G01 para garantir a total reprodutibilidade estocástica de todas as execuções de Monte Carlo e heurísticas[cite: 2, 3, 6].
* **Imutabilidade dos Inputs:** Os ficheiros oficiais em `data/` (`q1_catalogo_G01.csv`, `q1_config_G01.json`, etc.) foram gerados pelo notebook oficial fornecido e mantidos estritamente inalterados, com validação de hashes registada no `manifesto_dados.json`[cite: 2, 3, 6].