# Grupo G01
## Integrantes:

* RAFAEL TAVAVARES SANTOS RM: 563487
* GABRIEL OLIVEIRA ALARAL RM: 563872
* FELIPE YAMAGUCHI MESQUITA RM: 556170

### Seed Base: 9
### Seed Final: 1009
### Identificador da Instância: af787c15102a

# Descrição do Fluxo da questão 1
## Nesta questão utilizamos 3 algoritmos integrados em um pipeline sequencial e dependente
*1. Etapa A (Brute Force): Executa uma busca exaustiva num subconjunto reduzido de 6 SKUs para determinar a solução ótima global e estabelecer o benchmark exato.
*2. Etapa B (Algoritmo Genético): Valida a sua calibração no mesmo subconjunto (alcançando 0.00% de gap em relação ao Brute Force) e, em seguida, otimiza a política de preços base do catálogo completo (N SKUs).
*2. Etapa C (Algoritmo Greedy): Recebe a política de preços consolidada do Algoritmo Genético e aloca estrategicamente o orçamento promocional de R$ 89.164,18 entre os 19 SKUs com maior demanda e estoque, acelerando o giro sem ferir o piso de margem.