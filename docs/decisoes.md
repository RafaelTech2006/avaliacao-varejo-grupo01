# Decisões de modelagem, arquitetura e limitações

## 1. Cadeias de algoritmos
**Q1 — Brute Force → GA → Greedy.** O Brute Force produz o ótimo exato de 6 SKUs (sorteados com a seed); o GA roda sobre o mesmo subconjunto e o gap é calculado contra esse ótimo; depois o GA otimiza o catálogo completo e a política resultante (vetor de níveis de preço) é a entrada do Greedy, que distribui o orçamento de incentivos.
**Q2 — Monte Carlo → DP → Greedy → Monte Carlo.** O MC (2.000 cenários) gera a distribuição de vendas por (semana, SKU, preço) → a DP usa essas distribuições para escolher a trajetória de preços → o Greedy usa o plano e as mesmas estatísticas para escolher destaques → o plano completo é simulado em 5.000 cenários com outra seed.

## 2. Modelo (vem da instância)
- Demanda linear na elasticidade: `D = base·max(0, 1 − e·(preço/ref − 1))` (Q1 sobre o preço de referência; Q2 sobre o preço relativo ao concorrente).
- Caixa unitário = `preço·(1 − taxa_marketplace) − custo_unitario − custo_logistico` (mesma definição usada pelo notebook gerador para calcular metas e dívidas). Margem unitária = caixa unitário / preço; piso vem de `margem_minima_percentual`.
- Níveis de preço, meta, dívidas, orçamentos, sigmas, caixa inicial, nº de semanas e limite de risco são **lidos dos JSON**. Nada disso é constante no código.
- Q1: SKUs sem nenhum preço viável (Q1-017, 024, 027, 029) ficam fora da política (não vendem).
- Q2: dívidas **cumulativas** — o compromisso da semana d é pago no prazo se `caixa_inicial + caixa gerado até d − dívidas vencidas até d ≥ 0`. Vendas são inteiras (`rint` da demanda), limitadas pelo estoque, controlado por cenário.

## 3. Premissas próprias (não vêm da instância — ajustáveis por parâmetro)
| Premissa | Valor padrão | Onde alterar |
|---|---|---|
| Fração comum dos choques entre SKUs (RHO) | 0,5 | `questao_02/modelo.py: RHO` |
| Passeio aleatório do preço do concorrente (σ semanal da instância, acumulado) | — | `sigma_preco_concorrente` |
| Destaque Q2: aumento de demanda / custo (% da receita nominal semanal) | +20 % / 5 % | `--uplift`, `--custo-destaque-pct` |
| Aversão a risco da DP (penalidade por desvio-padrão) / urgência por vencimento | λ=0,5 / β=0,10 | `--lambda-risco`, `--beta-urgencia` |
| Valor terminal do estoque não vendido na DP | 0 | `dynamic_programming.py` |
| Incentivo Q1: degraus 2/4/6 % do preço; fração dos compradores que já comprariam e resgatam o benefício | 25 % | `--niveis-incentivo`, `--taxa-resgate-base` |
| Mutação do GA, população etc. | ver `--help` | CLI |

Sem a taxa de resgate (incentivo grátis só sobre vendas incrementais) o Greedy gastaria o orçamento inteiro e o ganho seria sobrestimado; com ela só SKUs de margem alta e elasticidade alta justificam o incentivo.

## 4. Limitações assumidas (para a defesa)
1. **Q1 é separável por SKU** enquanto não há incentivos: o ótimo exato do catálogo é o argmax por linha. O GA atinge esse ótimo; o valor didático está em validar o GA contra o Brute Force e em comparar configurações. Usamos o argmax como referência extra.
2. **Q2 – política de preço em malha aberta:** a DP resolve por SKU (estoque é o estado) e o plano é uma trajetória *esperada*; na validação os preços do plano são fixos, mesmo que o estoque simulado fuja do esperado. Uma política adaptativa (preço em função do estoque observado) tenderia a ser melhor; optamos pela trajetória fixa porque o enunciado pede o caminho de preços reconstruído e exibido, e porque é o que o Greedy consome.
3. **A DP é por SKU, sem restrição de caixa agregada:** o risco de pagar as dívidas é tratado indiretamente (λ e β) e medido pelo Monte Carlo, não otimizado diretamente.
4. Entre semanas, os choques de demanda e de custo logístico são independentes; só o preço do concorrente tem memória (passeio aleatório), mas a DP não o inclui como estado: usa a distribuição marginal de cada semana estimada pelo Monte Carlo.
5. No cenário base a instância tem folga de caixa, então as probabilidades ficam ≈100 % para qualquer política; as diferenças aparecem nos cenários de estresse (`desempenho.md`).

## 5. Reprodutibilidade e integridade
- Todo componente aleatório usa `numpy.random.default_rng(seed)`: sorteio do subconjunto, GA, Monte Carlo (estimação: `seed`; validação: `seed + 10.000`, garantindo cenários independentes).
- `data/` preservado; `comum.dados.verificar_manifesto()` compara SHA-256 a cada execução; `tests/test_dados.py` falha se algum arquivo mudar.
- Cenário-surpresa por parâmetro (`--meta-mult`, `--divida-mult`, `--concorrente-delta`, `--custo-log-mult`, `--orcamento-mult`, `--remover-sku`, `--mutacao`, `--populacao`...), sem editar código.
