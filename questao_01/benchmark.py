"""Medição de tempo/memória em entradas pequena, média e grande (Big O empírico) — Questão 1.

    python questao_01/benchmark.py
Grava questao_01/resultados/q1_benchmark.csv e .md
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from comum.dados import carregar_instancia_bruta
from questao_01.brute_force import brute_force
from questao_01.genetic_algorithm import AlgoritmoGenetico, ConfigGA
from questao_01.greedy import GreedyIncentivos
from questao_01.modelo import ModeloQ1


def catalogo_replicado(df, n):
    reps = int(np.ceil(n / len(df)))
    big = pd.concat([df.assign(sku=df["sku"] + f"#{r}") for r in range(reps)], ignore_index=True)
    return big.iloc[:n].reset_index(drop=True)


def main():
    df, cfg = carregar_instancia_bruta(1, "G01")
    base = ModeloQ1.construir(df, cfg)
    linhas = []
    # Brute Force: tempo O(P^S)
    por_viab = np.argsort(-base.viavel.sum(1))
    for S in (4, 6, 8):
        pos = np.sort(por_viab[:S])
        r = brute_force(base, pos)
        linhas.append({"algoritmo": "Brute Force", "tamanho": f"S={S}", "variavel": S,
                       "trabalho": r["combinacoes"], "tempo_s": r["tempo_s"], "memoria_mb": r["memoria_pico_mb"]})
    # GA e Greedy: variam com N (K=50, G=100 fixos; sem parada antecipada p/ medir G*K*N)
    for N in (30, 3_000, 30_000):
        m = ModeloQ1.construir(catalogo_replicado(df, N), cfg)
        ativos = m.ativos
        ga = AlgoritmoGenetico(m, ConfigGA(paciencia=10**9), posicoes=ativos, seed=1).otimizar()
        linhas.append({"algoritmo": "Algoritmo Genético", "tamanho": f"N={N}", "variavel": N,
                       "trabalho": ga.avaliacoes * len(ativos), "tempo_s": ga.tempo_s, "memoria_mb": ga.memoria_pico_mb})
        idx = np.zeros(m.n, dtype=int)
        idx[ativos] = ga.idx
        gr = GreedyIncentivos(m, cfg["orcamento_promocional"] * N / 30).alocar(idx)
        linhas.append({"algoritmo": "Greedy", "tamanho": f"N={N}", "variavel": N, "trabalho": N,
                       "tempo_s": gr["tempo_s"], "memoria_mb": gr["memoria_pico_mb"]})
    out = pd.DataFrame(linhas)
    pasta = RAIZ / "questao_01" / "resultados"
    out.to_csv(pasta / "q1_benchmark.csv", index=False)
    (pasta / "q1_benchmark.md").write_text(out.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
