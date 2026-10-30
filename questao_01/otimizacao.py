"""Otimização demonstrada: fitness do GA — versão ANTES (loop pandas) x DEPOIS (tabela NumPy).

``fitness_antes`` reproduz a abordagem inicial do grupo (``iloc`` + cálculo por linha a cada
avaliação). ``fitness_depois`` usa as tabelas (N x L) pré-computadas em ModeloQ1.
"""
from __future__ import annotations

import time

import numpy as np

from questao_01.modelo import ModeloQ1


def fitness_antes(modelo: ModeloQ1, idx: np.ndarray) -> float:
    df, total = modelo.df, 0.0
    for i in range(modelo.n):
        row = df.iloc[i]
        preco = row["preco_atual"] * modelo.niveis[idx[i]]
        var = (preco - row["preco_atual"]) / row["preco_atual"]
        dem = max(0.0, row["demanda_base_90d"] * (1 - row["elasticidade"] * var))
        vend = min(dem, row["estoque"])
        cun = preco * (1 - row["taxa_marketplace"]) - row["custo_unitario"] - row["custo_logistico"]
        if cun / preco >= modelo.piso:
            total += vend * cun
    return total


def fitness_depois(modelo: ModeloQ1, idx: np.ndarray) -> float:
    ar = np.arange(modelo.n)
    return float(np.where(modelo.viavel[ar, idx], modelo.caixa[ar, idx], 0.0).sum())


def comparar(modelo: ModeloQ1, n_avaliacoes: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, modelo.n_niveis, size=(n_avaliacoes, modelo.n))
    t0 = time.perf_counter()
    a = [fitness_antes(modelo, ind) for ind in pop]
    t_antes = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = [fitness_depois(modelo, ind) for ind in pop]
    t_depois = time.perf_counter() - t0
    assert np.allclose(a, d), "as duas versões devem produzir os mesmos valores"
    return {"avaliacoes": n_avaliacoes, "tempo_antes_s": t_antes, "tempo_depois_s": t_depois,
            "speedup": t_antes / t_depois if t_depois > 0 else float("inf"), "resultados_iguais": True}
