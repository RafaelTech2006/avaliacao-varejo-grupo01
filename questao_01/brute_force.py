"""Etapa A — Brute Force: ótimo exato num subconjunto reduzido (benchmark do GA)."""
from __future__ import annotations

import itertools
import time
import tracemalloc

import numpy as np

from questao_01.modelo import ModeloQ1


def selecionar_subconjunto(modelo: ModeloQ1, tamanho: int = 6, seed: int = 0) -> np.ndarray:
    """Sorteia, com seed, ``tamanho`` SKUs ativos (reprodutível)."""
    ativos = modelo.ativos
    if tamanho < 1 or tamanho > len(ativos):
        raise ValueError(f"subconjunto de {tamanho} SKUs inválido: há {len(ativos)} SKUs ativos")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ativos, size=tamanho, replace=False))


def brute_force(modelo: ModeloQ1, posicoes: np.ndarray) -> dict:
    """Enumera TODAS as combinações de preços permitidos (que respeitam a margem).

    Tempo O(P^S) com P = níveis viáveis por SKU e S = tamanho do subconjunto; memória O(S).
    """
    posicoes = np.asarray(posicoes)
    opcoes = [np.flatnonzero(modelo.viavel[i]) for i in posicoes]
    tabelas = [modelo.caixa[i, o] for i, o in zip(posicoes, opcoes)]
    tracemalloc.start()
    t0 = time.perf_counter()
    melhor, melhor_comb, n_comb = -np.inf, None, 0
    for comb in itertools.product(*[range(len(o)) for o in opcoes]):
        n_comb += 1
        total = 0.0
        for j, c in enumerate(comb):
            total += tabelas[j][c]
        if total > melhor:
            melhor, melhor_comb = total, comb
    tempo = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    idx = np.array([opcoes[j][c] for j, c in enumerate(melhor_comb)]) if melhor_comb else None
    return {"idx": idx, "caixa": melhor, "combinacoes": n_comb, "tempo_s": tempo,
            "memoria_pico_mb": pico / 2**20}
