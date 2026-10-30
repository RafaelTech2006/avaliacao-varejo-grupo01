"""Etapa B — Algoritmo Genético (busca no catálogo completo, validado pelo Brute Force)."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from questao_01.modelo import ModeloQ1


@dataclass
class ConfigGA:
    nome: str = "A"
    tamanho_populacao: int = 50
    geracoes: int = 100
    taxa_crossover: float = 0.8
    taxa_mutacao: float = 0.05      # probabilidade por gene
    tamanho_torneio: int = 3
    n_elite: int = 2                # elitismo: melhores indivíduos copiados intactos
    paciencia: int = 30             # parada antecipada: gerações sem melhora
    penalidade: float = 1e7         # penalização por gene que viola o piso de margem


@dataclass
class ResultadoGA:
    idx: np.ndarray
    fitness: float
    historico: list = field(default_factory=list)
    geracao_melhor: int = 0
    geracoes_executadas: int = 0
    tempo_s: float = 0.0
    memoria_pico_mb: float = 0.0
    avaliacoes: int = 0


class AlgoritmoGenetico:
    """Cromossomo = vetor de índices de nível de preço (um gene por SKU)."""

    def __init__(self, modelo: ModeloQ1, config: ConfigGA, posicoes: np.ndarray | None = None,
                 seed: int = 0):
        self.cfg = config
        self.seed = seed
        self.posicoes = np.arange(modelo.n) if posicoes is None else np.asarray(posicoes)
        self.n_genes = len(self.posicoes)
        self.n_niveis = modelo.n_niveis
        # tabela de fitness (n_genes x L): caixa, com penalização forte onde a margem é violada
        self.tabela = np.where(modelo.viavel[self.posicoes], modelo.caixa[self.posicoes],
                               -config.penalidade)
        self._ar = np.arange(self.n_genes)

    def fitness(self, pop: np.ndarray) -> np.ndarray:
        """Fitness vetorizado da população inteira: O(K*N) por geração."""
        return self.tabela[self._ar[None, :], pop].sum(axis=1)

    def otimizar(self) -> ResultadoGA:
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        K, n, L = c.tamanho_populacao, self.n_genes, self.n_niveis
        n_elite = min(max(c.n_elite, 0), K)
        tracemalloc.start()
        t0 = time.perf_counter()

        pop = rng.integers(0, L, size=(K, n))
        melhor_fit, melhor_ind, ger_melhor, hist, aval = -np.inf, None, 0, [], 0
        g = 0
        for g in range(1, c.geracoes + 1):
            fit = self.fitness(pop)
            aval += K
            i_best = int(fit.argmax())
            if fit[i_best] > melhor_fit + 1e-9:
                melhor_fit, melhor_ind, ger_melhor = float(fit[i_best]), pop[i_best].copy(), g
            hist.append({"geracao": g, "melhor_geracao": float(fit[i_best]),
                         "media": float(fit.mean()), "melhor_global": melhor_fit})
            if g - ger_melhor >= c.paciencia:          # critério de parada: estagnação
                break

            elite = pop[np.argsort(-fit)[:n_elite]]
            n_filhos = K - n_elite
            n_pares = (n_filhos + 1) // 2
            # seleção por torneio (vetorizada)
            cand = rng.integers(0, K, size=(2 * n_pares, min(c.tamanho_torneio, K)))
            venc = cand[np.arange(2 * n_pares), fit[cand].argmax(axis=1)]
            p1, p2 = pop[venc[:n_pares]], pop[venc[n_pares:]]
            # crossover de ponto único
            if n >= 2:
                faz = (rng.random(n_pares) < c.taxa_crossover)[:, None]
                ponto = rng.integers(1, n, size=n_pares)[:, None]
                pref = self._ar[None, :] < ponto
                f1 = np.where(faz, np.where(pref, p1, p2), p1)
                f2 = np.where(faz, np.where(pref, p2, p1), p2)
            else:
                f1, f2 = p1, p2
            filhos = np.vstack([f1, f2])[:n_filhos]
            # mutação uniforme por gene
            mut = rng.random(filhos.shape) < c.taxa_mutacao
            filhos[mut] = rng.integers(0, L, size=int(mut.sum()))
            pop = np.vstack([elite, filhos]) if n_elite else filhos

        tempo = time.perf_counter() - t0
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ResultadoGA(idx=melhor_ind, fitness=melhor_fit, historico=hist,
                           geracao_melhor=ger_melhor, geracoes_executadas=g, tempo_s=tempo,
                           memoria_pico_mb=pico / 2**20, avaliacoes=aval)


def gap_percentual(otimo: float, obtido: float) -> float | None:
    """Gap (%) do GA em relação ao ótimo do Brute Force."""
    if not np.isfinite(otimo) or otimo == 0:
        return None
    return float((otimo - obtido) / abs(otimo) * 100.0)


def otimo_separavel(modelo: ModeloQ1) -> np.ndarray:
    """Sem incentivos, o problema é separável por SKU: o argmax por linha é o ótimo exato.

    Serve como limite de referência para o GA no catálogo completo (onde o Brute Force é inviável).
    """
    return np.where(modelo.viavel, modelo.caixa, -np.inf).argmax(axis=1)
