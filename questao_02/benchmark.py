"""Escalabilidade da Questão 2 em três tamanhos de entrada (catálogo x1, x4, x16).

    python questao_02/benchmark.py
Grava questao_02/resultados/q2_benchmark.csv e .md
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from questao_02.main import executar_pipeline
from questao_02.modelo import Instancia


def main():
    base = Instancia.de_arquivos("G01")
    linhas = []
    for fator in (1, 4, 16):
        inst = base.replicar(fator)
        r = executar_pipeline(inst, seed=1009, n_val=5000, n_est=2000, lam=0.5, beta=0.10, uplift=0.20, custo_pct=0.05)
        for etapa, v in r["medidas"].items():
            linhas.append({"N_skus": inst.n, "etapa": etapa, "tempo_s": v["tempo_s"], "memoria_mb": v["memoria_pico_mb"]})
        linhas.append({"N_skus": inst.n, "etapa": "TOTAL", "tempo_s": sum(v["tempo_s"] for v in r["medidas"].values()),
                       "memoria_mb": max(v["memoria_pico_mb"] for v in r["medidas"].values())})
        print(f"N={inst.n}: P(todas)={r['val']['prob_todas']:.3f}")
    out = pd.DataFrame(linhas)
    pasta = RAIZ / "questao_02" / "resultados"
    out.to_csv(pasta / "q2_benchmark.csv", index=False)
    (pasta / "q2_benchmark.md").write_text(out.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
