"""Gera docs/arquitetura.png (fluxos das Questões 1 e 2).   python docs/gerar_arquitetura.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axs = plt.subplots(2, 1, figsize=(13, 8.2))


def caixa(ax, x, y, w, h, titulo, sub, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=cor, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h * 0.66, titulo, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.8)


def seta(ax, x0, x1, y, txt="", y_txt=0.12, rad=0.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.6, connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((x0 + x1) / 2, y + y_txt, txt, ha="center", fontsize=7.5, style="italic")


ax = axs[0]
ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("Questão 1 — Operação Caixa 90:  Brute Force  →  Algoritmo Genético  →  Greedy", fontsize=12, weight="bold")
caixa(ax, 0.2, 0.9, 2.6, 1.4, "Dados G01", "q1_catalogo + q1_config\n(SHA-256 verificado)", "#eeeeee")
caixa(ax, 3.4, 0.9, 2.8, 1.4, "A  Brute Force", "6 SKUs, todas as combinações\n→ ótimo exato (benchmark)", "#cfe8ff")
caixa(ax, 6.8, 0.9, 2.8, 1.4, "B  Algoritmo Genético", "valida no subconjunto (gap)\ndepois catálogo completo", "#d6f5d6")
caixa(ax, 10.2, 0.9, 2.6, 1.4, "C  Greedy", "orçamento de incentivos\nmaior ganho líquido / R$", "#ffe3b8")
seta(ax, 2.8, 3.4, 1.6); seta(ax, 6.2, 6.8, 1.6, "ótimo BF\n→ gap do GA", 0.45)
seta(ax, 9.6, 10.2, 1.6, "política de\npreços do GA", 0.45)
ax.text(6.5, 0.35, "Saída: caixa, margem, estoque remanescente, cobertura da dívida, uso do orçamento  |  fitness por geração", ha="center", fontsize=9)

ax = axs[1]
ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
ax.set_title("Questão 2 — Plano de 12 Semanas:  Monte Carlo  →  Programação Dinâmica  →  Greedy  →  Monte Carlo (validação)",
             fontsize=12, weight="bold")
caixa(ax, 0.1, 1.2, 2.9, 1.5, "A  Monte Carlo", "2.000 cenários: demanda,\nconcorrente, logística\n→ pmf de vendas por (sem, SKU, preço)", "#cfe8ff")
caixa(ax, 3.4, 1.2, 2.9, 1.5, "B  Prog. Dinâmica", "estado (semana, estoque)\ndecisão: nível de preço\n→ trajetória semanal ótima", "#d6f5d6")
caixa(ax, 6.7, 1.2, 2.8, 1.5, "C  Greedy semanal", "destaques/frete promocional\nmaior ganho líquido / R$\ncom orçamento semanal", "#ffe3b8")
caixa(ax, 9.9, 1.2, 3.0, 1.5, "D  Monte Carlo", "5.000 cenários independentes\n(outra seed) → P(pagar S4/S8/S12),\ncaixa, risco, estoque", "#f5d0f0")
seta(ax, 3.0, 3.4, 1.95, "estatísticas", 0.3); seta(ax, 6.3, 6.7, 1.95, "plano de preços", 0.3)
seta(ax, 9.5, 9.9, 1.95, "plano completo", 0.3)
ax.annotate("", xy=(8.1, 1.2), xytext=(1.5, 1.2), arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", connectionstyle="arc3,rad=0.3"))
ax.text(4.8, 0.3, "estatísticas do MC também alimentam o Greedy (ganho esperado de um destaque)", ha="center", fontsize=8, style="italic")
ax.text(6.5, 3.25, "Dívidas nas semanas 4, 8 e 12  •  caixa inicial  •  estoque por SKU", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("arquitetura.png"), dpi=140)
