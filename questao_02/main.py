"""Questão 2 — Plano de Sobrevivência em 12 Semanas.  Pipeline: Monte Carlo -> DP -> Greedy -> Monte Carlo.

Exemplos:
    python questao_02/main.py --seed 14 --cenarios 5000
    python questao_02/main.py --divida-mult 1.15 --concorrente-delta -0.08 --custo-log-mult 1.15   # cenário-surpresa
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import DATA_DIR, verificar_manifesto
from questao_02 import dynamic_programming as dp
from questao_02 import greedy, monte_carlo as mc
from questao_02.modelo import Instancia, custo_promo


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Questão 2 - Plano de Sobrevivência em 12 Semanas")
    p.add_argument("--grupo", default="G01")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--saida", default=str(RAIZ / "questao_02" / "resultados"))
    p.add_argument("--seed", type=int, default=None, help="default: seed_final da instância")
    p.add_argument("--cenarios", type=int, default=5000, help="cenários da validação final (>= 5000)")
    p.add_argument("--cenarios-estimacao", type=int, default=2000, help="cenários da estimação (>= 2000)")
    p.add_argument("--lambda-risco", type=float, default=0.5, help="aversão a risco (desvio-padrão) na DP")
    p.add_argument("--beta-urgencia", type=float, default=0.10, help="peso extra para caixa antes dos vencimentos")
    p.add_argument("--uplift", type=float, default=0.20, help="aumento de demanda com destaque")
    p.add_argument("--custo-destaque-pct", type=float, default=0.05, help="custo do destaque (% receita nominal semanal)")
    p.add_argument("--percentil-risco", type=float, default=None, help="default: 100*limite_risco_falha da instância")
    # cenário-surpresa
    p.add_argument("--divida-mult", type=float, default=1.0)
    p.add_argument("--concorrente-delta", type=float, default=0.0, help="ex.: -0.08 = concorrente 8%% mais barato")
    p.add_argument("--custo-log-mult", type=float, default=1.0)
    p.add_argument("--orcamento-mult", type=float, default=1.0)
    p.add_argument("--sigma-mult", type=float, default=1.0)
    p.add_argument("--remover-sku", nargs="*", default=[])
    p.add_argument("--sem-graficos", action="store_true")
    return p.parse_args(argv)


def brl(x):
    return f"R$ {x:,.2f}"


def executar_pipeline(inst: Instancia, seed: int, n_val: int, n_est: int, lam: float, beta: float, uplift: float,
                      custo_pct: float) -> dict:
    """Roda as 4 etapas medindo tempo e pico de memória de cada uma."""
    medidas, out = {}, {}

    def etapa(nome, fn):
        tracemalloc.start()
        t0 = time.perf_counter()
        r = fn()
        t = time.perf_counter() - t0
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        medidas[nome] = {"tempo_s": t, "memoria_pico_mb": pico / 2**20}
        return r

    est = etapa("monte_carlo_estimacao", lambda: mc.estimar(inst, n_est, seed, uplift))
    sol = etapa("programacao_dinamica", lambda: dp.otimizar(inst, est, lam, beta))
    plano, registro = etapa("greedy", lambda: greedy.alocar(inst, est, sol["nivel_idx"], custo_pct=custo_pct))
    plano_sem_destaque = mc.PlanoCompleto(plano.nivel_idx, np.zeros_like(plano.destaque), plano.custo_destaque)
    seed_val = seed + 10_000                          # seed distinta -> cenários independentes da estimação
    val = etapa("monte_carlo_validacao", lambda: mc.validar(inst, plano, n_val, seed_val, uplift))
    out.update(est=est, sol=sol, plano=plano, registro=registro, plano_sem_destaque=plano_sem_destaque,
               val=val, medidas=medidas, seed_val=seed_val)
    return out


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    problemas = verificar_manifesto(data_dir) if (data_dir / "manifesto_dados.json").exists() else []
    base = Instancia.de_arquivos(args.grupo, data_dir)
    seed = args.seed if args.seed is not None else int(json.load(open(data_dir / f"q2_config_{args.grupo}.json"))["seed_final"])
    inst = base.com_cenario(divida_mult=args.divida_mult, concorrente_delta=args.concorrente_delta,
                            custo_log_mult=args.custo_log_mult, orcamento_mult=args.orcamento_mult,
                            sigma_mult=args.sigma_mult, remover_skus=args.remover_sku)
    if args.cenarios < 5000:
        print(f"[aviso] o enunciado exige >= 5000 cenários na validação (recebido {args.cenarios}).")
    if args.cenarios_estimacao < 2000:
        print(f"[aviso] o enunciado exige >= 2000 cenários na estimação (recebido {args.cenarios_estimacao}).")
    pct_risco = args.percentil_risco if args.percentil_risco is not None else 100 * inst.limite_risco
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f" QUESTÃO 2 — PLANO DE SOBREVIVÊNCIA EM {inst.n_semanas} SEMANAS  (grupo {args.grupo}, seed {seed})")
    print("=" * 72)
    print("Integridade dos dados (SHA-256):", "OK" if not problemas else problemas)
    print(f"SKUs: {inst.n} | níveis de preço: {inst.niveis.tolist()} | caixa inicial {brl(inst.caixa_inicial)}")
    print("Dívidas:", " | ".join(f"semana {d}: {brl(v)}" for d, v in inst.dividas.items()),
          f"| orçamento promo semanal {brl(inst.orcamento_semanal)}")
    print(f"Variabilidade: demanda {inst.sigma_dem:.4f} | concorrente {inst.sigma_conc:.4f} | logística {inst.sigma_log:.4f}")

    r = executar_pipeline(inst, seed, args.cenarios, args.cenarios_estimacao, args.lambda_risco,
                          args.beta_urgencia, args.uplift, args.custo_destaque_pct)
    est, sol, plano, val = r["est"], r["sol"], r["plano"], r["val"]
    # referências com os MESMOS cenários da validação (seed idêntica)
    nivel_fixo = np.full_like(plano.nivel_idx, int(np.argmin(np.abs(inst.niveis - 1.0))))
    sem_plano = mc.PlanoCompleto(nivel_fixo, np.zeros_like(plano.destaque), plano.custo_destaque)
    val_fixa = mc.validar(inst, sem_plano, args.cenarios, r["seed_val"], args.uplift)
    val_dp = mc.validar(inst, r["plano_sem_destaque"], args.cenarios, r["seed_val"], args.uplift)

    print("\n[1] Monte Carlo (estimação):", f"{est.n_cenarios} cenários -> pmf de vendas por (semana, SKU, preço).")
    print(f"[2] Programação Dinâmica: {sol['subproblemas']:,} subproblemas (semana,estoque) resolvidos 1x cada; "
          f"{sol['transicoes']:,} transições avaliadas (sem memoização: {sol['trajetorias_sem_memo_por_sku']:,} trajetórias/SKU).")
    n_dest = int(plano.destaque.sum())
    gasto = float((plano.custo_destaque * plano.destaque).sum())
    print(f"[3] Greedy: {n_dest} destaques concedidos (gasto total {brl(gasto)}; "
          f"orçamento total {brl(inst.orcamento_semanal * inst.n_semanas)}).")
    print(f"[4] Monte Carlo (validação): {val['n_cenarios']} cenários independentes (seed {val['seed']}).")

    def linha(nome, v):
        pp = " | ".join(f"S{d}: {v['prob_pagamento'][d]:.1%}" for d in inst.semanas_vencimento)
        print(f"  {nome:<26} {pp} | todas: {v['prob_todas']:.1%} | caixa final médio {brl(v['caixa_final_medio'])}")
    print("\n=== PROBABILIDADE DE PAGAR NO PRAZO (validação) ===")
    linha("Política fixa (preço 1,0)", val_fixa)
    linha("DP (sem destaques)", val_dp)
    linha("DP + Greedy (plano final)", val)
    meta_prob = 1 - inst.limite_risco
    print(f"\n=== PLANO FINAL ===")
    print(f"Caixa final esperado (após todas as dívidas): {brl(val['caixa_final_medio'])}")
    print(f"Percentil de risco escolhido P{pct_risco:g}: {brl(float(np.percentile(val['caixa_final_cenarios'], pct_risco)))}  "
          f"(P5 {brl(val['caixa_final_p5'])} | P50 {brl(val['caixa_final_p50'])} | P95 {brl(val['caixa_final_p95'])})")
    print(f"Margem média: {val['margem_media']:.2%} | Estoque final médio: {val['estoque_final_medio_un']:,.0f} un "
          f"({val['estoque_final_pct']:.1%} do inicial)")
    print(f"Cenários de falha (qualquer dívida): {val['cenarios_falha_qualquer']} de {val['n_cenarios']} "
          f"| por semana: {val['cenarios_falha_por_semana']}")
    atende = all(val['prob_pagamento'][d] >= meta_prob for d in inst.semanas_vencimento)
    print(f"Critério de risco da instância (falha <= {inst.limite_risco:.0%} em cada vencimento): "
          f"{'ATENDIDO' if atende else 'NÃO ATENDIDO'}")
    print("\nDesempenho por etapa:")
    for k, v in r["medidas"].items():
        print(f"  {k:<26} {v['tempo_s']:.3f}s | {v['memoria_pico_mb']:.2f} MB")

    # ---------------- arquivos ----------------
    W = inst.n_semanas
    linhas = []
    for i, sku in enumerate(inst.skus):
        for w in range(W):
            linhas.append({"sku": sku, "semana": w + 1, "nivel_preco": inst.niveis[plano.nivel_idx[i, w]],
                           "preco": inst.pref[i] * inst.niveis[plano.nivel_idx[i, w]],
                           "destaque": bool(plano.destaque[i, w]), "estoque_inicio_semana_esperado": sol["estoque_esperado"][i, w],
                           "estoque_medio_simulado": val["estoque_medio_por_semana"][w, i]})
    pd.DataFrame(linhas).to_csv(saida / "q2_plano_precos.csv", index=False)
    pd.DataFrame(r["registro"]).to_csv(saida / "q2_greedy_destaques.csv", index=False)
    pd.DataFrame([{"politica": n, **{f"prob_semana_{d}": v["prob_pagamento"][d] for d in inst.semanas_vencimento},
                   "prob_todas": v["prob_todas"], "caixa_final_medio": v["caixa_final_medio"],
                   "caixa_final_p5": v["caixa_final_p5"], "caixa_final_p10": v["caixa_final_p10"],
                   "falhas": v["cenarios_falha_qualquer"]}
                  for n, v in [("fixa_1.0", val_fixa), ("dp_sem_destaques", val_dp), ("dp_mais_greedy", val)]]
                 ).to_csv(saida / "q2_probabilidades.csv", index=False)
    pd.DataFrame([{"etapa": k, **v} for k, v in r["medidas"].items()]).to_csv(saida / "q2_desempenho.csv", index=False)
    resumo = {"grupo": args.grupo, "seed": seed, "seed_validacao": r["seed_val"], "integridade_ok": not problemas,
              "cenarios_estimacao": est.n_cenarios, "cenarios_validacao": val["n_cenarios"],
              "parametros": {"lambda_risco": args.lambda_risco, "beta_urgencia": args.beta_urgencia,
                             "uplift": args.uplift, "custo_destaque_pct": args.custo_destaque_pct,
                             "percentil_risco": pct_risco, "dividas": inst.dividas,
                             "cenario": {"divida_mult": args.divida_mult, "concorrente_delta": args.concorrente_delta,
                                         "custo_log_mult": args.custo_log_mult, "orcamento_mult": args.orcamento_mult,
                                         "sigma_mult": args.sigma_mult, "remover_sku": args.remover_sku}},
              "dp": {k: v for k, v in sol.items() if k in ("subproblemas", "transicoes", "trajetorias_sem_memo_por_sku", "valor")},
              "greedy": {"destaques": n_dest, "gasto": gasto},
              "desempenho": r["medidas"],
              "validacao": {n: {k: v for k, v in x.items() if not isinstance(v, np.ndarray)}
                            for n, x in [("fixa_1.0", val_fixa), ("dp_sem_destaques", val_dp), ("dp_mais_greedy", val)]},
              "percentil_risco_valor": float(np.percentile(val["caixa_final_cenarios"], pct_risco)),
              "criterio_risco_atendido": bool(atende)}
    with open(saida / "q2_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=float)

    if not args.sem_graficos:
        graficos(inst, plano, sol, val, val_fixa, val_dp, pct_risco, saida)
    print(f"\nArquivos gravados em {saida}")
    return resumo


def graficos(inst, plano, sol, val, val_fixa, val_dp, pct_risco, saida):
    W = inst.n_semanas
    # distribuição do caixa final
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, nome, cor in [(val_fixa, "Política fixa (1,0)", "tab:gray"), (val_dp, "DP", "tab:orange"),
                         (val, "DP + Greedy", "tab:blue")]:
        ax.hist(v["caixa_final_cenarios"] / 1e3, bins=60, alpha=0.5, label=nome, color=cor)
    p = np.percentile(val["caixa_final_cenarios"], pct_risco) / 1e3
    ax.axvline(0, color="red", linestyle="--", label="Caixa zero após todas as dívidas")
    ax.axvline(p, color="k", linestyle=":", label=f"P{pct_risco:g} (plano final)")
    ax.set_xlabel("Caixa final após pagar as 3 dívidas (R$ mil)"); ax.set_ylabel("Cenários")
    ax.set_title("Q2 — Distribuição do caixa final (validação independente)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(saida / "q2_distribuicao_caixa.png", dpi=130); plt.close(fig)
    # saldo semanal
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sem = np.arange(1, W + 1)
    ax.plot(sem, val["saldo_medio_por_semana"] / 1e3, label="Saldo médio (após dívidas vencidas)")
    ax.plot(sem, val["saldo_p10_por_semana"] / 1e3, linestyle="--", label="P10")
    ax.axhline(0, color="red", linewidth=0.8)
    for d in inst.semanas_vencimento:
        ax.axvline(d, color="gray", linestyle=":")
    ax.set_xlabel("Semana"); ax.set_ylabel("R$ mil"); ax.set_title("Q2 — Saldo de caixa por semana"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(saida / "q2_saldo_semanal.png", dpi=130); plt.close(fig)
    # trajetórias para 3 SKUs representativos (menor, mediano e maior estoque inicial)
    ordem = np.argsort(inst.estoque0)
    rep = [ordem[0], ordem[len(ordem) // 2], ordem[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for c, i in enumerate(rep):
        axs[0, c].step(sem, inst.niveis[plano.nivel_idx[i]], where="mid")
        for w in np.flatnonzero(plano.destaque[i]):
            axs[0, c].plot(w + 1, inst.niveis[plano.nivel_idx[i, w]], "r*")
        axs[0, c].set_title(f"{inst.skus[i]} (estoque {inst.estoque0[i]})"); axs[0, c].set_ylabel("Nível de preço")
        axs[1, c].plot(np.arange(0, W + 1), sol["estoque_esperado"][i], label="esperado (DP, sem destaques)")
        axs[1, c].plot(np.arange(0, W + 1), val["estoque_medio_por_semana"][:, i], "--", label="média simulada (com destaques)")
        axs[1, c].set_xlabel("Semana"); axs[1, c].set_ylabel("Estoque")
    axs[1, 0].legend(fontsize=7)
    fig.suptitle("Q2 — Trajetória semanal de preço (★ = destaque) e estoque", y=1.0)
    fig.tight_layout(); fig.savefig(saida / "q2_trajetorias.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
