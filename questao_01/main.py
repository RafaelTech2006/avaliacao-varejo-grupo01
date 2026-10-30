"""Questão 1 — Operação Caixa 90.  Pipeline: Brute Force -> GA -> Greedy.

Exemplos:
    python questao_01/main.py --seed 14 --modo completo
    python questao_01/main.py --modo subconjunto
    python questao_01/main.py --meta-mult 1.10 --orcamento-mult 0.75 --mutacao 0.2   # cenário-surpresa
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.dados import DATA_DIR, carregar_instancia_bruta, verificar_manifesto
from questao_01 import otimizacao
from questao_01.brute_force import brute_force, selecionar_subconjunto
from questao_01.genetic_algorithm import (AlgoritmoGenetico, ConfigGA, gap_percentual,
                                          otimo_separavel)
from questao_01.greedy import GreedyIncentivos
from questao_01.modelo import ModeloQ1, aplicar_cenario, status_meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Questão 1 - Operação Caixa 90")
    p.add_argument("--grupo", default="G01")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--saida", default=str(RAIZ / "questao_01" / "resultados"))
    p.add_argument("--seed", type=int, default=None, help="default: seed_final da instância")
    p.add_argument("--modo", choices=["subconjunto", "completo"], default="completo")
    p.add_argument("--tam-subconjunto", type=int, default=6)
    # hiperparâmetros do GA (configuração A; a B é derivada para comparação)
    p.add_argument("--populacao", type=int, default=50)
    p.add_argument("--geracoes", type=int, default=100)
    p.add_argument("--mutacao", type=float, default=0.05)
    p.add_argument("--crossover", type=float, default=0.8)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--paciencia", type=int, default=30)
    # incentivos (Greedy)
    p.add_argument("--niveis-incentivo", type=float, nargs="+", default=[0.02, 0.04, 0.06])
    p.add_argument("--taxa-resgate-base", type=float, default=0.25)
    # cenário-surpresa
    p.add_argument("--meta-mult", type=float, default=1.0)
    p.add_argument("--orcamento-mult", type=float, default=1.0)
    p.add_argument("--custo-log-mult", type=float, default=1.0)
    p.add_argument("--divida-mult", type=float, default=1.0)
    p.add_argument("--remover-sku", nargs="*", default=[])
    p.add_argument("--sem-graficos", action="store_true")
    return p.parse_args(argv)


def brl(x):
    return f"R$ {x:,.2f}"


def executar(args) -> dict:
    data_dir = Path(args.data_dir)
    problemas = verificar_manifesto(data_dir) if (data_dir / "manifesto_dados.json").exists() else []
    df, config = carregar_instancia_bruta(1, args.grupo, data_dir)
    seed = args.seed if args.seed is not None else int(config["seed_final"])
    df, config = aplicar_cenario(df, config, meta_mult=args.meta_mult, orcamento_mult=args.orcamento_mult,
                                 custo_log_mult=args.custo_log_mult, divida_mult=args.divida_mult,
                                 remover_skus=args.remover_sku)
    modelo = ModeloQ1.construir(df, config)
    meta, orcamento, divida = config["meta_caixa_90d"], config["orcamento_promocional"], config["divida_total_90d"]
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f" QUESTÃO 1 — OPERAÇÃO CAIXA 90   (grupo {args.grupo}, seed {seed})")
    print("=" * 72)
    print("Integridade dos dados (SHA-256):", "OK" if not problemas else problemas)
    print(f"SKUs: {modelo.n} (ativos: {len(modelo.ativos)}) | níveis de preço: {modelo.niveis.tolist()}")
    print(f"Meta de caixa: {brl(meta)} | Dívida 90d: {brl(divida)} | Orçamento incentivos: {brl(orcamento)} "
          f"| Piso de margem: {modelo.piso:.2%}")
    inativos = [modelo.skus[i] for i in range(modelo.n) if i not in set(modelo.ativos)]
    if inativos:
        print(f"[aviso] SKUs sem preço viável (não vendem): {inativos}")

    resumo = {"grupo": args.grupo, "seed": seed, "integridade_ok": not problemas,
              "parametros": {"meta": meta, "divida": divida, "orcamento": orcamento, "piso_margem": modelo.piso,
                             "niveis": modelo.niveis.tolist(), "cenario": {
                                 "meta_mult": args.meta_mult, "orcamento_mult": args.orcamento_mult,
                                 "custo_log_mult": args.custo_log_mult, "divida_mult": args.divida_mult,
                                 "remover_sku": args.remover_sku}}}

    # ---------------- Etapa A: Brute Force no subconjunto ----------------
    sub = selecionar_subconjunto(modelo, args.tam_subconjunto, seed)
    bf = brute_force(modelo, sub)
    print(f"\n--- ETAPA A: BRUTE FORCE ({len(sub)} SKUs: {[modelo.skus[i] for i in sub]}) ---")
    print(f"Combinações avaliadas: {bf['combinacoes']:,} | caixa ótimo: {brl(bf['caixa'])} "
          f"| tempo {bf['tempo_s']:.3f}s | memória {bf['memoria_pico_mb']:.3f} MB")

    # ---------------- Etapa B1: GA no mesmo subconjunto (validação) ----------------
    cfg_a = ConfigGA(nome="A", tamanho_populacao=args.populacao, geracoes=args.geracoes,
                     taxa_mutacao=args.mutacao, taxa_crossover=args.crossover, n_elite=args.elite,
                     paciencia=args.paciencia)
    ga_sub = AlgoritmoGenetico(modelo, cfg_a, posicoes=sub, seed=seed).otimizar()
    gap = gap_percentual(bf["caixa"], ga_sub.fitness)
    print(f"\n--- ETAPA B (validação): GA NO MESMO SUBCONJUNTO ---")
    print(f"Caixa GA: {brl(ga_sub.fitness)} | gap vs Brute Force: {gap:.4f}% | melhor na geração "
          f"{ga_sub.geracao_melhor}/{ga_sub.geracoes_executadas} | {ga_sub.tempo_s:.4f}s")
    tab_bf_ga = pd.DataFrame([
        {"metodo": "Brute Force", "caixa": bf["caixa"], "gap_pct": 0.0, "avaliacoes": bf["combinacoes"],
         "tempo_s": bf["tempo_s"], "memoria_mb": bf["memoria_pico_mb"]},
        {"metodo": "Algoritmo Genético", "caixa": ga_sub.fitness, "gap_pct": gap, "avaliacoes": ga_sub.avaliacoes,
         "tempo_s": ga_sub.tempo_s, "memoria_mb": ga_sub.memoria_pico_mb}])
    tab_bf_ga.to_csv(saida / "q1_bf_vs_ga.csv", index=False)
    resumo["brute_force_vs_ga"] = {"skus": [modelo.skus[i] for i in sub], "caixa_bf": bf["caixa"],
                                   "caixa_ga": ga_sub.fitness, "gap_pct": gap,
                                   "politica_bf": bf["idx"].tolist(), "politica_ga": ga_sub.idx.tolist()}

    if args.modo == "completo":
        # ---------------- Etapa B2: GA no catálogo completo (2 configurações) ----------------
        ativos = modelo.ativos
        cfg_b = replace(cfg_a, nome="B", tamanho_populacao=cfg_a.tamanho_populacao * 2,
                        geracoes=cfg_a.geracoes * 2, taxa_mutacao=max(cfg_a.taxa_mutacao / 2, 0.005),
                        paciencia=cfg_a.paciencia * 2)
        res = {}
        print("\n--- ETAPA B: GA NO CATÁLOGO COMPLETO (comparação de configurações) ---")
        for cfg in (cfg_a, cfg_b):
            r = AlgoritmoGenetico(modelo, cfg, posicoes=ativos, seed=seed).otimizar()
            res[cfg.nome] = (cfg, r)
            print(f"Config {cfg.nome}: pop={cfg.tamanho_populacao} ger<={cfg.geracoes} mut={cfg.taxa_mutacao} "
                  f"-> caixa {brl(r.fitness)} | melhor na ger. {r.geracao_melhor} (de {r.geracoes_executadas}) "
                  f"| {r.tempo_s:.3f}s | {r.memoria_pico_mb:.3f} MB")
        nome_melhor = max(res, key=lambda k: res[k][1].fitness)
        cfg_ok, r_ok = res[nome_melhor]
        idx = np.zeros(modelo.n, dtype=int)
        idx[ativos] = r_ok.idx
        otimo = otimo_separavel(modelo)
        caixa_exato = float(modelo.caixa[ativos, otimo[ativos]].sum())
        gap_full = gap_percentual(caixa_exato, r_ok.fitness)
        print(f"Política escolhida: config {nome_melhor}. Ótimo exato (separável): {brl(caixa_exato)} "
              f"-> gap do GA: {gap_full:.4f}%")

        # ---------------- Etapa C: Greedy sobre a política do GA ----------------
        gr = GreedyIncentivos(modelo, orcamento, args.niveis_incentivo, args.taxa_resgate_base).alocar(idx)
        base = modelo.resumo_politica(idx)
        caixa_liquido = base["caixa"] + gr["ganho_liquido_total"]
        st_antes, st_depois = status_meta(base["caixa"], meta), status_meta(caixa_liquido, meta)
        vendas_final = np.where(modelo.viavel[np.arange(modelo.n), idx], gr["vendas_final"], 0.0)
        receita_final = float((vendas_final * modelo.preco[np.arange(modelo.n), idx]).sum())
        estoque0 = modelo.df["estoque"].to_numpy(float)
        estoque_rem = float((estoque0 - vendas_final).sum())
        print("\n--- ETAPA C: GREEDY (incentivos) SOBRE A POLÍTICA DO GA ---")
        print(f"Orçamento usado: {brl(gr['orcamento_usado'])} de {brl(orcamento)} "
              f"({gr['orcamento_usado'] / orcamento:.1%}) | SKUs beneficiados: {gr['skus_beneficiados']}")
        print(f"Caixa antes do Greedy: {brl(base['caixa'])} -> depois (líquido de incentivos): {brl(caixa_liquido)} "
              f"(ganho {brl(gr['ganho_liquido_total'])})")
        print(f"Tempo {gr['tempo_s']:.4f}s | memória {gr['memoria_pico_mb']:.4f} MB")

        print("\n=== RESULTADO FINAL ===")
        print(f"Caixa gerado (líquido)    : {brl(caixa_liquido)}")
        print(f"Meta de caixa             : {brl(meta)} -> {st_depois['status']}: {brl(st_depois['folga'])}")
        print(f"Margem global             : {caixa_liquido / receita_final:.2%}  (piso unitário {modelo.piso:.2%})")
        print(f"Estoque remanescente      : {estoque_rem:,.0f} un ({estoque_rem / estoque0.sum():.1%}) ")
        print(f"Cobertura da dívida (90d) : {caixa_liquido / divida:.2f}x  (dívida {brl(divida)})")
        print(f"Uso do orçamento promo    : {gr['orcamento_usado'] / orcamento:.1%}")

        # tabelas e saídas
        pol = pd.DataFrame({"sku": modelo.skus, "nivel_preco": modelo.niveis[idx],
                            "preco_ga": modelo.preco[np.arange(modelo.n), idx],
                            "ativo": modelo.viavel[np.arange(modelo.n), idx],
                            "vendas_ga": modelo.vendas[np.arange(modelo.n), idx],
                            "caixa_ga": modelo.caixa[np.arange(modelo.n), idx],
                            "margem_unit": modelo.margem[np.arange(modelo.n), idx]})
        pol.to_csv(saida / "q1_politica_precos.csv", index=False)
        ad = pd.DataFrame({"sku": modelo.skus, "incentivo_pct": np.where(gr["degrau"] >= 0,
                           np.asarray(args.niveis_incentivo)[np.maximum(gr["degrau"], 0)], 0.0),
                           "vendas_antes": gr["vendas_antes_sku"], "vendas_depois": gr["vendas_final"],
                           "caixa_antes": gr["caixa_antes_sku"], "caixa_depois_liquido": gr["caixa_final_sku"],
                           "gasto_incentivo": gr["gasto"]})
        ad["ganho_liquido"] = ad["caixa_depois_liquido"] - ad["caixa_antes"]
        ad[ad["incentivo_pct"] > 0].to_csv(saida / "q1_antes_depois_greedy.csv", index=False)

        comp = pd.DataFrame([{"config": k, "populacao": c.tamanho_populacao, "geracoes_max": c.geracoes,
                              "mutacao": c.taxa_mutacao, "geracao_melhor": r.geracao_melhor,
                              "geracoes_executadas": r.geracoes_executadas, "caixa": r.fitness,
                              "tempo_s": r.tempo_s, "memoria_mb": r.memoria_pico_mb} for k, (c, r) in res.items()])
        comp.to_csv(saida / "q1_comparacao_ga.csv", index=False)

        otim = otimizacao.comparar(modelo)
        print(f"\nOtimização do fitness: antes {otim['tempo_antes_s']:.3f}s -> depois {otim['tempo_depois_s']:.5f}s "
              f"({otim['speedup']:.0f}x) para {otim['avaliacoes']} avaliações")

        if not args.sem_graficos:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for k, (c, r) in res.items():
                h = pd.DataFrame(r.historico)
                ax.plot(h["geracao"], h["melhor_global"], label=f"Config {k}: melhor (pop {c.tamanho_populacao})")
                ax.plot(h["geracao"], h["media"], linestyle=":", label=f"Config {k}: média da população")
            ax.axhline(caixa_exato, color="k", linestyle="--", linewidth=0.8, label="Ótimo exato (separável)")
            ax.set_xlabel("Geração"); ax.set_ylabel("Fitness (caixa, R$)"); ax.set_title("Q1 — Fitness por geração do GA")
            ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(saida / "q1_fitness_por_geracao.png", dpi=130); plt.close(fig)

        resumo["ga_completo"] = {"configs": comp.to_dict("records"), "config_escolhida": nome_melhor,
                                 "caixa_ga": r_ok.fitness, "caixa_otimo_exato_separavel": caixa_exato,
                                 "gap_vs_exato_pct": gap_full}
        resumo["greedy"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gr.items()
                            if k in ("orcamento", "orcamento_usado", "orcamento_restante", "ganho_liquido_total",
                                     "skus_beneficiados", "tempo_s", "memoria_pico_mb")}
        resumo["resultado_final"] = {
            "caixa_antes_greedy": base["caixa"], "caixa_liquido_final": caixa_liquido, "meta": meta,
            "folga_ou_insuficiencia": st_depois["folga"], "meta_atingida": st_depois["meta_atingida"],
            "margem_global": caixa_liquido / receita_final, "estoque_remanescente": estoque_rem,
            "estoque_remanescente_pct": estoque_rem / float(estoque0.sum()),
            "cobertura_divida": caixa_liquido / divida, "uso_orcamento_pct": gr["orcamento_usado"] / orcamento,
            "status_antes_greedy": st_antes["status"], "status_final": st_depois["status"]}
        resumo["otimizacao_codigo"] = otim
    with open(saida / "q1_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nArquivos gravados em {saida}")
    return resumo


if __name__ == "__main__":
    executar(parse_args())
