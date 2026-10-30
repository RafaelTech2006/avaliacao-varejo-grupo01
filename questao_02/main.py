import argparse
import json
import time
import tracemalloc
import pandas as pd
from monte_carlo import simular_demanda_e_caixa, validar_plano_monte_carlo
from dynamic_programming import otimizar_catalogo_dp
from greedy import alocar_promocoes_greedy

def main():
    parser = argparse.ArgumentParser(description="Questão 2 - Plano de Sobrevivência em 12 Semanas")
    parser.add_argument("--seed", type=int, default=14, help="Seed para reprodutibilidade")
    parser.add_argument("--cenarios", type=int, default=5000, help="Quantidade de cenários no Monte Carlo final")
    args = parser.parse_args()

    print(f"=== Iniciando Execução - Questão 2 (Seed: {args.seed}) ===")
    
    # 1. Carregamento de dados e configurações
    df_catalogo = pd.read_csv("data/q2_catalogo_G01.csv")
    with open("data/q2_config_G01.json", "r") as f:
        config = json.load(f)
        
    multiplicadores = config.get("multiplicadores_preco", [0.8, 0.9, 1.0, 1.1, 1.2])
    metas_semanais = {
        4: config.get("meta_caixa_semana_4", 15000),
        8: config.get("meta_caixa_semana_8", 35000),
        12: config.get("meta_caixa_semana_12", 60000)
    }

    tracemalloc.start()
    t_inicio = time.perf_counter()

    # ETAPA A: Monte Carlo Estimativo (2.000 cenários)
    print("\n[1/4] Executando Monte Carlo Inicial (2.000 cenários)...")
    est_mc = simular_demanda_e_caixa(df_catalogo, multiplicadores, n_cenarios=2000, seed=args.seed)

    # ETAPA B: Programação Dinâmica (Trajetória Semanal)
    print("[2/4] Otimizando trajetória de preços via Programação Dinâmica...")
    plano_dp, caixa_dp = otimizar_catalogo_dp(df_catalogo, est_mc, multiplicadores)

    # ETAPA C: Greedy Semanal (Alocação de Incentivos Promocionais)
    print("[3/4] Aplicando Greedy Semanal para alocação promocional...")
    plano_final, df_promo = alocar_promocoes_greedy(df_catalogo, plano_dp, orcamento_semanal=300.0)

    # ETAPA D: Validação Monte Carlo Final (Cenários independentes)
    print(f"[4/4] Validando plano em {args.cenarios} cenários Monte Carlo independentes...")
    res_validacao = validar_plano_monte_carlo(df_catalogo, plano_final, metas_semanais, n_cenarios=args.cenarios, seed=args.seed + 999)

    t_fim = time.perf_counter()
    mem_atual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Exibição de Resultados Técnicos e Financeiros
    print("\n" + "="*50)
    print("RESUMO DOS RESULTADOS DA QUESTÃO 2")
    print("="*50)
    print(f"Tempo de execução total: {t_fim - t_inicio:.4f} segundos")
    print(f"Pico de memória utilizada: {mem_pico / 10**6:.2f} MB")
    print(f"\nCaixa Médio Esperado: R$ {res_validacao['caixa_medio_final']:,.2f}")
    print(f"Caixa Percentil 5% (Risco): R$ {res_validacao['caixa_p5']:,.2f}")
    print(f"Caixa Percentil 95%: R$ {res_validacao['caixa_p95']:,.2f}")
    
    print("\nProbabilidade de Pagamento das Obrigações:")
    for sem, prob in res_validacao['probabilidades_pagamento'].items():
        print(f" - Semana {sem} (Meta R$ {metas_semanais[sem]:,.2f}): {prob * 100:.2f}% de sucesso")

if __name__ == "__main__":
    main()