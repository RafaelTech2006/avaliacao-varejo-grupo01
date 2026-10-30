import argparse
import json
import pandas as pd
from brute_force import PrecificadorSKU, BruteForce, LeitorCsv, AvaliadorMeta
from genetic_algorithm import AlgoritmoGenetico, calcular_gap_percentual
from greedy import GreedyIncentivos


def carregar_dados(caminho_catalogo, caminho_config):
    """Carrega o catálogo CSV e as configurações JSON oficiais da instância"""
    catalogo = pd.read_csv(caminho_catalogo)
    with open(caminho_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return catalogo, config


def main():
    parser = argparse.ArgumentParser(description="Execução da Questão 1 - Operação Caixa 90")
    parser.add_argument("--seed", type=int, default=14, help="Seed base/final da instância")
    parser.add_argument("--modo", type=str, default="completo", choices=["subconjunto", "completo"], help="Modo de execução")
    args = parser.parse_args()

    print(f"\n====================================================================")
    print(f"  OPERAÇÃO CAIXA 90 - QUESTÃO 1 (INTEGRAÇÃO COMPLETA: BRUTE FORCE, GENETIC ALGORITHM & GREEDY)      ")
    print(f"====================================================================\n")

    # Caminhos oficiais gerados pelo notebook gerador do grupo
    caminho_catalogo = "data/q1_catalogo_G01.csv"
    caminho_config = "data/q1_config_G01.json"
    
    try:
        catalogo, config = carregar_dados(caminho_catalogo, caminho_config)
    except FileNotFoundError:
        print(f"[!] Erro: Ficheiros de dados não encontrados em 'data/'. Certifica-te de executar o gerador oficial.")
        return
    
    meta_caixa = config.get("meta_caixa_90d", 331680.98)
    piso_margem = config.get("piso_margem_minima", 0.0831)
    multiplicadores = config.get("multiplicadores_seed", [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    orcamento_promocional = config.get("orcamento_promocional", 5000.0)

    print(f"[+] Instância Carregada com Sucesso!")
    print(f" • Meta de Caixa (90d) : R$ {meta_caixa:,.2f}")
    print(f" • Piso de Margem Mín. : {piso_margem * 100:.2f}%")
    print(f" • Orçamento Promocional: R$ {orcamento_promocional:,.2f}")

    # Dicionário de preços permitidos por SKU baseado nos multiplicadores oficiais
    precos_permitidos_dict = {}
    for _, row in catalogo.iterrows():
        sku = row["sku"]
        preco_ref = row["preco_atual"]
        precos_permitidos_dict[sku] = [round(preco_ref * m, 2) for m in multiplicadores]

    # Instanciação dos componentes comuns de negócio
    precificador_instancia = PrecificadorSKU(piso_margem_minimo=piso_margem)
    avaliador_meta = AvaliadorMeta(meta_caixa=meta_caixa)

    # --- 1. ETAPA A: BRUTE FORCE (Subconjunto de 6 SKUs) ---
    leitor = LeitorCsv(caminho_catalogo)
    subconjunto_6 = leitor.leitura(seed=args.seed)
    
    print(f"\n--- ETAPA A: VALIDAÇÃO COM BRUTE FORCE (6 SKUs) ---")
    brute_force_instancia = BruteForce(precificador=precificador_instancia)
    melhor_comb_bf, caixa_bf = brute_force_instancia.testeCombinacoes(subconjunto_6, precos_permitidos_dict)
    
    print(f" • Caixa Ótimo Brute Force (Benchmark): {'R$ {:,.2f}'.format(caixa_bf) if caixa_bf != -float('inf') else 'Inviável (-inf)'}")

    # --- 2. ETAPA B: ALGORITMO GENÉTICO (Subconjunto Reduzido) ---
    print(f"\n--- ETAPA B: ALGORITMO GENÉTICO (Subconjunto Reduzido) ---")
    ga_subconjunto = AlgoritmoGenetico(
        subconjunto_skus=subconjunto_6,
        precos_permitidos=precos_permitidos_dict,
        precificador=precificador_instancia,
        avaliador=avaliador_meta,
        tamanho_populacao=30,
        geracoes=50,
        seed=args.seed
    )

    melhor_sol_sub, fitness_sub, metricas_sub = ga_subconjunto.otimizar()
    print(f" • Melhor Fitness (Caixa) GA no Subconjunto: R$ {fitness_sub:,.2f}")
    print(f" • Desempenho GA (Subconjunto) -> Tempo: {metricas_sub['tempo_execucao']:.4f}s | Memória: {metricas_sub['memoria_pico_mb']:.4f} MB")

    # Cálculo do Gap Percentual
    gap = calcular_gap_percentual(caixa_bf, fitness_sub)
    print(f" • Gap Percentual (Brute Force vs GA): {f'{gap:.2f}%' if gap is not None else 'N/A'}")

    
    if args.modo == "completo":
        # --- ETAPA B: ALGORITMO GENÉTICO NO CATÁLOGO COMPLETO ---
        print(f"\n--- ETAPA B: ALGORITMO GENÉTICO NO CATÁLOGO COMPLETO ---")
        ga_completo = AlgoritmoGenetico(
            subconjunto_skus=catalogo,
            precos_permitidos=precos_permitidos_dict,
            precificador=precificador_instancia,
            avaliador=avaliador_meta,
            tamanho_populacao=50,
            geracoes=100,
            seed=args.seed
        )

        melhor_sol_comp, fitness_comp, metricas_comp = ga_completo.otimizar()
        print(f" • Caixa Gerado pelo GA (Sem Promos): R$ {fitness_comp:,.2f}")
        print(f" • Desempenho GA (Completo) -> Tempo: {metricas_comp['tempo_execucao']:.4f}s | Memória: {metricas_comp['memoria_pico_mb']:.4f} MB")

        # --- 3. ETAPA C: ALGORITMO GREEDY (Alocação de Incentivos Comerciais) ---
        print(f"\n--- ETAPA C: ALGORITMO GREEDY SOBRE A POLÍTICA DO GA ---")
        greedy_instancia = GreedyIncentivos(precificador=precificador_instancia, orcamento=orcamento_promocional)
        res_greedy = greedy_instancia.alocar_incentivos(catalogo, melhor_sol_comp)

        caixa_final_acumulado = fitness_comp + res_greedy["ganho_marginal_total"]
        status_meta = avaliador_meta.avaliar(caixa_final_acumulado)

        print(f" • Orçamento Promocional Alocado: R$ {res_greedy['orcamento_usado']:,.2f} de R$ {orcamento_promocional:,.2f}")
        print(f" • SKUs Beneficiados com Promos  : {res_greedy['total_skus_beneficiados']}")
        print(f" • Ganho Marginal de Caixa       : R$ {res_greedy['ganho_marginal_total']:,.2f}")
        print(f" • Caixa Operacional Final Total : R$ {caixa_final_acumulado:,.2f}")
        print(f" • Status Final da Meta          : {status_meta['status']} (Diferença: R$ {status_meta['diferenca']:,.2f})")
        print(f" • Desempenho Greedy -> Tempo: {res_greedy['tempo_execucao']:.4f}s | Memória: {res_greedy['memoria_pico_mb']:.4f} MB")
        print(f"\n====================================================================\n")


if __name__ == "__main__":
    main()