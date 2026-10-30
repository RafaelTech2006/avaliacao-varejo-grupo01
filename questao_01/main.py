import argparse
import json
import pandas as pd
from brute_force import PrecificadorSKU, BruteForce, LeitorCsv, AvaliadorMeta
from genetic_algorithm import AlgoritmoGenetico, calcular_gap_percentual


def carregar_dados(caminho_catalogo, caminho_config):
    catalogo = pd.read_csv(caminho_catalogo)
    with open(caminho_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return catalogo, config


def main():
    parser = argparse.ArgumentParser(description="Execução da Questão 1 - Operação Caixa 90")
    parser.add_argument("--seed", type=int, default=14, help="Seed base/final da instância")
    parser.add_argument("--modo", type=str, default="completo", choices=["subconjunto", "completo"], help="Modo de execução")
    args = parser.parse_args()

    print(f"\n=======================================================================================")
    print(f"       OPERAÇÃO CAIXA 90 - QUESTÃO 1 (INTEGRAÇÃO BRUTE FORCE E GENETIC ALGORITHM)        ")
    print(f"==========================================================================================\n")

    # Caminhos oficiais gerados pelo notebook
    caminho_catalogo = "data/q1_catalogo_G01.csv"
    caminho_config = "data/q1_config_G01.json"
    
    try:
        catalogo, config = carregar_dados(caminho_catalogo, caminho_config)
    except FileNotFoundError:
        print(f"[!] Erro: Ficheiros de dados não encontrados em 'data/'. Certifica-te de que os caminhos estão corretos.")
        return
    
    meta_caixa = config.get("meta_caixa_90d", 331680.98)
    piso_margem = config.get("piso_margem_minima", 0.0831)
    multiplicadores = config.get("multiplicadores_seed", [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])

    print(f"[+] Instância Carregada com Sucesso!")
    print(f" • Meta de Caixa (90d) : R$ {meta_caixa:,.2f}")
    print(f" • Piso de Margem Mín. : {piso_margem * 100:.2f}%")

    # Dicionário de preços permitidos por SKU baseado nos multiplicadores oficiais
    precos_permitidos_dict = {}
    for _, row in catalogo.iterrows():
        sku = row["sku"]
        preco_ref = row["preco_atual"]
        precos_permitidos_dict[sku] = [round(preco_ref * m, 2) for m in multiplicadores]

    # Instanciação dos componentes da Etapa A
    precificador_instancia = PrecificadorSKU(piso_margem_minimo=piso_margem)
    avaliador_meta = AvaliadorMeta(meta_caixa=meta_caixa)

    # --- 1. SELEÇÃO DO SUBCONJUNTO REDUZIDO (6 SKUs para Validação com Brute Force) ---
    # Usando a mesma lógica do LeitorCsv ou pegando os 6 primeiros
    leitor = LeitorCsv(caminho_catalogo)
    subconjunto_6 = leitor.leitura(seed=args.seed)
    
    print(f"\n--- A EXECUTAR ALGORITMO GENÉTICO NO SUBCONJUNTO (6 SKUs) ---")
    
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
    print(f" • Melhor Fitness (Caixa) Obtido pelo GA no Subconjunto: R$ {fitness_sub:,.2f}")
    print(f" • Tempo de Execução GA (Subconjunto): {metricas_sub['tempo_execucao']:.4f} segundos")
    print(f" • Pico de Memória GA (Subconjunto): {metricas_sub['memoria_pico_mb']:.4f} MB")

    
    brute_force_instancia = BruteForce(precificador=precificador_instancia)
    melhor_comb_bf, caixa_bf = brute_force_instancia.testeCombinacoes(subconjunto_6, precos_permitidos_dict)
    
    print(f" • Caixa Ótimo Brute Force (Benchmark): {'R$ {:,.2f}'.format(caixa_bf) if caixa_bf != -float('inf') else 'Inviável (-inf)'}")

    # Cálculo do Gap Percentual exigido pelo edital
    gap = calcular_gap_percentual(caixa_bf, fitness_sub)
    print(f" • Gap Percentual em relação ao Brute Force: {f'{gap:.2f}%' if gap is not None else 'N/A'}")

    if args.modo == "completo":
        print(f"\n--- A EXECUTAR ALGORITMO GENÉTICO NO CATÁLOGO COMPLETO ---")
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
        
        # Avaliação final da meta de caixa
        status_meta = avaliador_meta.avaliar(fitness_comp)
        
        print(f" • Melhor Caixa Gerado (Catálogo Completo): R$ {fitness_comp:,.2f}")
        print(f" • Status da Meta : {status_meta['status']} (Diferença: R$ {status_meta['diferenca']:,.2f})")
        print(f" • Tempo de Execução (Completo): {metricas_comp['tempo_execucao']:.4f} segundos")
        print(f" • Pico de Memória (Completo): {metricas_comp['memoria_pico_mb']:.4f} MB")
        print(f"\n====================================================================\n")


if __name__ == "__main__":
    main()