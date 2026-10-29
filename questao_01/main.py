import argparse
import json
from pathlib import Path
import time
import tracemalloc

# Importação das classes modulares do brute_force.py
from brute_force import AvaliadorMeta, BruteForce, LeitorCsv, PrecificadorSKU


def main():
    # 1. Suporte a CLI com argumento de seed para reprodutibilidade (Edital Seção 6)
    parser = argparse.ArgumentParser(
        description="Operação Caixa 90 - Questão 1 (Etapa A: Brute Force Benchmark)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="Seed determinística para a amostragem dos SKUs",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("   OPERAÇÃO CAIXA 90 - QUESTÃO 1 (ETAPA A: BRUTE FORCE BENCHMARK)")
    print("=" * 68)

    # 2. Localização e validação dos ficheiros oficiais da instância (data/)
    diretorio_atual = Path(__file__).parent
    diretorio_raiz = diretorio_atual.parent

    caminho_csv = diretorio_raiz / "data" / "q1_catalogo_G01.csv"
    caminho_config = diretorio_raiz / "data" / "q1_config_G01.json"

    # Fallbacks de caminho caso os dados estejam na mesma pasta da questão
    if not caminho_csv.exists():
        caminho_csv = diretorio_atual / "q1_catalogo_G01.csv"
    if not caminho_config.exists():
        caminho_config = diretorio_atual / "q1_config_G01.json"

    if not caminho_csv.exists() or not caminho_config.exists():
        print("❌ ERRO CRÍTICO: Ficheiros oficiais de entrada não encontrados em /data.")
        print(f"   • Verifique se '{caminho_csv.name}' e '{caminho_config.name}' existem.")
        return

    # 3. Leitura dinâmica da configuração oficial da seed do grupo (Edital Seção 2.3)
    with open(caminho_config, "r", encoding="utf-8") as f:
        config_oficial = json.load(f)

    multiplicadores_preco = config_oficial["niveis_preco_relativos"]
    piso_margem_oficial = config_oficial["margem_minima_percentual"]
    meta_caixa_oficial = config_oficial["meta_caixa_90d"]

    print(f"\n[+] Instância Carregada para o Grupo {config_oficial.get('grupo', 'G01')}:")
    print(f"    • Meta de Caixa (90d) : R$ {meta_caixa_oficial:,.2f}")
    print(f"    • Piso de Margem Mín. : {piso_margem_oficial * 100:.2f}%")
    print(f"    • Multiplicadores Seed : {multiplicadores_preco}")

    # 4. Início da medição de desempenho (Tempo e Memória)
    tracemalloc.start()
    tempo_inicio = time.perf_counter()

    # 5. Amostragem determinística dos 6 SKUs usando a Seed
    print(f"\n[+] A carregar o catálogo e a selecionar 6 SKUs (Seed: {args.seed})...")
    leitor = LeitorCsv(str(caminho_csv))
    subconjunto_6_skus = leitor.leitura(seed=args.seed)

    # 6. Construção do dicionário de preços permitidos discretos
    precos_permitidos_dict = {}
    for _, row in subconjunto_6_skus.iterrows():
        sku = row["sku"]
        preco_atual = row["preco_atual"]
        precos_permitidos_dict[sku] = [
            round(preco_atual * m, 2) for m in multiplicadores_preco
        ]

    # 7. Otimização via Busca Exaustiva (Brute Force)
    precificador = PrecificadorSKU(piso_margem_minimo=piso_margem_oficial)
    otimizador_bf = BruteForce(precificador)
    avaliador = AvaliadorMeta(meta_caixa=meta_caixa_oficial)

    total_combinacoes = len(multiplicadores_preco) ** len(subconjunto_6_skus)
    print(f"[+] A testar exaustivamente todas as {total_combinacoes:,} combinações via Brute Force...")

    melhor_combinacao, melhor_caixa = otimizador_bf.testeCombinacoes(
        subconjunto_6_skus, precos_permitidos_dict
    )

    # Coleta de métricas computacionais (Performance)
    tempo_fim = time.perf_counter()
    _, memoria_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 8. Apresentação dos Resultados
    print("\n" + "=" * 68)
    print("--- 1. SUBSET DE SKUs SELECIONADO (6 SKUs) ---")
    print(
        subconjunto_6_skus[
            ["sku", "categoria", "preco_atual", "estoque", "elasticidade"]
        ].to_string(index=False)
    )

    print("\n--- 2. BENCHMARK ÓTIMO ENCONTRADO (BRUTE FORCE) ---")
    if melhor_combinacao is None or melhor_caixa == -float("inf"):
        print("⚠️ AVISO: Nenhuma combinação de preços atendeu ao piso de margem mínima para todos os SKUs em simultâneo.")
        print("   Todas as combinações foram consideradas inviáveis segundo as travas de segurança.")
    else:
        for i, sku in enumerate(subconjunto_6_skus["sku"]):
            preco_orig = subconjunto_6_skus.iloc[i]["preco_atual"]
            preco_otimo = melhor_combinacao[i]
            var_perc = ((preco_otimo - preco_orig) / preco_orig) * 100
            print(
                f"  • SKU {sku:10s} | Preço Atual: R$ {preco_orig:>8.2f} | Preço Ótimo: R$ {preco_otimo:>8.2f} ({var_perc:+5.1f}%)"
            )

        # 9. Avaliação financeira perante a meta
        relatorio = avaliador.avaliar(melhor_caixa)
        print("\n--- 3. AVALIAÇÃO DA META FINANCEIRA ---")
        print(f"  • Caixa Total Gerado  : R$ {relatorio['caixa_gerado']:>14,.2f}")
        print(f"  • Meta de Caixa (90d) : R$ {relatorio['meta_caixa']:>14,.2f}")
        print(f"  • Diferença / Folga   : R$ {relatorio['diferenca']:>14,.2f}")
        print(f"  • Status Final        : {relatorio['status']}")

    print("\n--- 4. EVIDÊNCIAS COMPUTAÇÃO (BIG O & DESEMPENHO) ---")
    print(f"  • Tempo de Execução   : {tempo_fim - tempo_inicio:.4f} segundos")
    print(f"  • Pico de Memória     : {memoria_pico / (1024 * 1024):.4f} MB")
    print(f"  • Combinações Lidas   : {total_combinacoes:,} cenários")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()