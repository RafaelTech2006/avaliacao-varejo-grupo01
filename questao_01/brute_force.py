import itertools 
import pandas as pd

class LeitorCsv:
    """Classe responsável pela leitura e extração de informações do CSV."""

    def __init__(self, caminho : str) -> None:
        """Inicializando a instância do objeto caminho."""
        self.caminho = caminho

    def leitura(self, seed: int = 9) -> pd.DataFrame:
        """Carregando o CSV e retornando 6 dados aleatórios do DataFrame."""
        return pd.read_csv(self.caminho).sample(n=6, random_state=seed)

class PrecificadorSKU:
    """
        Classe responsável por aplicar o modelo matemático
        de precificação, elasticidade e validação de margem
        para um único SKU.
    """
    def __init__(self, piso_margem_minimo: float = 0.0831):
        self.piso_margem_minima = piso_margem_minimo

    def calcular_caixa_produto(self, preco_candidato: float, row: pd.Series) -> float:
        """Calculando o caixa gerado para um SKU dado um preço candidato"""

        # Variação percentual do preço em relação ao preço atual
        var_preco = (preco_candidato - row["preco_atual"]) / row["preco_atual"]

        # Elasticidade e variação da demanda
        var_demanda = -row["elasticidade"] * var_preco
        demanda_novo = row["demanda_base_90d"] * (1 + var_demanda)
        demanda_novo = max(0, demanda_novo) # Demanda não pode ser negativa

        # Trava de estoque
        qtd_vendida = min(demanda_novo, row["estoque"])

        # Receita líquida e custo total unitário
        receita_liquida = preco_candidato * (1 - row["taxa_marketplace"])
        custo_total_unitario = row["custo_unitario"] + row["custo_logistico"]
        caixa_unitario = receita_liquida - custo_total_unitario

        # Validação do piso de margem 
        margem_unitaria = (
            caixa_unitario / preco_candidato if preco_candidato > 0 else 0
        )
        if margem_unitaria < self.piso_margem_minima:
            return -float("inf") # Combinação inviável por violação de margem

        # Caixa total gerado pelo SKU
        return qtd_vendida * caixa_unitario

class BruteForce:
    """
        Executando o algoritmo Brute Force sobre todas as
        combinações permitidas de preço em um subconjunto 
        com 6 produtos.
    """
    def __init__(self, precificador: PrecificadorSKU):
        self.precificador = precificador

    def testeCombinacoes(self, subconjunto: pd.DataFrame, precos_permitidos: dict):
        """
            Testa todas as cobinações de preços e
            retorna a melhor cixa e combinação.
        """
        skus = subconjunto["sku"].to_list()
        precos = [precos_permitidos[sku] for sku in skus]

        melhor_caixa = -float("inf")
        melhor_combinacao = None

        # testando todas as combinações possíveis
        for combinacao_precos in itertools.product(*precos):
            caixa_atual = 0
            viavel = True

            for i, sku in enumerate(skus):
                preco_escolhido = combinacao_precos[i]
                row = subconjunto.loc[subconjunto["sku"] == sku].iloc[0]

                caixa_sku = self.precificador.calcular_caixa_produto(preco_escolhido, row)

                if caixa_sku == -float("inf"):
                    viavel = False
                    break

                caixa_atual += caixa_sku

            if viavel and caixa_atual > melhor_caixa:
                melhor_caixa = caixa_atual
                melhor_combinacao = combinacao_precos
        return melhor_combinacao, melhor_caixa

class AvaliadorMeta:
    """Comparando o resultado gerado com a meta de caixa."""
    def __init__(self, meta_caixa: float = 331680.98):
        self.meta_caixa = meta_caixa

    def avaliar(self, caixa_gerado: float) -> dict:
        diferenca = caixa_gerado - self.meta_caixa
        meta_atingida = caixa_gerado >= self.meta_caixa
        return {
            "caixa_gerado": caixa_gerado,
            "meta_caixa": self.meta_caixa,
            "meta_atingida": meta_atingida,
            "diferenca": diferenca,
            "status": (
                "META ATINGIDA (SUPERÁVIT)"
                if meta_atingida
                else "META NÃO ATINGIDA (DÉFICIT)"
            ),
        }
            
    





   


    


    


