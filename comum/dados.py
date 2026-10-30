"""Utilitários compartilhados: caminhos, leitura da instância e verificação de integridade (SHA-256)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "data"


def ler_json(caminho: Path) -> dict:
    with open(caminho, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8-sig")


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def verificar_manifesto(data_dir: Path = DATA_DIR) -> list[str]:
    """Compara o hash dos arquivos oficiais com manifesto_dados.json.

    Retorna a lista de arquivos divergentes/ausentes (vazia = instância íntegra).
    """
    manifesto = ler_json(Path(data_dir) / "manifesto_dados.json")
    problemas = []
    for nome, hash_esperado in manifesto["arquivos"].items():
        caminho = Path(data_dir) / nome
        if not caminho.exists():
            problemas.append(f"{nome}: ausente")
        elif sha256_arquivo(caminho) != hash_esperado:
            problemas.append(f"{nome}: hash diferente do manifesto")
    return problemas


def caminhos_instancia(questao: int, grupo: str, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    return (Path(data_dir) / f"q{questao}_catalogo_{grupo}.csv",
            Path(data_dir) / f"q{questao}_config_{grupo}.json")


def carregar_instancia_bruta(questao: int, grupo: str = "G01", data_dir: Path = DATA_DIR):
    csv, cfg = caminhos_instancia(questao, grupo, data_dir)
    return ler_csv(csv), ler_json(cfg)
