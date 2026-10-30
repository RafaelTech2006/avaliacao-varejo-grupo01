import json
from pathlib import Path

import pytest

from comum.dados import DATA_DIR, sha256_arquivo, verificar_manifesto


def test_instancia_oficial_integra():
    assert verificar_manifesto(DATA_DIR) == []


def test_manifesto_detecta_alteracao(tmp_path):
    for f in DATA_DIR.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    assert verificar_manifesto(tmp_path) == []
    alvo = tmp_path / "q1_catalogo_G01.csv"
    alvo.write_bytes(alvo.read_bytes() + b"\nX,Y")
    assert any("q1_catalogo_G01.csv" in p for p in verificar_manifesto(tmp_path))


def test_seed_do_manifesto_confere_com_matriculas():
    m = json.loads((DATA_DIR / "manifesto_dados.json").read_text(encoding="utf-8"))
    assert m["seed_final"] == 1000 * 1 + m["seed_base"]
