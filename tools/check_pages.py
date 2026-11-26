"""Comprueba que las 7 páginas de app.py se ejecutan sin errores contra una base de datos real.

Uso:  python tools/check_pages.py [ruta/a/radar.db] [--html carpeta]
No necesita Streamlit ni Plotly: usa sustitutos mínimos (tools/stubs) que ejecutan cada página de verdad (consultas,
cálculos, HTML) y registran cualquier excepción (AttributeError, KeyError...). Devuelve código 1 si alguna falla.
Para probar más combinaciones de filtros: STUB_OVERRIDES='{"m_sp":"basketball","pf_tier":"combinadas"}'."""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    db = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else ROOT / "data" / "radar.db"
    out_dir = Path(argv[argv.index("--html") + 1]) if "--html" in argv else None
    sys.path[:0] = [str(Path(__file__).resolve().parent / "stubs"), str(ROOT)]
    os.environ["RADAR_DB"] = str(db)
    import streamlit as st  # el sustituto

    st.OUT["overrides"].update(json.loads(os.getenv("STUB_OVERRIDES", "{}")))
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    pages = [p for p in st.OUT["html"] if p != "_"] + [p for p in st.OUT["errors"] if p not in st.OUT["html"]]
    css = "".join(t for _, t in st.OUT["html"].get("_", []) if "<style>" in t)
    for page in pages:
        err = st.OUT["errors"].get(page)
        n = len(st.OUT["html"].get(page, []))
        print(f"{'FALLO' if err else 'OK   '} {page:26s} {n:3d} bloques")
        if err:
            print(err)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            body = "\n".join(t for _, t in st.OUT["html"].get(page, []))
            name = re.sub(r"[^a-z0-9]+", "_", page.lower()).strip("_")
            (out_dir / f"{name}.html").write_text(f'<!doctype html><meta charset="utf-8">{css}'
                                                  f'<body style="background:#06080C">{body}</body>', encoding="utf-8")
    print(f"\n{len(pages) - len(st.OUT['errors'])} de {len(pages)} páginas sin errores · base de datos: {db}")
    return 1 if st.OUT["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
