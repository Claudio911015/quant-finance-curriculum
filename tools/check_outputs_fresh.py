#!/usr/bin/env python3
"""Verifica que las salidas comprometidas de un notebook siguen siendo las que
produce el código hoy.

Los notebooks se comprometen CON sus salidas, así que quien lee el repo en
GitHub ve números que nadie vuelve a comprobar. Si el código cambia y no se
reejecuta, esas cifras quedan rancias y el lector se lleva un resultado falso.

Compara solo los bloques de texto (stdout). Las imágenes de las gráficas se
ignoran a propósito: varían byte a byte entre ejecuciones sin que el resultado
cambie. Los notebooks fijan semilla (np.random.default_rng), así que el texto
sí es reproducible — verificado antes de escribir esto.

Uso:  python tools/check_outputs_fresh.py <notebook.ipynb> [...]
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import nbformat


def stream_text(nb) -> list[str]:
    """Texto de stdout por celda de codigo, concatenado.

    Se concatena a proposito: Jupyter puede partir una misma salida en varios
    bloques `stream` segun como caiga el buffering, y esa particion cambia
    entre ejecuciones sin que el resultado cambie. Comparar bloque a bloque
    produce falsos positivos; comparar el texto por celda, no.
    """
    out = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        text = "".join(
            "".join(o.get("text", "")) if isinstance(o.get("text"), list) else o.get("text", "")
            for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        if text:
            out.append(text)
    return out


def check(path: pathlib.Path) -> bool:
    committed = stream_text(nbformat.read(path, as_version=4))
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / path.name
        proc = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.kernel_name=python3",
                "--ExecutePreprocessor.timeout=600",
                "--output", str(out), str(path),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"FALLO ejecutando {path}")
            print(proc.stderr[-1500:])
            return False
        fresh = stream_text(nbformat.read(out, as_version=4))

    if fresh == committed:
        print(f"OK   {path}")
        return True

    print(f"RANCIO {path}: las salidas comprometidas no coinciden con una ejecución fresca")
    for i, (a, b) in enumerate(zip(committed, fresh)):
        if a != b:
            print(f"  bloque {i}:")
            print(f"    comprometido: {a.strip()[:200]!r}")
            print(f"    fresco:       {b.strip()[:200]!r}")
            break
    if len(committed) != len(fresh):
        print(f"  distinto número de bloques: {len(committed)} comprometidos vs {len(fresh)} frescos")
    print("  -> reejecuta el notebook y compromételo de nuevo")
    return False


def main() -> int:
    paths = [pathlib.Path(a) for a in sys.argv[1:]]
    if not paths:
        print("uso: check_outputs_fresh.py <notebook.ipynb> [...]", file=sys.stderr)
        return 2
    return 0 if all([check(p) for p in paths]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
