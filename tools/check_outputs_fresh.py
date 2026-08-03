#!/usr/bin/env python3
"""Verifica que las salidas comprometidas de un notebook siguen siendo las que
produce el código hoy.

Los notebooks se comprometen CON sus salidas, así que quien lee el repo en
GitHub ve números que nadie vuelve a comprobar. Si el código cambia y no se
reejecuta, esas cifras quedan rancias y el lector se lleva un resultado falso.

Compara solo el texto (stdout); las imágenes se ignoran a propósito, varían
byte a byte sin que el resultado cambie.

La comparación NO es de texto exacto. Los números se comparan con tolerancia
y el resto del texto exactamente. El motivo es concreto: el entorno de
desarrollo (conda, BLAS de MKL) y el del CI (ruedas de pip, OpenBLAS) difieren
en el último puñado de dígitos — p. ej. 0.175056007708 contra 0.175056053535.
Eso es ruido de punto flotante, no una salida rancia, y con comparación exacta
el check fallaría siempre en CI.

La separación es holgada: el ruido entre entornos ronda 1e-7 relativo,
mientras que una salida genuinamente rancia cambia mucho más (el caso que
motivó esto movía un parámetro calibrado un 2.7%). La tolerancia de 1e-5 deja
dos órdenes de margen sobre el ruido y sigue cazando lo que importa.

Uso:  python tools/check_outputs_fresh.py <notebook.ipynb> [...]
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

import nbformat


NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
RTOL = 1e-5
ATOL = 1e-9   # por debajo de esto todo es "cero numerico": 2.8e-14 y 7.1e-15
              # son el mismo resultado, aunque difieran 4x en relativo.


def equivalent(a: str, b: str) -> bool:
    """True si dos salidas dicen lo mismo salvo ruido de punto flotante."""
    if a == b:
        return True
    # El texto sin los numeros debe coincidir exactamente: un cambio de
    # redaccion, una columna nueva o una linea de mas no es ruido.
    if NUM.sub("#", a) != NUM.sub("#", b):
        return False
    for x, y in zip(NUM.findall(a), NUM.findall(b)):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            if x != y:
                return False
            continue
        if abs(fx) < ATOL and abs(fy) < ATOL:
            continue
        if abs(fx - fy) > RTOL * max(abs(fx), abs(fy)):
            return False
    return True


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

    if len(fresh) == len(committed) and all(
        equivalent(a, b) for a, b in zip(committed, fresh)
    ):
        print(f"OK   {path}")
        return True

    print(f"RANCIO {path}: las salidas comprometidas no coinciden con una ejecución fresca")
    for i, (a, b) in enumerate(zip(committed, fresh)):
        if not equivalent(a, b):
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
