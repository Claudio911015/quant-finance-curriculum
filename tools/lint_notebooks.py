#!/usr/bin/env python3
"""Chequeos estáticos de calidad sobre los notebooks del currículum.

No ejecuta nada: lee el .ipynb comprometido. Corre en segundos y protege
contra regresiones de calidad que ya ocurrieron en este repo:

  - asserts tautológicos (p. ej. evaluar x*0.0 y verificar que sea <1e-9)
  - factores de ajuste inventados sin derivación ni cita
  - encabezados sueltos que saltan directo al código, sin prosa
  - notebooks sin sección de Validación o con muy pocos asserts
  - enlaces rotos en el README
  - incoherencia entre requirements-lock.txt y pyproject.toml

Uso:  python tools/lint_notebooks.py
Sale con código 1 si hay algún fallo.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_GLOB = "notebooks/*/*.ipynb"

# Piso deliberadamente bajo: exigir una cuota alta incentivaría rellenar con
# asserts de adorno, que es justo la patología (validaciones tautológicas) que
# este lint existe para evitar. Lo que se exige es que HAYA validación real y
# una sección que la documente; que sea buena lo juzga una revisión, no un
# contador.
MIN_ASSERTS = 1
BARE_HEADER_BODY = 20    # chars de prosa bajo un encabezado para no considerarlo "pelado"

# Encabezados que legítimamente no necesitan párrafo de transición.
HEADER_WHITELIST = re.compile(
    r"^#+\s*(0\.\s*Configuraci|Referencias|Validaci)", re.IGNORECASE
)

failures: list[str] = []


def fail(where: str, msg: str) -> None:
    failures.append(f"{where}: {msg}")


def cells(nb, kind):
    return [c for c in nb["cells"] if c["cell_type"] == kind]


def source(c) -> str:
    return "".join(c["source"])


def check_notebook(path: pathlib.Path) -> None:
    rel = path.relative_to(ROOT)
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(source(c) for c in cells(nb, "code"))
    md = "\n".join(source(c) for c in cells(nb, "markdown"))

    # --- integridad de lo comprometido -----------------------------------
    for c in cells(nb, "code"):
        if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
            fail(rel, "hay una celda con output de error comprometido")
            break
    if any(c.get("execution_count") is None for c in cells(nb, "code")):
        fail(rel, "hay celdas de código sin ejecutar (execution_count nulo)")

    # --- tautologías y números mágicos -----------------------------------
    # Evaluar algo multiplicado por cero y compararlo contra una tolerancia
    # no puede fallar bajo ningún bug: es aritmética, no validación.
    if re.search(r"\*\s*0\.0\s*\)?[^\n]{0,60}<\s*1e-", code):
        fail(rel, "assert tautológico: algo multiplicado por 0.0 comparado contra una tolerancia")
    for m in re.finditer(r"(\w*factor\w*)\s*=\s*(0\.\d+)", code):
        fail(rel, f"factor de ajuste sin derivación: {m.group(1)}={m.group(2)}")
    for m in re.finditer(r"def\s+(\w*proxy\w*)\s*\(", code):
        fail(rel, f"función '{m.group(1)}': un proxy sin derivar suele esconder un modelo inventado")

    # --- cobertura de validación -----------------------------------------
    # Cuenta ambos idiomas: `assert` pelado y np.testing.assert_* (que además
    # reporta el diff al fallar, así que es preferible, no peor).
    n_assert = (
        len(re.findall(r"^\s*assert\b", code, re.M))
        + len(re.findall(r"np\.testing\.assert_\w+", code))
    )
    if n_assert < MIN_ASSERTS:
        fail(rel, f"solo {n_assert} validaciones (mínimo {MIN_ASSERTS}) — "
                  "ni assert ni np.testing.assert_*")
    if not re.search(r"^#+\s*\d*\.?\s*Validaci", md, re.M | re.IGNORECASE):
        fail(rel, "falta la sección de Validación")
    if not re.search(r"^#+\s*Referencias", md, re.M | re.IGNORECASE):
        fail(rel, "falta la sección de Referencias")

    # --- prosa: ningún encabezado salta directo al código ----------------
    for c in cells(nb, "markdown"):
        s = source(c).strip()
        if not s.startswith("#"):
            continue
        head, _, body = s.partition("\n")
        if HEADER_WHITELIST.match(head):
            continue
        if len(body.strip()) < BARE_HEADER_BODY:
            fail(rel, f"encabezado sin prosa antes del código: {head[:60]!r}")


def check_readme() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    linked = set()
    for m in re.finditer(r"\]\((notebooks/[^)]+\.ipynb)\)", text):
        target = ROOT / m.group(1)
        linked.add(m.group(1))
        if not target.exists():
            fail("README.md", f"enlace roto: {m.group(1)}")
    on_disk = {str(p.relative_to(ROOT)) for p in ROOT.glob(NB_GLOB)}
    for missing in sorted(on_disk - linked):
        fail("README.md", f"notebook no listado: {missing}")


def check_deps() -> None:
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    m = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', pyproject)
    if not m:
        fail("pyproject.toml", "no declara requires-python")
        return
    declared = (int(m.group(1)), int(m.group(2)))

    # numpy 2.5+ exige Python 3.12+; es justo la incoherencia que se coló una vez.
    mn = re.search(r"^numpy==(\d+)\.(\d+)", lock, re.M)
    if mn:
        nv = (int(mn.group(1)), int(mn.group(2)))
        needed = (3, 12) if nv >= (2, 5) else (3, 10)
        if declared < needed:
            fail(
                "pyproject.toml",
                f"requires-python >={declared[0]}.{declared[1]} pero el lock fija "
                f"numpy {nv[0]}.{nv[1]}, que exige >={needed[0]}.{needed[1]}",
            )

    for pkg in ("numpy", "scipy", "matplotlib"):
        if not re.search(rf"^{pkg}==", lock, re.M):
            fail("requirements-lock.txt", f"no fija {pkg}")


def main() -> int:
    paths = sorted(ROOT.glob(NB_GLOB))
    if not paths:
        print("no se encontraron notebooks", file=sys.stderr)
        return 1
    for p in paths:
        check_notebook(p)
    check_readme()
    check_deps()

    if failures:
        print(f"lint: {len(failures)} problema(s) en {len(paths)} notebooks\n")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"lint: OK — {len(paths)} notebooks sin problemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
