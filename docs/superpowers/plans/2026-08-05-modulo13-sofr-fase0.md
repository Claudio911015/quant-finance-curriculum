# Módulo 13 — SOFR, Fase 0 (esqueleto + convenciones institucionales): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) o superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Arrancar el Módulo 13 (SOFR) del currículum: una función de composición SOFR con las cuatro convenciones de devengo RFR (lookback, lockout, observation shift, rate cutoff) en `qflib.sofr`, un pipeline de datos reales del NY Fed con snapshot fechado, y los dos primeros notebooks (contexto institucional + convenciones con el entregable verificable).

**Architecture:** Cuatro tareas secuenciales. `qflib/sofr.py` (Tarea 1) es pura lógica de composición, testeada con datos sintéticos hand-computed — no depende de red ni de los datos reales. `tools/fetch_sofr_data.py` (Tarea 2) es independiente de `sofr.py`: descarga y parsea la API del NY Fed, con las funciones de parseo testeadas contra fixtures JSON (sin red en tests) y ejecutado una vez de verdad para producir el snapshot fechado que Tarea 4 consume. `13.1` (Tarea 3) es narrativa pura, sin dependencia de las otras piezas. `13.2` (Tarea 4) es donde todo se junta: usa `qflib.sofr.compound_sofr` sobre el snapshot real de Tarea 2 para el entregable.

**Tech Stack:** Python 3.12, entorno conda `qfcurriculum`; `qflib` (paquete propio, sin numpy en este módulo — fechas y composición son aritmética simple con `datetime.date`); `requests` para el cliente HTTP (nueva dependencia, se declara en `pyproject.toml`); pytest; nbformat/nbconvert para los notebooks.

**Spec:** `docs/superpowers/specs/2026-08-05-modulo13-sofr-fase0-design.md`

## Global Constraints

- Prosa en español; código, nombres, docstrings y comentarios en inglés — mismo estándar que M0–M12.
- Estructura de notebook: título+motivación → teoría (celdas MD, LaTeX inline) → setup → demos (pares MD+code) → sección **Validación** con `assert`/`np.testing.assert_*` (mínimo 1, exigido por `tools/lint_notebooks.py`, `MIN_ASSERTS=1`) → referencias. El lint también exige que la sección de Validación tenga ese encabezado literal (`## Validación` o similar, regex case-insensitive) y rechaza asserts tautológicos (algo multiplicado por 0 comparado contra una tolerancia).
- Ejecución de notebooks: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <nb>` — debe correr limpio, sin errores ni warnings en las salidas.
- Los datos de mercado son reales (excepción documentada en el spec al "sin APIs externas" del resto del currículum) — todo snapshot va en `data/`, con fecha en el nombre de archivo (`sofr_YYYY-MM-DD.csv`, `sofrai_YYYY-MM-DD.csv`), y se **comitea al repo** para que CI y el notebook sean reproducibles sin red.
- Regla qflib: `qflib/sofr.py` no importa nada de otros módulos del currículum (no hay dependencia con `curves.py` en esta fase — eso viene en Fase 1 cuando se bootstrapea la curva).
- READMEs: al completar cada notebook, marcar ✅ en `README.md` (raíz) y en `notebooks/13-sofr/README.md`.
- CI: `.github/workflows/ci.yml` tiene una matrix por módulo (`00-probabilidad` ... `12-riesgo-mercado`) que ejecuta los notebooks de ese módulo y corre `check_outputs_fresh.py`. Agregar `13-sofr` a esa matrix es parte de la Tarea 3 (primer notebook del módulo).
- Commits: `feat: <qué> (M13.N)` para notebooks, `feat: qflib.sofr <qué>` / `feat: tools/fetch_sofr_data <qué>` para código de librería, sin Co-Authored-By. Un commit por tarea (o por step "commit" explícito dentro de la tarea).
- `requests` ya está en `requirements-lock.txt` (dependencia transitiva de otra cosa) pero **no** en `pyproject.toml` — declararla ahí como dependencia directa es parte de la Tarea 2, porque `fetch_sofr_data.py` es el primer código del repo que la usa deliberadamente.

## File Structure

**Librería (crear):**
- `qflib/sofr.py` — `compound_sofr()` y helpers internos de resolución de convención.
- `tools/fetch_sofr_data.py` — cliente del NY Fed + CLI para producir el snapshot.

**Config (modificar):**
- `pyproject.toml` — añadir `requests` como dependencia directa.

**Notebooks (crear):**
- `notebooks/13-sofr/README.md`
- `notebooks/13-sofr/13.1-institutional-context.ipynb`
- `notebooks/13-sofr/13.2-accrual-conventions.ipynb`

**Datos (crear, generados por Tarea 2, comiteados):**
- `data/sofr_YYYY-MM-DD.csv`
- `data/sofrai_YYYY-MM-DD.csv`

**Tests (crear/modificar):**
- `tests/test_sofr.py` (crear en Tarea 1, extendido en Tarea 4)
- `tests/test_fetch_sofr_data.py` (crear en Tarea 2)

**Docs/CI (modificar):** `README.md` (raíz), `.github/workflows/ci.yml`.

---

### Task 1: `qflib/sofr.py` — composición SOFR con convenciones de devengo

**Files:**
- Create: `qflib/sofr.py`
- Test: `tests/test_sofr.py`

**Interfaces:**
- Consumes: nada (módulo de arranque, solo `datetime.date` y `bisect` de stdlib).
- Produces: `compound_sofr(dates, rates, period_start, period_end, lookback=0, lockout=0, observation_shift=False, rate_cutoff=0) -> float`. `dates: list[date]` ascendente, fixings de días hábiles; `rates: list[float]` alineado 1:1 con `dates` (decimal, ej. `0.0530`); `period_start`/`period_end: date`, periodo `[period_start, period_end)`. Devuelve la tasa compuesta anualizada Act/360. Usada por Tarea 4.

**Contexto del algoritmo** (para quien implemente): la composición base (`in-arrears`, sin convenciones) es
`R = ((Π_i (1 + r_i · n_i/360)) - 1) · 360/D`
donde `i` recorre los días hábiles del periodo, `r_i` es el fixing SOFR de ese día, `n_i` los días calendario que ese fixing cubre (1 entre días hábiles consecutivos, 3 si cubre un fin de semana), y `D = Σ n_i` = días calendario totales del periodo. Las cuatro convenciones modifican qué `r_i`/`n_i` se usan:
- **lookback** (X días hábiles): el peso `n_i` sigue viniendo del periodo original, pero la tasa se toma de X días hábiles antes en el array `dates`.
- **observation shift**: tanto la tasa como el peso vienen de la ventana completa desplazada X días hábiles hacia atrás — equivale a correr la composición base sobre `[period_start desplazado, period_end desplazado)`.
- **lockout** (X días hábiles): los últimos X días hábiles del periodo reusan la tasa observada en el día `len(periodo)-X`.
- **rate_cutoff**: mismo mecanismo que `lockout` (freeze de los últimos días), nombre distinto por convención de mercado — se implementan con la misma lógica interna.

- [ ] **Step 1: Escribir el test de composición base (sin convenciones)**

```python
# tests/test_sofr.py
from datetime import date

import pytest

from qflib.sofr import compound_sofr

# Dos semanas hábiles sintéticas: semana 1 al 5%, semana 2 al 6% flat.
# Viernes cubre 3 días calendario (sáb+dom+el propio), el resto 1.
DATES = [
    date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9),
    date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14), date(2026, 1, 15), date(2026, 1, 16),
    date(2026, 1, 19),
]
RATES = [0.05] * 5 + [0.06] * 5 + [0.06]


def test_composicion_base_sin_convenciones():
    # Periodo = primera semana completa [Jan 5, Jan 12), pesos [1,1,1,1,3], D=7.
    r = compound_sofr(DATES, RATES, date(2026, 1, 5), date(2026, 1, 12))
    assert r == pytest.approx(0.05001786017442151, abs=1e-12)
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/test_sofr.py::test_composicion_base_sin_convenciones -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'qflib.sofr'`

- [ ] **Step 3: Implementar `compound_sofr` (composición base + los cuatro parámetros de convención)**

```python
# qflib/sofr.py
"""Composición de SOFR sobre un periodo de devengo bajo las convenciones RFR
estándar (lookback, lockout, observation shift, rate cutoff).

Ver notebooks/13-sofr/13.2-accrual-conventions.ipynb para la derivación y el
entregable que compara el resultado contra el SOFR Average publicado por el
NY Fed.
"""
import bisect
from datetime import date


def _shift_business_days(dates: list[date], d: date, n: int) -> date:
    """Fecha `n` días hábiles antes (n<0) o después (n>0) de `d`, según `dates`."""
    i = bisect.bisect_left(dates, d)
    j = i + n
    if j < 0 or j >= len(dates):
        raise ValueError(
            "shift fuera del rango de `dates`; amplía la ventana de fixings"
        )
    return dates[j]


def _calendar_weights(dates: list[date], idx: list[int], window_end: date) -> list[int]:
    weights = []
    for k, i in enumerate(idx):
        nxt = dates[idx[k + 1]] if k + 1 < len(idx) else window_end
        weights.append((nxt - dates[i]).days)
    return weights


def compound_sofr(
    dates: list[date],
    rates: list[float],
    period_start: date,
    period_end: date,
    lookback: int = 0,
    lockout: int = 0,
    observation_shift: bool = False,
    rate_cutoff: int = 0,
) -> float:
    """Tasa SOFR compuesta anualizada (Act/360) sobre [period_start, period_end).

    dates: fixings de días hábiles, ascendente, con margen suficiente antes de
        period_start si lookback/observation_shift > 0.
    rates: tasas SOFR (decimal, ej. 0.0530), alineadas 1:1 con `dates`.
    lookback: días hábiles de rezago en la TASA (no en el peso). 0 = in-arrears puro.
    lockout: días hábiles al final del periodo que congelan la tasa en su valor
        `lockout` días hábiles antes del fin.
    observation_shift: si True, tasa Y peso vienen de la ventana desplazada
        `lookback` días hábiles hacia atrás, en vez de solo la tasa.
    rate_cutoff: mismo mecanismo que `lockout` (freeze de los últimos días
        hábiles), nombre distinto por convención de mercado.
    """
    if observation_shift:
        shifted_start = _shift_business_days(dates, period_start, -lookback)
        shifted_end = _shift_business_days(dates, period_end, -lookback)
        idx = [i for i, d in enumerate(dates) if shifted_start <= d < shifted_end]
        weights = _calendar_weights(dates, idx, shifted_end)
        rate_idx = idx
    else:
        idx = [i for i, d in enumerate(dates) if period_start <= d < period_end]
        weights = _calendar_weights(dates, idx, period_end)
        rate_idx = [i - lookback for i in idx]

    freeze = max(lockout, rate_cutoff)
    freeze_at = max(0, len(rate_idx) - freeze)
    accrual = 1.0
    for k, ri in enumerate(rate_idx):
        use_idx = rate_idx[freeze_at] if freeze and k >= freeze_at else ri
        accrual *= 1.0 + rates[use_idx] * weights[k] / 360.0

    total_days = sum(weights)
    return (accrual - 1.0) * 360.0 / total_days
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/test_sofr.py::test_composicion_base_sin_convenciones -v`
Expected: PASS

- [ ] **Step 5: Escribir los tests de las cuatro convenciones (lookback, observation shift, lockout, rate_cutoff)**

```python
def test_lookback_sin_shift():
    # Periodo = segunda semana [Jan 12, Jan 19), lookback=2 días hábiles:
    # cada tasa se toma 2 días hábiles antes (mezcla semana 1 al 5% con semana 2 al 6%),
    # pero los pesos [1,1,1,1,3] siguen siendo los del periodo original.
    r = compound_sofr(DATES, RATES, date(2026, 1, 12), date(2026, 1, 19), lookback=2)
    assert r == pytest.approx(0.057165758287423936, abs=1e-12)


def test_observation_shift():
    # Mismo periodo y lookback que el test anterior, pero con observation_shift=True:
    # la ventana completa (tasa y peso) se corre 2 días hábiles hacia atrás.
    r = compound_sofr(
        DATES, RATES, date(2026, 1, 12), date(2026, 1, 19),
        lookback=2, observation_shift=True,
    )
    assert r == pytest.approx(0.054307266012279296, abs=1e-12)


def test_lookback_sin_shift_difiere_de_observation_shift():
    # El punto pedagógico del entregable 2 del currículum: lookback puro y
    # observation shift dan resultados distintos sobre el mismo periodo,
    # porque uno preserva los pesos originales y el otro no.
    r_lookback = compound_sofr(DATES, RATES, date(2026, 1, 12), date(2026, 1, 19), lookback=2)
    r_shift = compound_sofr(
        DATES, RATES, date(2026, 1, 12), date(2026, 1, 19),
        lookback=2, observation_shift=True,
    )
    assert r_lookback != pytest.approx(r_shift, abs=1e-6)


def test_lockout():
    # Fixture aparte con tasa distinta cada día (sube 1bp/día) para que el
    # freeze sea observable: sin lockout, los últimos 2 días pesan su propia
    # tasa (más alta); con lockout=2 quedan congelados en la tasa de 2 días
    # hábiles antes del fin del periodo.
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
    rates = [0.0500, 0.0510, 0.0520, 0.0530, 0.0540]
    r_lockout = compound_sofr(dates, rates, date(2026, 1, 5), date(2026, 1, 12), lockout=2)
    r_base = compound_sofr(dates, rates, date(2026, 1, 5), date(2026, 1, 12))
    assert r_lockout == pytest.approx(0.052162172041690776, abs=1e-12)
    assert r_base == pytest.approx(0.052590988760978466, abs=1e-12)
    assert r_lockout < r_base  # el freeze evita las 2 tasas más altas del final


def test_rate_cutoff_usa_el_mismo_mecanismo_que_lockout():
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
    rates = [0.0500, 0.0510, 0.0520, 0.0530, 0.0540]
    r_cutoff = compound_sofr(dates, rates, date(2026, 1, 5), date(2026, 1, 12), rate_cutoff=2)
    r_lockout = compound_sofr(dates, rates, date(2026, 1, 5), date(2026, 1, 12), lockout=2)
    assert r_cutoff == pytest.approx(r_lockout, abs=1e-12)


def test_shift_fuera_de_rango_de_dates_lanza_error():
    with pytest.raises(ValueError, match="fuera del rango"):
        compound_sofr(DATES, RATES, date(2026, 1, 5), date(2026, 1, 12), lookback=10)
```

- [ ] **Step 6: Correr todos los tests de `test_sofr.py` y verificar que pasan**

Run: `pytest tests/test_sofr.py -v`
Expected: 7 PASS (composición base + 5 de convenciones + el de error de rango)

- [ ] **Step 7: Commit**

```bash
git add qflib/sofr.py tests/test_sofr.py
git commit -m "feat: qflib.sofr compound_sofr con lookback/lockout/observation-shift/rate-cutoff"
```

---

### Task 2: `tools/fetch_sofr_data.py` — cliente NY Fed y snapshot fechado

**Files:**
- Create: `tools/fetch_sofr_data.py`
- Test: `tests/test_fetch_sofr_data.py`
- Modify: `pyproject.toml`
- Create (al correr el script, no a mano): `data/sofr_YYYY-MM-DD.csv`, `data/sofrai_YYYY-MM-DD.csv`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces: los dos CSV en `data/`, consumidos por la Tarea 4. Formato `sofr_*.csv`: columnas `date,sofr` (fecha ISO, tasa decimal). Formato `sofrai_*.csv`: columnas `date,average_30d,average_90d,average_180d,index`.

**Endpoints verificados** (respuesta real confirmada antes de escribir este plan):
- `https://markets.newyorkfed.org/api/rates/secured/sofr/last/{n}.json` → `{"refRates": [{"effectiveDate": "2026-08-04", "type": "SOFR", "percentRate": 3.66, ...}, ...]}`
- `https://markets.newyorkfed.org/api/rates/secured/sofrai/last/{n}.json` → `{"refRates": [{"effectiveDate": "2026-08-05", "type": "SOFRAI", "average30day": 3.62213, "average90day": 3.62708, "average180day": 3.66293, "index": 1.25363504, ...}, ...]}`

- [ ] **Step 1: Declarar `requests` como dependencia directa**

En `pyproject.toml`, dentro de `dependencies = [...]`, agregar la línea:

```toml
    "requests>=2.32",
```

(Ya está fijada en `requirements-lock.txt` como transitiva; esto la vuelve explícita porque `fetch_sofr_data.py` es el primer código que la usa a propósito.)

- [ ] **Step 2: Escribir los tests de las funciones de parseo (sin red)**

```python
# tests/test_fetch_sofr_data.py
from datetime import date

from tools.fetch_sofr_data import parse_sofr_json, parse_sofrai_json

SOFR_FIXTURE = {
    "refRates": [
        {"effectiveDate": "2026-08-04", "type": "SOFR", "percentRate": 3.66,
         "percentPercentile1": 3.60, "percentPercentile25": 3.64,
         "percentPercentile75": 3.71, "percentPercentile99": 3.74,
         "volumeInBillions": 3036, "revisionIndicator": ""},
        {"effectiveDate": "2026-08-03", "type": "SOFR", "percentRate": 3.65,
         "percentPercentile1": 3.61, "percentPercentile25": 3.63,
         "percentPercentile75": 3.70, "percentPercentile99": 3.73,
         "volumeInBillions": 3055, "revisionIndicator": ""},
    ]
}

SOFRAI_FIXTURE = {
    "refRates": [
        {"effectiveDate": "2026-08-05", "type": "SOFRAI", "average30day": 3.62213,
         "average90day": 3.62708, "average180day": 3.66293, "index": 1.25363504,
         "revisionIndicator": ""},
        {"effectiveDate": "2026-08-04", "type": "SOFRAI", "average30day": 3.62146,
         "average90day": 3.62652, "average180day": 3.66288, "index": 1.25350760,
         "revisionIndicator": ""},
    ]
}


def test_parse_sofr_json_convierte_a_decimal_y_ordena_ascendente():
    rows = parse_sofr_json(SOFR_FIXTURE)
    assert rows == [
        (date(2026, 8, 3), 0.0365),
        (date(2026, 8, 4), 0.0366),
    ]


def test_parse_sofrai_json_convierte_a_decimal_y_ordena_ascendente():
    rows = parse_sofrai_json(SOFRAI_FIXTURE)
    assert rows[0]["date"] == date(2026, 8, 4)
    assert rows[0]["average_90d"] == 0.0362652
    assert rows[1]["date"] == date(2026, 8, 5)
    assert rows[1]["index"] == 1.25363504
```

- [ ] **Step 3: Correr los tests y verificar que fallan**

Run: `pytest tests/test_fetch_sofr_data.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tools.fetch_sofr_data'`

- [ ] **Step 4: Implementar `tools/fetch_sofr_data.py`**

```python
#!/usr/bin/env python3
"""Descarga la serie SOFR y SOFR Averages/Index del NY Fed y guarda un
snapshot fechado en data/. Ver notebooks/13-sofr/13.2-accrual-conventions.ipynb.

Uso:  python tools/fetch_sofr_data.py [--n-days 400] [--out-dir data]
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import requests

SOFR_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/{n}.json"
SOFRAI_URL = "https://markets.newyorkfed.org/api/rates/secured/sofrai/last/{n}.json"


def fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_sofr_json(payload: dict) -> list[tuple[date, float]]:
    """[(fecha, tasa_decimal)] ascendente por fecha."""
    rows = [
        (date.fromisoformat(r["effectiveDate"]), r["percentRate"] / 100.0)
        for r in payload["refRates"]
    ]
    return sorted(rows)


def parse_sofrai_json(payload: dict) -> list[dict]:
    """[{date, average_30d, average_90d, average_180d, index}] ascendente por fecha."""
    rows = [
        {
            "date": date.fromisoformat(r["effectiveDate"]),
            "average_30d": r["average30day"] / 100.0,
            "average_90d": r["average90day"] / 100.0,
            "average_180d": r["average180day"] / 100.0,
            "index": r["index"],
        }
        for r in payload["refRates"]
    ]
    return sorted(rows, key=lambda x: x["date"])


def write_sofr_csv(rows: list[tuple[date, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sofr"])
        for d, r in rows:
            writer.writerow([d.isoformat(), r])


def write_sofrai_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "average_30d", "average_90d", "average_180d", "index"])
        for row in rows:
            writer.writerow([
                row["date"].isoformat(), row["average_30d"],
                row["average_90d"], row["average_180d"], row["index"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=400,
                         help="días hábiles a descargar (default 400, ~19 meses)")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    today = date.today().isoformat()

    sofr_rows = parse_sofr_json(fetch_json(SOFR_URL.format(n=args.n_days)))
    sofr_path = args.out_dir / f"sofr_{today}.csv"
    write_sofr_csv(sofr_rows, sofr_path)

    sofrai_rows = parse_sofrai_json(fetch_json(SOFRAI_URL.format(n=args.n_days)))
    sofrai_path = args.out_dir / f"sofrai_{today}.csv"
    write_sofrai_csv(sofrai_rows, sofrai_path)

    print(f"Escrito: {sofr_path.name} ({len(sofr_rows)} filas), "
          f"{sofrai_path.name} ({len(sofrai_rows)} filas)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Correr los tests y verificar que pasan**

Run: `pytest tests/test_fetch_sofr_data.py -v`
Expected: 2 PASS

- [ ] **Step 6: Correr el script de verdad para producir el snapshot**

Run: `cd ~/Git/quant-finance-curriculum && python tools/fetch_sofr_data.py`
Expected: imprime `Escrito: sofr_<hoy>.csv (~400 filas), sofrai_<hoy>.csv (~400 filas)`; revisar a mano que `data/sofr_<hoy>.csv` y `data/sofrai_<hoy>.csv` existen y tienen pinta razonable (`head -5 data/sofr_*.csv`).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tools/fetch_sofr_data.py tests/test_fetch_sofr_data.py data/sofr_*.csv data/sofrai_*.csv
git commit -m "feat: tools/fetch_sofr_data cliente NY Fed + snapshot fechado"
```

---

### Task 3: Notebook 13.1 — `13.1-institutional-context.ipynb`

**Files:**
- Create: `notebooks/13-sofr/13.1-institutional-context.ipynb`
- Create: `notebooks/13-sofr/README.md`
- Modify: `README.md` (raíz)
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nada (notebook narrativo, sin código de `qflib`).
- Produces: nada para otras tareas — es contexto, no librería.

**Contenido** (prosa en español, celdas MD con LaTeX donde aplique, código mínimo):
1. **Construcción del índice SOFR**: repo tripartito, GCF, bilateral; publicación diaria del NY Fed a partir de ~$1-1.5 trillones/día de transacciones de repo colateralizadas con Treasuries.
2. **Cronología de la reforma**: tabla con (fecha, evento) — anuncio FCA (2017-07-27), formación ARRC, cesación LIBOR USD (tenors no representativos dic-2021, resto jun-2023).
3. **El discounting big bang (oct 2020)**: LCH y CME cambian el descuento de Fed Funds a SOFR el mismo fin de semana; mecánica de la compensación en efectivo (cash compensation) para neutralizar el cambio de valor, entrega de basis swaps compensatorios, impacto en swaptions vigentes (el numerario cambia, así que el valor de mercado de una swaption ATM cambia aunque el subyacente no se mueva).
4. **Fallbacks ISDA**: spread de 26.161 bp para 3M USD LIBOR (mediana histórica del basis LIBOR-SOFR compuesto a 5 años, congelada el día del anuncio de cesación); el ajuste de 25bp al strike en opciones sobre eurodólar.
5. **Convenciones de devengo** (enunciado conceptual — la mecánica completa y el código viven en 13.2): lookback, lockout, observation shift, rate cutoff, cada una cambia el payoff.
6. **CME Term SOFR**: metodología forward-looking derivada de 13 futuros SOFR 1M consecutivos y 5 futuros SOFR 3M consecutivos, publicada en tenors 1M/3M/6M/12M.
7. **Estacionalidad y spikes**: fin de trimestre, fin de año, la escasez de reservas de sep-2019 (el evento que en retrospectiva mostró por qué SOFR es más volátil día a día que Fed Funds).

**Validación:**

```python
# Constantes regulatorias citadas arriba, verificadas por consistencia interna
# (no es un cálculo de modelo — es una validación de que la cita numérica
# está bien transcrita en dos representaciones independientes).
FALLBACK_SPREAD_3M_LIBOR_BP = 26.161  # ARRC/ISDA, congelado 2021-03-05

assert FALLBACK_SPREAD_3M_LIBOR_BP / 100 / 100 == pytest.approx(0.0026161)

# La cronología de la reforma debe ser estrictamente ascendente en el tiempo.
timeline = [
    date(2017, 7, 27),   # anuncio FCA
    date(2018, 4, 3),    # primera publicación SOFR por el NY Fed
    date(2020, 10, 16),  # discounting big bang (CME)
    date(2021, 3, 5),    # ISDA congela el fallback spread
    date(2021, 12, 31),  # cesación LIBOR USD tenors no representativos
    date(2023, 6, 30),   # cesación LIBOR USD resto de tenors
]
assert timeline == sorted(timeline)
```

(Usa `import pytest` solo para `pytest.approx` en el notebook — patrón ya usado en otros notebooks del currículum para asserts con tolerancia; si el linter/ejecución se queja de la dependencia, usar `abs(x - y) < 1e-12` en su lugar.)

**Referencias:** ARRC (*SOFR: A Year in Review*, guías de convención), CME Group (*Term SOFR Reference Rates Benchmark Methodology*, Rulebook caps. 460/900/902), Klingler & Syrstad (2021) *Life After LIBOR*, JFE 141, Henrard (2019) *LIBOR Fallback and Quantitative Finance*, Risks 7(3), Henrard *Discounting Transition: Big Bang Impacts*, SSRN 3530464.

- [ ] **Step 1: Crear `notebooks/13-sofr/README.md`**

```markdown
# M13 — SOFR

Especialización sobre modelos SOFR de principio a fin. Fase 0: convenciones
institucionales. Fases posteriores (bootstrap de curva, convexidad de
futuros, forward/backward-looking rates, caps/floors, SABR RFR, cubo de
swaptions, modelos de término) tienen su propio spec/plan cuando arranquen.

| Notebook | Tema | Estado |
|---|---|---|
| [13.1-institutional-context.ipynb](13.1-institutional-context.ipynb) | Construcción del índice, reforma LIBOR→SOFR, big bang de descuento, fallbacks ISDA, CME Term SOFR, estacionalidad | ⬜ |
| [13.2-accrual-conventions.ipynb](13.2-accrual-conventions.ipynb) | Convenciones de devengo (lookback/lockout/shift/cutoff) y comparación contra el SOFR Average real del NY Fed | ⬜ |

**Prerequisitos:** ninguno del resto del currículum (módulo autocontenido) — sí asume comodidad con cálculo estocástico y medidas de martingala del resto del currículum, per el currículum de referencia.
```

- [ ] **Step 2: Escribir el notebook `13.1-institutional-context.ipynb`** (nbformat, secciones y validación de arriba)

- [ ] **Step 3: Agregar `13-sofr` a la matrix de `.github/workflows/ci.yml`**

En el job `notebooks`, dentro de `strategy.matrix.module`, agregar después de `12-riesgo-mercado`:

```yaml
          - 13-sofr
```

- [ ] **Step 4: Ejecutar el notebook in-place y verificar que corre limpio**

Run: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/13-sofr/13.1-institutional-context.ipynb`
Expected: termina sin error; abrir el `.ipynb` y confirmar que las celdas de código muestran el assert pasando (sin `AssertionError` en la salida).

- [ ] **Step 5: Correr el linter de notebooks**

Run: `python tools/lint_notebooks.py`
Expected: sale con código 0 (o solo reporta issues en notebooks preexistentes, no en `13.1`).

- [ ] **Step 6: Actualizar READMEs — marcar 13.1 como ✅**

En `notebooks/13-sofr/README.md`, cambiar el estado de `13.1` a `✅`. En `README.md` (raíz), agregar una sección `### M13 — SOFR` con la misma tabla (siguiendo el formato de las secciones M0–M12 existentes), con `13.1` en ✅ y `13.2` en ⬜.

- [ ] **Step 7: Commit**

```bash
git add notebooks/13-sofr/README.md notebooks/13-sofr/13.1-institutional-context.ipynb .github/workflows/ci.yml README.md
git commit -m "feat: notebook M13.1 (contexto institucional SOFR)"
```

---

### Task 4: Notebook 13.2 — `13.2-accrual-conventions.ipynb` (entregable)

**Files:**
- Create: `notebooks/13-sofr/13.2-accrual-conventions.ipynb`
- Modify: `tests/test_sofr.py`
- Modify: `notebooks/13-sofr/README.md`, `README.md` (raíz)

**Interfaces:**
- Consumes: `qflib.sofr.compound_sofr` (Tarea 1); `data/sofr_<fecha>.csv` y `data/sofrai_<fecha>.csv` (Tarea 2).
- Produces: nada para otras tareas — es el entregable final de esta fase.

**Contenido:**
1. Recordatorio breve (una celda MD) de las cuatro convenciones, ahora con la fórmula de `compound_sofr` explícita (la misma que documenta el docstring de la Tarea 1).
2. Cargar `data/sofr_<fecha>.csv` (fixings diarios) y `data/sofrai_<fecha>.csv` (averages) con `csv.DictReader` — sin pandas, consistente con que `qflib.sofr` no depende de numpy/pandas.
3. **Entregable 1:** elegir un `period_end` fijo dentro del rango cubierto por el snapshot (con al menos 100 días calendario de margen antes, para tener el periodo de 90 días completo) — ej. el último `date` presente en `sofrai_*.csv` menos 5 días hábiles, para evitar el borde exacto del snapshot. Calcular `period_start = period_end - timedelta(days=90)`, correr `compound_sofr(sofr_dates, sofr_rates, period_start, period_end)`, y comparar contra el `average_90d` publicado en `sofrai_*.csv` para esa `period_end`. Nota conceptual a documentar en el notebook: el SOFR Average de 90 días del NY Fed es una ventana rodante de exactamente 90 días calendario terminando en esa fecha — mismo cálculo que la composición base sin convenciones, así que deben coincidir dentro de tolerancia de redondeo de la fuente (los `percentRate`/`average90day` publicados vienen redondeados).
4. **Entregable 2:** mismo `period_start`/`period_end`, ahora con `lookback=5` (5 días hábiles), una vez con `observation_shift=False` y otra con `observation_shift=True`. Tabla o gráfica (`matplotlib`, siguiendo `qflib.plotting.apply_style`) mostrando la diferencia en bp entre las tres variantes (base, lookback puro, con shift).
5. **Cierre en prosa** (celda MD, no opcional — es el criterio "ya lo dominas" del currículum): explicar por qué un swap con lookback sin observation shift no es exactamente replicable con la curva de descuento — la curva de descuento vive en tiempo del periodo de pago, pero lookback-sin-shift usa tasas observadas fuera de ese periodo con pesos que no corresponden a esas tasas, así que no hay una única curva de forwards consistente con ese devengo (a diferencia de observation shift, que sí es exactamente "la curva evaluada en la ventana desplazada").

**Validación:**

```python
# Entregable 1: compuesto base vs SOFR Average 90d publicado.
assert abs(r_base_90d - published_average_90d) < 5e-5  # 0.5bp de tolerancia por redondeo de la fuente

# Entregable 2: lookback puro y observation shift deben diferir (el punto pedagógico).
assert abs(r_lookback - r_shift) > 1e-6
```

- [ ] **Step 1: Extender `tests/test_sofr.py` con el test de comparación contra datos reales**

```python
# tests/test_sofr.py (agregar al final)
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _latest_snapshot(prefix: str) -> Path:
    candidates = sorted(DATA_DIR.glob(f"{prefix}_*.csv"))
    assert candidates, f"no hay snapshot {prefix}_*.csv en data/ — correr tools/fetch_sofr_data.py"
    return candidates[-1]


def test_compuesto_base_90d_coincide_con_sofr_average_publicado():
    sofr_path = _latest_snapshot("sofr")
    sofrai_path = _latest_snapshot("sofrai")

    with sofr_path.open() as f:
        rows = list(csv.DictReader(f))
    sofr_dates = [date.fromisoformat(r["date"]) for r in rows]
    sofr_rates = [float(r["sofr"]) for r in rows]

    with sofrai_path.open() as f:
        sofrai_rows = {date.fromisoformat(r["date"]): r for r in csv.DictReader(f)}

    # period_end = 5 días hábiles antes del último dato del snapshot, para
    # tener margen y evitar el borde exacto de la serie descargada.
    period_end = sofr_dates[-6]
    period_start = period_end - timedelta(days=90)
    published = float(sofrai_rows[period_end]["average_90d"])

    r_base = compound_sofr(sofr_dates, sofr_rates, period_start, period_end)
    assert r_base == pytest.approx(published, abs=5e-5)
```

(Agregar `from datetime import date, timedelta` al import existente de `datetime` si hace falta.)

- [ ] **Step 2: Correr el test y verificar que pasa**

Run: `pytest tests/test_sofr.py::test_compuesto_base_90d_coincide_con_sofr_average_publicado -v`
Expected: PASS (si falla por tolerancia, ajustar `abs=5e-5` según lo que el snapshot real produzca — documentar el ajuste con un comentario si difiere del punto de partida).

- [ ] **Step 3: Escribir el notebook `13.2-accrual-conventions.ipynb`** (secciones, entregables 1 y 2, y el cierre en prosa de arriba)

- [ ] **Step 4: Ejecutar el notebook in-place y verificar que corre limpio**

Run: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/13-sofr/13.2-accrual-conventions.ipynb`
Expected: termina sin error; ambos asserts de la Validación pasan.

- [ ] **Step 5: Correr el linter de notebooks y la suite completa de tests**

Run: `python tools/lint_notebooks.py && pytest tests/ -v`
Expected: lint sale con código 0; todos los tests pasan, incluidos los 8 de `test_sofr.py` y los 2 de `test_fetch_sofr_data.py`.

- [ ] **Step 6: Actualizar READMEs — marcar 13.2 como ✅**

En `notebooks/13-sofr/README.md` y en la sección `### M13 — SOFR` de `README.md` (raíz), cambiar el estado de `13.2` a `✅`.

- [ ] **Step 7: Commit**

```bash
git add notebooks/13-sofr/13.2-accrual-conventions.ipynb tests/test_sofr.py notebooks/13-sofr/README.md README.md
git commit -m "feat: notebook M13.2 (convenciones de devengo, entregable Fase 0)"
```
