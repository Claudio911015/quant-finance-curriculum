# Módulo 13 — SOFR, Fase 0 (esqueleto + convenciones institucionales) — Design Spec

**Fecha:** 2026-08-05
**Repo:** `~/Git/quant-finance-curriculum`

## Propósito

El currículum base (M0–M12) está completo. Este spec arranca una especialización nueva y separada — **modelos SOFR de principio a fin**, ~8 módulos / ~6 meses según el currículum de referencia — como **Módulo 13** del mismo repo, siguiendo el patrón de fases ya usado (`fase-0-esqueleto`, `fase-1-core-teórico`, ...).

Alcance de este spec: **solo** el esqueleto del módulo 13 y el contenido del "Módulo 0" del currículum SOFR (convenciones institucionales). Bootstrap de curva, convexidad de futuros, forward/backward-looking rates, caps/floors, SABR RFR, cubo de swaptions y modelos de término completo quedan fuera — cada uno será su propia fase con su propio spec, una vez esta esté verificada.

## Desviación de diseño respecto al spec base

El spec original del currículum (`2026-07-29-quant-curriculum-design.md`) fija: *"Datos: Sintéticos, generados paramétricamente por `qflib/`... Sin APIs externas."* Este módulo rompe esa regla deliberadamente: el entregable exige reproducir el SOFR Average publicado exactamente, lo cual solo es verificable contra la serie real del NY Fed. Se documenta aquí como excepción explícita, acotada al módulo 13 — el resto del currículum sigue usando datos sintéticos.

## Decisiones de diseño

| Decisión | Elección |
|---|---|
| Ubicación | Nuevo módulo `13-sofr` dentro de `quant-finance-curriculum` (no repo aparte, no C++) |
| Datos | Reales, vía API pública del NY Fed (`markets.newyorkfed.org`), como snapshot fechado en `data/` — excepción documentada arriba |
| División en notebooks | Dos: `13.1-institutional-context.ipynb` (narrativa/historia, poco código) y `13.2-accrual-conventions.ipynb` (convenciones + entregable) |
| Estilo de código | Funciones puras sobre arrays de NumPy, igual que `qflib/curves.py` — no clases salvo que la fase 2 (curva) lo requiera |
| Validación | Tests hand-computed (2-3 días de fixings a mano) + comparación exacta contra el SOFR Average publicado en la serie real |

## Estructura de archivos

```
notebooks/13-sofr/
  README.md
  13.1-institutional-context.ipynb
  13.2-accrual-conventions.ipynb
qflib/
  sofr.py              # compound_sofr(), convenciones (lookback/lockout/shift/cutoff)
tools/
  fetch_sofr_data.py   # descarga y guarda snapshot fechado
data/
  sofr_YYYY-MM-DD.csv  # fixings diarios (endpoint /rates/secured/sofr/)
  sofrai_YYYY-MM-DD.csv # averages 30/90/180d + index (endpoint /rates/secured/sofrai/)
tests/
  test_sofr.py
```

### `13.1-institutional-context.ipynb`

Narrativa con fuentes citadas, sin entregable de código (o mínimo):
- Construcción del índice SOFR: repo tripartito, GCF, bilateral; publicación NY Fed
- Cronología LIBOR → SOFR: FCA 2017, ARRC, cesación dic 2021 / jun 2023
- Big bang de descuento (oct 2020): LCH/CME, compensación en efectivo, basis swaps
- Fallbacks ISDA: spread 26.161bp (3M USD LIBOR), ajuste de 25bp a opciones sobre eurodólar
- Metodología CME Term SOFR (forward-looking, derivada de futuros 1M/3M)
- Estacionalidad: fin de trimestre/año, escasez de reservas sep 2019

Fuentes: ARRC (*SOFR: A Year in Review*, guías de convención), CME (*Term SOFR Reference Rates Benchmark Methodology*, Rulebook caps. 460/900/902), Klingler & Syrstad (2021), Henrard (2019, SSRN 3530464).

### `13.2-accrual-conventions.ipynb`

- Definición y cálculo de: lookback, lockout, observation shift, rate cutoff — cada una como parámetro de `compound_sofr`, con un ejemplo numérico chico a mano antes del caso real
- **Entregable 1:** compuesto de un periodo de 3M con la serie real vs. SOFR Average (90d) publicado — deben coincidir a precisión numérica (`atol` acorde a redondeo de la fuente)
- **Entregable 2:** mismo periodo con lookback de 5 días hábiles, con y sin observation shift — tabla/gráfica de la diferencia en bp
- Cierre en prosa: por qué un swap con lookback sin shift no es exactamente replicable con la curva (esto es el criterio "ya lo dominas" del currículum — debe quedar explícito en el notebook, no solo implícito en el código)

## `qflib/sofr.py`

```python
def compound_sofr(rates, day_counts, lookback=0, lockout=0, observation_shift=False, rate_cutoff=0):
    """Compone SOFR diario sobre un periodo de devengo. Ver notebooks/13-sofr/13.2..."""
```

Una función central con las convenciones como parámetros (no subclases ni estrategias separadas) — YAGNI hasta que la fase de curva demuestre que hace falta más estructura. Docstring apunta al notebook, como en `curves.py`.

`tools/fetch_sofr_data.py`: script CLI simple que llama los dos endpoints (`sofr`, `sofrai`), valida el JSON, escribe los dos CSV fechados en `data/`. Sin caché de reintentos ni programación — se corre a mano cuando se necesita un snapshot nuevo.

## Tests (`tests/test_sofr.py`)

- `compound_sofr` sin convenciones especiales, 2-3 fixings a mano (`pytest.approx`)
- Lookback puro vs. lockout vs. observation shift, cada uno con caso a mano donde difieren
- Rate cutoff con último fixing repetido
- Comparación contra el snapshot real: el compuesto de 90 días calculado coincide con `average90day` de `sofrai` dentro de tolerancia numérica

## Fuera de alcance

Bootstrap de curva SOFR, futuros y convexidad, forward vs. backward-looking, caps/floors, SABR RFR, cubo de swaptions, modelos de término completo — currículum SOFR módulos 1–8, cada uno en su propia fase/spec posterior.

## Criterio de éxito de esta fase

- Los dos notebooks corren limpio y pasan CI (mismo patrón que M0–M12: asserts inline + pytest)
- El entregable reproduce el SOFR Average real dentro de tolerancia numérica
- El notebook explica en prosa, no solo en código, por qué lookback-sin-shift no es replicable con la curva
