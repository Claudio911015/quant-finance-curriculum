# M13 — SOFR

Especialización sobre modelos SOFR de principio a fin. Fase 0: convenciones
institucionales. Fases posteriores (bootstrap de curva, convexidad de
futuros, forward/backward-looking rates, caps/floors, SABR RFR, cubo de
swaptions, modelos de término) tienen su propio spec/plan cuando arranquen.

| Notebook | Tema | Estado |
|---|---|---|
| [13.1-institutional-context.ipynb](13.1-institutional-context.ipynb) | Construcción del índice, reforma LIBOR→SOFR, big bang de descuento, fallbacks ISDA, CME Term SOFR, estacionalidad | ✅ |
| 13.2-accrual-conventions.ipynb | Convenciones de devengo (lookback/lockout/shift/cutoff) y comparación contra el SOFR Average real del NY Fed | ⬜ |

**Prerequisitos:** ninguno del resto del currículum (módulo autocontenido) — sí asume comodidad con cálculo estocástico y medidas de martingala del resto del currículum, per el currículum de referencia.
