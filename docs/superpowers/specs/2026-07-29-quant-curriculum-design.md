# Quant Finance Curriculum — Design Spec

**Fecha:** 2026-07-29
**Repo:** `~/Git/quant-finance-curriculum` (GitHub: público, nombre tentativo `quant-finance-curriculum`)

## Propósito

Currículum extenso de finanzas cuantitativas como colección de notebooks de Jupyter: teoría con derivaciones + implementación en Python desde cero + validación contra QuantLib. Doble uso: material de estudio riguroso para claudio y portfolio público en GitHub.

## Decisiones de diseño

| Decisión | Elección |
|---|---|
| Audiencia | Híbrido estudio + portfolio: riguroso pero presentable; pedagogía moderada |
| Implementación | Desde cero con NumPy/SciPy; validar contra QuantLib-Python donde aplique |
| Idioma | Prosa en español; terminología técnica estándar en inglés; código, docstrings y nombres en inglés |
| Datos | Sintéticos, generados paramétricamente por `qflib/` (curvas, superficies, trayectorias). Sin APIs externas |
| Estructura de código | Notebooks autocontenidos en su tema + paquete mínimo `qflib/` para utilidades transversales |
| Organización | Lineal progresiva (cada módulo asume los anteriores) |

## Currículum — 13 módulos, ~49 notebooks

### M0 — Fundamentos de probabilidad
- 0.1 Espacios de probabilidad, variables aleatorias, distribuciones, momentos, convergencia (LLN/CLT con simulación)
- 0.2 Esperanza condicional y martingalas en tiempo discreto
- 0.3 Simulación: generación de aleatorios, método de transformada inversa, Box-Muller, correlación vía Cholesky

### M1 — Procesos estocásticos
- 1.1 Caminatas aleatorias y cadenas de Markov
- 1.2 Movimiento browniano: construcción, propiedades, simulación
- 1.3 Integral de Itô y lema de Itô
- 1.4 SDEs: GBM, OU, CIR — soluciones exactas, Euler-Maruyama, Milstein, convergencia fuerte/débil
- 1.5 Girsanov y cambio de numerario

### M2 — Derivados en tiempo discreto
- 2.1 Modelo de un periodo: no-arbitraje, medida neutral al riesgo, completitud
- 2.2 Binomial multi-periodo (CRR): europeas y americanas
- 2.3 Teoremas fundamentales del asset pricing
- 2.4 Convergencia binomial → Black-Scholes

### M3 — Derivados en tiempo continuo
- 3.1 Black-Scholes: derivación por réplica (PDE) y por martingalas
- 3.2 Griegas y delta hedging: P&L simulado de un hedge discreto
- 3.3 Exóticas: digitales, barreras, asiáticas, lookback
- 3.4 Americanas: free boundary, aproximaciones analíticas
- 3.5 Forward measure y pricing bajo distintos numerarios

### M4 — Métodos numéricos
- 4.1 Monte Carlo: fundamentos — error estándar, intervalos de confianza, sesgo de discretización vs error estadístico
- 4.2 Reducción de varianza I: variables antitéticas, control variates, muestreo estratificado, Latin Hypercube
- 4.3 Reducción de varianza II: muestreo de importancia (deep OTM), QMC con secuencias de Sobol
- 4.4 Longstaff-Schwartz para americanas/bermudas
- 4.5 Diferencias finitas I: esquemas forward (explícito), backward (implícito), θ-scheme; estabilidad y convergencia (von Neumann, CFL)
- 4.6 Diferencias finitas II: Crank-Nicolson, condiciones de frontera, grids no uniformes, barreras y americanas por PSOR

### M5 — Curvas de tasas
- 5.1 Instrumentos y convenciones: depósitos, FRAs, futuros, swaps, OIS, daycounts, calendarios
- 5.2 Bootstrapping de curva única
- 5.3 Multi-curva: OIS discounting, basis, curva de proyección vs descuento
- 5.4 Interpolación (log-discount lineal, splines, monotone convex) y su impacto en forwards
- 5.5 Sensibilidades: DV01, key-rate durations, jacobiano de calibración

### M6 — Modelos de tasa corta
- 6.1 Vasicek y Hull-White: precios de bono analíticos, fit exacto a la curva
- 6.2 Hull-White: calibración a caps/swaptions, árbol trinomial
- 6.3 CIR y CIR++
- 6.4 Simulación de exposiciones de tasas (puente a XVA)

### M7 — HJM y LMM
- 7.1 Marco HJM y condición de drift
- 7.2 LMM: dinámica de forwards, drifts bajo spot/terminal measure
- 7.3 Calibración a caplets/swaptions, estructura de correlación
- 7.4 Simulación LMM y Bermudan swaption vía Longstaff-Schwartz

### M8 — Volatilidad
- 8.1 Superficie implícita: arbitraje estático (butterfly/calendar), parametrización SVI
- 8.2 Volatilidad local: Dupire
- 8.3 Heston: pricing por función característica, calibración a superficie
- 8.4 SABR: aproximación de Hagan, calibración al smile de swaptions
- 8.5 Rough volatility (panorama) y varianza forward

### M9 — FX
- 9.1 Garman-Kohlhagen, paridades, convenciones de smile (RR/BF), vanna-volga
- 9.2 Quantos y composites
- 9.3 Cross-currency basis y colateral multi-divisa

### M10 — Crédito
- 10.1 Merton estructural y modelos de intensidad
- 10.2 CDS: pricing y bootstrap de curva de crédito
- 10.3 Cópulas y correlación de default

### M11 — XVAs
- 11.1 Exposiciones: EE/EPE/PFE sobre un portafolio de swaps
- 11.2 CVA/DVA y wrong-way risk
- 11.3 FVA, ColVA, MVA/KVA (panorama)
- 11.4 Netting, colateral y CSA

### M12 — Riesgo de mercado
- 12.1 VaR/ES: paramétrico, histórico, Monte Carlo
- 12.2 Backtesting: Kupiec, Christoffersen
- 12.3 Sensibilidades de portafolio, escenarios y stress testing

## Formato estándar de notebook

Cada notebook sigue esta plantilla:

1. **Motivación** (breve): qué problema resuelve el tema y dónde se usa en la práctica.
2. **Teoría**: derivaciones completas en LaTeX (Markdown cells). Rigor de posgrado; se enuncia sin demostrar solo lo que excede el alcance (con referencia).
3. **Implementación**: Python desde cero (NumPy/SciPy). Código legible > código clever. Sin clases salvo que el tema lo pida (p.ej. árboles, grids).
4. **Ejemplos y gráficas**: matplotlib, casos numéricos concretos.
5. **Validación** (donde aplique): comparación numérica contra QuantLib-Python con tolerancias explícitas.
6. **Referencias**: libros/papers estándar del tema (Shreve, Glasserman, Brigo-Mercurio, Gatheral, Green, Andersen-Piterbarg…).

Convenciones:
- Prosa en español; términos técnicos en inglés (forward measure, smile, bootstrapping — sin traducir).
- Código, nombres de variables, docstrings y comentarios en inglés.
- Seeds fijos en toda simulación (reproducibilidad).
- Nombre de archivo: `MM.N-slug-en-ingles.ipynb` (p.ej. `04.5-finite-differences-i.ipynb`).

## Paquete `qflib/`

Utilidades transversales mínimas — solo lo que se repetiría en ≥3 notebooks. Lo pedagógico vive en los notebooks, no aquí.

Alcance inicial:
- `qflib/market.py` — generadores de mercado sintético: curvas de descuento paramétricas (Nelson-Siegel), superficies de vol sintéticas (SVI), cotizaciones de swaps/caps/swaptions coherentes con esas curvas.
- `qflib/curves.py` — clase `DiscountCurve` (df, forward, zero) con interpoladores, usada después de que M5 la construye desde cero.
- `qflib/plotting.py` — estilo matplotlib consistente del repo, helpers (superficies 3D, paths, convergencia).
- `qflib/mc.py` — helpers de simulación (paths GBM/OU/CIR correlacionados) disponibles a partir de M5+, después de que M1/M4 los derivan desde cero.

Regla de dependencia: un notebook solo importa de `qflib` lo que un módulo anterior ya construyó y explicó desde cero. Tests con pytest para `qflib/` (`tests/`).

## Estructura del repo

```
quant-finance-curriculum/
├── README.md                  # currículum completo con links a cada notebook, badges, cómo correr
├── environment.yml            # conda env: numpy scipy pandas matplotlib jupyter QuantLib pytest
├── qflib/
├── tests/
├── notebooks/
│   ├── 00-probabilidad/
│   ├── 01-procesos-estocasticos/
│   ├── 02-tiempo-discreto/
│   ├── 03-tiempo-continuo/
│   ├── 04-metodos-numericos/
│   ├── 05-curvas/
│   ├── 06-tasa-corta/
│   ├── 07-hjm-lmm/
│   ├── 08-volatilidad/
│   ├── 09-fx/
│   ├── 10-credito/
│   ├── 11-xva/
│   └── 12-riesgo-mercado/
└── docs/superpowers/specs/
```

- Cada carpeta de módulo lleva un `README.md` corto con el índice del módulo y prerequisitos.
- Notebooks se commitean **con outputs ejecutados** (gráficas visibles en GitHub) — es material de lectura, no solo código.

## Entorno y tooling

- Conda env nuevo `qfcurriculum` (environment.yml versionado). Python 3.12+.
- Dependencias: numpy, scipy, pandas, matplotlib, jupyter, QuantLib (pip), pytest.
- Sin CI al inicio (YAGNI); si el repo crece, `nbconvert --execute` como smoke test es el candidato.
- GitHub: repo público `Claudio911015/quant-finance-curriculum`.

## Roadmap de implementación

El repo se construye por fases; cada fase deja el repo publicable (currículum completo visible en README desde el día 1, notebooks marcados como pendientes/completos).

1. **Fase 0 — Esqueleto**: estructura de carpetas, environment.yml, README con currículum completo, `qflib/market.py` + `plotting.py` mínimos, primer notebook (0.1) como plantilla de referencia.
2. **Fase 1 — Core teórico**: M0–M3 (probabilidad → tiempo continuo).
3. **Fase 2 — Numérico**: M4.
4. **Fase 3 — Tasas**: M5–M7 (curvas, tasa corta, HJM/LMM).
5. **Fase 4 — Vol y FX**: M8–M9.
6. **Fase 5 — Crédito, XVA, riesgo**: M10–M12.

Dentro de cada fase, un notebook = una unidad de trabajo revisable. El orden de fases sigue las dependencias del currículum.

## Criterios de éxito

- Cada notebook se ejecuta limpio de inicio a fin (`Restart & Run All`) en el env `qfcurriculum`.
- Las validaciones contra QuantLib pasan con las tolerancias declaradas en el propio notebook.
- Un lector con base matemática de posgrado puede seguir cualquier notebook leyendo solo los módulos anteriores.
- `pytest tests/ -v` verde para `qflib/`.

## Fuera de alcance (por ahora)

- Datos reales de mercado y conectores a APIs.
- Machine learning para finanzas.
- Portar contenido a la librería C++ `Quant_Finance` (proyecto separado; este repo puede referenciarla).
- CI, binder/colab badges, jupytext.
