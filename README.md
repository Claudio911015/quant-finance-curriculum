# Quant Finance Curriculum

Currículum extenso de finanzas cuantitativas como colección de notebooks de Jupyter: teoría con derivaciones + implementación en Python desde cero + validación contra QuantLib. Doble uso: material de estudio riguroso y portfolio público en GitHub.

## Cómo correr

```bash
conda env create -f environment.yml
conda activate qfcurriculum
jupyter lab
```

Este procedimiento también instala `qflib` en modo editable. Para actualizar un ambiente existente, ejecutar `conda env update -f environment.yml --prune`.

## Currículum

13 módulos, progresión lineal (cada módulo asume los anteriores).

### M0 — Fundamentos de probabilidad

| Notebook | Tema | Estado |
|---|---|---|
| [00.1-probability-spaces.ipynb](notebooks/00-probabilidad/00.1-probability-spaces.ipynb) | Espacios de probabilidad, variables aleatorias, distribuciones, momentos, convergencia (LLN/CLT con simulación) | ✅ |
| [00.2-conditional-expectation-martingales.ipynb](notebooks/00-probabilidad/00.2-conditional-expectation-martingales.ipynb) | Esperanza condicional y martingalas en tiempo discreto | ✅ |
| [00.3-random-number-generation.ipynb](notebooks/00-probabilidad/00.3-random-number-generation.ipynb) | Simulación: generación de aleatorios, método de transformada inversa, Box-Muller, correlación vía Cholesky | ✅ |

### M1 — Procesos estocásticos

| Notebook | Tema | Estado |
|---|---|---|
| [01.1-random-walks-markov-chains.ipynb](notebooks/01-procesos-estocasticos/01.1-random-walks-markov-chains.ipynb) | Caminatas aleatorias y cadenas de Markov | ✅ |
| [01.2-brownian-motion.ipynb](notebooks/01-procesos-estocasticos/01.2-brownian-motion.ipynb) | Movimiento browniano: construcción, propiedades, simulación | ✅ |
| [01.3-ito-integral-lemma.ipynb](notebooks/01-procesos-estocasticos/01.3-ito-integral-lemma.ipynb) | Integral de Itô y lema de Itô | ✅ |
| [01.4-sdes-simulation.ipynb](notebooks/01-procesos-estocasticos/01.4-sdes-simulation.ipynb) | SDEs: GBM, OU, CIR — soluciones exactas, Euler-Maruyama, Milstein, convergencia fuerte/débil | ✅ |
| [01.5-girsanov-numeraire.ipynb](notebooks/01-procesos-estocasticos/01.5-girsanov-numeraire.ipynb) | Girsanov y cambio de numerario | ✅ |

### M2 — Derivados en tiempo discreto

| Notebook | Tema | Estado |
|---|---|---|
| [02.1-one-period-model.ipynb](notebooks/02-tiempo-discreto/02.1-one-period-model.ipynb) | Modelo de un periodo: no-arbitraje, medida neutral al riesgo, completitud | ✅ |
| [02.2-crr-binomial-tree.ipynb](notebooks/02-tiempo-discreto/02.2-crr-binomial-tree.ipynb) | Binomial multi-periodo (CRR): europeas y americanas | ✅ |
| [02.3-ftap.ipynb](notebooks/02-tiempo-discreto/02.3-ftap.ipynb) | Teoremas fundamentales del asset pricing | ✅ |
| [02.4-binomial-to-black-scholes.ipynb](notebooks/02-tiempo-discreto/02.4-binomial-to-black-scholes.ipynb) | Convergencia binomial → Black-Scholes | ✅ |

### M3 — Derivados en tiempo continuo

| Notebook | Tema | Estado |
|---|---|---|
| [03.1-black-scholes.ipynb](notebooks/03-tiempo-continuo/03.1-black-scholes.ipynb) | Black-Scholes: derivación por réplica (PDE) y por martingalas | ✅ |
| [03.2-greeks-delta-hedging.ipynb](notebooks/03-tiempo-continuo/03.2-greeks-delta-hedging.ipynb) | Griegas y delta hedging: P&L simulado de un hedge discreto | ✅ |
| [03.3-exotic-options.ipynb](notebooks/03-tiempo-continuo/03.3-exotic-options.ipynb) | Exóticas: digitales, barreras, asiáticas, lookback | ✅ |
| [03.4-american-options.ipynb](notebooks/03-tiempo-continuo/03.4-american-options.ipynb) | Americanas: free boundary, aproximaciones analíticas | ✅ |
| [03.5-forward-measure-numeraires.ipynb](notebooks/03-tiempo-continuo/03.5-forward-measure-numeraires.ipynb) | Forward measure y pricing bajo distintos numerarios | ✅ |

### M4 — Métodos numéricos

| Notebook | Tema | Estado |
|---|---|---|
| [04.1-monte-carlo-fundamentals.ipynb](notebooks/04-metodos-numericos/04.1-monte-carlo-fundamentals.ipynb) | Monte Carlo: fundamentos — error estándar, intervalos de confianza, sesgo de discretización vs error estadístico | ✅ |
| [04.2-variance-reduction-i.ipynb](notebooks/04-metodos-numericos/04.2-variance-reduction-i.ipynb) | Reducción de varianza I: variables antitéticas, control variates, muestreo estratificado, Latin Hypercube | ✅ |
| [04.3-variance-reduction-ii.ipynb](notebooks/04-metodos-numericos/04.3-variance-reduction-ii.ipynb) | Reducción de varianza II: muestreo de importancia (deep OTM), QMC con secuencias de Sobol | ✅ |
| [04.4-longstaff-schwartz.ipynb](notebooks/04-metodos-numericos/04.4-longstaff-schwartz.ipynb) | Longstaff-Schwartz para americanas/bermudas | ✅ |
| [04.5-finite-differences-i.ipynb](notebooks/04-metodos-numericos/04.5-finite-differences-i.ipynb) | Diferencias finitas I: esquemas forward (explícito), backward (implícito), θ-scheme; estabilidad y convergencia (von Neumann, CFL) | ✅ |
| [04.6-finite-differences-ii.ipynb](notebooks/04-metodos-numericos/04.6-finite-differences-ii.ipynb) | Diferencias finitas II: Crank-Nicolson, condiciones de frontera, grids no uniformes, barreras y americanas por PSOR | ✅ |

### M5 — Curvas de tasas

| Notebook | Tema | Estado |
|---|---|---|
| [05.1-instruments-conventions.ipynb](notebooks/05-curvas/05.1-instruments-conventions.ipynb) | Instrumentos y convenciones: depósitos, FRAs, futuros, swaps, OIS, daycounts, calendarios | ✅ |
| [05.2-single-curve-bootstrapping.ipynb](notebooks/05-curvas/05.2-single-curve-bootstrapping.ipynb) | Bootstrapping de curva única | ✅ |
| [05.3-multi-curve-ois-discounting.ipynb](notebooks/05-curvas/05.3-multi-curve-ois-discounting.ipynb) | Multi-curva: OIS discounting, basis, curva de proyección vs descuento | ✅ |
| [05.4-interpolation-methods.ipynb](notebooks/05-curvas/05.4-interpolation-methods.ipynb) | Interpolación (log-discount lineal, splines, monotone convex) y su impacto en forwards | ✅ |
| [05.5-sensitivities-dv01-krd.ipynb](notebooks/05-curvas/05.5-sensitivities-dv01-krd.ipynb) | Sensibilidades: DV01, key-rate durations, jacobiano de calibración | ✅ |

### M6 — Modelos de tasa corta

| Notebook | Tema | Estado |
|---|---|---|
| [06.1-vasicek-hull-white.ipynb](notebooks/06-tasa-corta/06.1-vasicek-hull-white.ipynb) | Vasicek y Hull-White: precios de bono analíticos, fit exacto a la curva | ✅ |
| [06.2-hull-white-calibration-trinomial-tree.ipynb](notebooks/06-tasa-corta/06.2-hull-white-calibration-trinomial-tree.ipynb) | Hull-White: calibración a caps/swaptions, árbol trinomial | ✅ |
| [06.3-cir-cir-plus-plus.ipynb](notebooks/06-tasa-corta/06.3-cir-cir-plus-plus.ipynb) | CIR y CIR++ | ✅ |
| [06.4-rate-exposure-simulation.ipynb](notebooks/06-tasa-corta/06.4-rate-exposure-simulation.ipynb) | Simulación de exposiciones de tasas (puente a XVA) | ✅ |

### M7 — HJM y LMM

| Notebook | Tema | Estado |
|---|---|---|
| [07.1-hjm-framework-drift-condition.ipynb](notebooks/07-hjm-lmm/07.1-hjm-framework-drift-condition.ipynb) | Marco HJM y condición de drift | ✅ |
| [07.2-lmm-forward-dynamics.ipynb](notebooks/07-hjm-lmm/07.2-lmm-forward-dynamics.ipynb) | LMM: dinámica de forwards, drifts bajo spot/terminal measure | ✅ |
| [07.3-lmm-calibration-correlation.ipynb](notebooks/07-hjm-lmm/07.3-lmm-calibration-correlation.ipynb) | Calibración a caplets/swaptions, estructura de correlación | ✅ |
| [07.4-lmm-simulation-bermudan-swaption.ipynb](notebooks/07-hjm-lmm/07.4-lmm-simulation-bermudan-swaption.ipynb) | Simulación LMM y Bermudan swaption vía Longstaff-Schwartz | ✅ |

### M8 — Volatilidad

| Notebook | Tema | Estado |
|---|---|---|
| [08.1-implied-vol-surface-svi.ipynb](notebooks/08-volatilidad/08.1-implied-vol-surface-svi.ipynb) | Superficie implícita: arbitraje estático (butterfly/calendar), parametrización SVI | ✅ |
| [08.2-local-volatility-dupire.ipynb](notebooks/08-volatilidad/08.2-local-volatility-dupire.ipynb) | Volatilidad local: Dupire | ✅ |
| [08.3-heston-model.ipynb](notebooks/08-volatilidad/08.3-heston-model.ipynb) | Heston: pricing por función característica, calibración a superficie | ✅ |
| [08.4-sabr-model.ipynb](notebooks/08-volatilidad/08.4-sabr-model.ipynb) | SABR: aproximación de Hagan, calibración al smile de swaptions | ✅ |
| [08.5-rough-volatility.ipynb](notebooks/08-volatilidad/08.5-rough-volatility.ipynb) | Rough volatility (panorama) y varianza forward | ✅ |

### M9 — FX

| Notebook | Tema | Estado |
|---|---|---|
| [09.1-garman-kohlhagen-vanna-volga.ipynb](notebooks/09-fx/09.1-garman-kohlhagen-vanna-volga.ipynb) | Garman-Kohlhagen, paridades, convenciones de smile (RR/BF), vanna-volga | ✅ |
| [09.2-quantos-composites.ipynb](notebooks/09-fx/09.2-quantos-composites.ipynb) | Quantos y composites | — |
| [09.3-cross-currency-basis.ipynb](notebooks/09-fx/09.3-cross-currency-basis.ipynb) | Cross-currency basis y colateral multi-divisa | — |

### M10 — Crédito

| Notebook | Tema | Estado |
|---|---|---|
| [10.1-merton-intensity-models.ipynb](notebooks/10-credito/10.1-merton-intensity-models.ipynb) | Merton estructural y modelos de intensidad | — |
| [10.2-cds-pricing-bootstrap.ipynb](notebooks/10-credito/10.2-cds-pricing-bootstrap.ipynb) | CDS: pricing y bootstrap de curva de crédito | — |
| [10.3-copulas-default-correlation.ipynb](notebooks/10-credito/10.3-copulas-default-correlation.ipynb) | Cópulas y correlación de default | — |

### M11 — XVAs

| Notebook | Tema | Estado |
|---|---|---|
| [11.1-exposures-ee-epe-pfe.ipynb](notebooks/11-xva/11.1-exposures-ee-epe-pfe.ipynb) | Exposiciones: EE/EPE/PFE sobre un portafolio de swaps | — |
| [11.2-cva-dva-wrong-way-risk.ipynb](notebooks/11-xva/11.2-cva-dva-wrong-way-risk.ipynb) | CVA/DVA y wrong-way risk | — |
| [11.3-fva-colva-mva-kva.ipynb](notebooks/11-xva/11.3-fva-colva-mva-kva.ipynb) | FVA, ColVA, MVA/KVA (panorama) | — |
| [11.4-netting-collateral-csa.ipynb](notebooks/11-xva/11.4-netting-collateral-csa.ipynb) | Netting, colateral y CSA | — |

### M12 — Riesgo de mercado

| Notebook | Tema | Estado |
|---|---|---|
| [12.1-var-es.ipynb](notebooks/12-riesgo-mercado/12.1-var-es.ipynb) | VaR/ES: paramétrico, histórico, Monte Carlo | — |
| [12.2-backtesting-kupiec-christoffersen.ipynb](notebooks/12-riesgo-mercado/12.2-backtesting-kupiec-christoffersen.ipynb) | Backtesting: Kupiec, Christoffersen | — |
| [12.3-portfolio-sensitivities-stress-testing.ipynb](notebooks/12-riesgo-mercado/12.3-portfolio-sensitivities-stress-testing.ipynb) | Sensibilidades de portafolio, escenarios y stress testing | — |
