# Fase 1 — Core teórico (M0–M3): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Completar los 16 notebooks restantes de M0–M3 (probabilidad → derivados en tiempo continuo), cada uno siguiendo la plantilla estructural del notebook 0.1.

**Architecture:** Un notebook = una tarea. Cada tarea sigue la "Plantilla de tarea por notebook" del plan de Fase 0: escribir con nbformat → ejecutar in-place → validar (asserts) → actualizar READMEs → commit. Promociones a `qflib` solo donde el plan lo indica (con tests pytest).

**Tech Stack:** conda env `qfcurriculum`; NumPy/SciPy/matplotlib; QuantLib-Python para validación; nbformat + nbconvert.

**Spec:** `docs/superpowers/specs/2026-07-29-quant-curriculum-design.md`

## Global Constraints

- Prosa en español; términos técnicos, código, nombres y comentarios en inglés. LaTeX en Markdown cells (`$$…$$` display). Rigor de posgrado: derivaciones completas salvo donde se indique "enunciado + referencia".
- Estructura de todo notebook (plantilla 0.1): título+motivación → teoría (2-4 celdas MD) → setup (`apply_style()`, `rng = np.random.default_rng(42)`) → demos (pares MD+code) → validación con `assert`s → referencias. Render de 0.1 como referencia de tono: `.superpowers/sdd/review-task4-notebook.txt`.
- Parámetros canónicos (usarlos salvo que el notebook indique otros): `S0=100, K=100, r=0.05` (continuo), `q=0.0`, `sigma=0.20, T=1.0`.
- Seed fijo `np.random.default_rng(42)`. MC: ≤200k paths por celda; runtime objetivo <60 s por celda.
- Ejecución: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 <nb>` — limpio, sin warnings en outputs.
- Validación vs QuantLib donde el notebook lo especifica, con tolerancias explícitas en `assert`s. Validaciones MC: asserts con 3 errores estándar, no igualdad exacta.
- matplotlib mathtext: usar `\geq`/`\leq` (no `\ge`/`\le`); todo plot con título, ejes etiquetados y legend cuando hay >1 serie.
- READMEs: al completar cada notebook, marcar ✅ en README principal; en el README del módulo, añadir columna Estado si no existe y marcar ✅.
- Commits: `feat: notebook M.N (<tema corto>)`, sin Co-Authored-By. Un notebook = un commit (promociones qflib pueden ir en commit propio `feat: qflib.<mod> …`).
- Regla qflib: el notebook desarrolla su tema desde cero; solo importa de qflib lo transversal ya derivado en módulos anteriores (plotting, market siempre OK).

---

### Task 1: Notebook 0.2 — `00.2-conditional-expectation-martingales.ipynb`

**Teoría:** σ-álgebra como información y filtraciones `(𝓕_n)`; `E[X|𝓖]` definición formal (medibilidad + propiedad de promedios) y propiedades con demostraciones cortas: tower property, taking out what is known, independencia ⇒ `E[X|𝓖]=E[X]`; martingala/submartingala/supermartingala; ejemplos analíticos (random walk, `M_n = ∏ Y_i` con `E[Y_i]=1`); stopping times y Optional Stopping Theorem (enunciado + condiciones, referencia Williams; discusión de por qué fallan sin condiciones — estrategia de doblar apuesta).
**Demos:** (1) verificación MC de tower property con X = suma de 3 dados, 𝓖 = primer dado; (2) random walk simétrico: `E[S_{n+1}|𝓕_n] = S_n` verificado empíricamente por regresión de incrementos; (3) martingala de apuestas dobladas: trayectorias, quiebra con capital finito (gráfica de wealth paths).
**Validación:** asserts numéricos (tower property dentro de 3 SE; media de incrementos condicionales ≈ 0).
**Referencias:** Williams caps. 9-10; Shreve I cap. 2.

- [ ] Escribir notebook (nbformat) según spec de arriba y plantilla 0.1
- [ ] Ejecutar in-place limpio; asserts pasan
- [ ] READMEs actualizados
- [ ] Commit `feat: notebook 0.2 (esperanza condicional y martingalas)`

### Task 2: Notebook 0.3 — `00.3-random-number-generation.ipynb`

**Teoría:** PRNGs (mención LCG histórico y sus fallas, PCG64 moderno); método de transformada inversa (derivación); Box-Muller y método polar (derivación de Box-Muller); aceptación-rechazo (derivación de la probabilidad de aceptación); normales correlacionadas vía factorización de Cholesky (derivación `L L^T = Σ`).
**Demos:** (1) LCG malo (RANDU) mostrando estructura en 3D/planos vs PCG64; (2) inverse transform para Exponential(λ=2): histograma vs pdf; (3) Box-Muller: scatter + marginales; (4) aceptación-rechazo para Beta(2,5) con envolvente uniforme: tasa de aceptación empírica vs teórica; (5) Cholesky 3D con Σ dada (corr 0.6/0.3/0.1): corr empírica vs objetivo (heatmap).
**Validación:** `kstest` exponencial y normal (p>0.05, asserts); `np.allclose(corr_empírica, Σ, atol=0.02)`.
**Referencias:** Glasserman cap. 2; Devroye.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 0.3 (generación de aleatorios)`

### Task 3: Notebook 1.1 — `01.1-random-walks-markov-chains.ipynb`

**Teoría:** SRW: momentos, principio de reflexión (derivación) y distribución del máximo; cadenas de Markov finitas: matriz de transición, Chapman-Kolmogorov, distribución estacionaria (existencia para cadenas irreducibles aperiódicas, enunciado), balance detallado; gambler's ruin: derivación analítica de probabilidad de ruina (caso justo y sesgado).
**Demos:** (1) SRW: distribución del máximo simulada vs reflexión; (2) cadena de 3 estados (matriz dada): convergencia de `π_0 P^n` a π (gráfica de distancia TV vs n) y comparación con eigenvector; (3) gambler's ruin (a=10, b=20, p∈{0.5, 0.48}): probabilidad simulada vs fórmula.
**Validación:** asserts: ruina simulada dentro de 3 SE de la fórmula; `π P = π` con atol 1e-10.
**Referencias:** Norris *Markov Chains*; Shreve I cap. 3.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 1.1 (caminatas aleatorias y cadenas de Markov)`

### Task 4: Notebook 1.2 — `01.2-brownian-motion.ipynb`

**Teoría:** definición de BM (incrementos independientes estacionarios gaussianos, continuidad); construcción como límite escalado de RW (Donsker, enunciado) y construcción por interpolación/puente (Lévy, esquema); propiedades: no diferenciabilidad (argumento heurístico con `(W_{t+h}-W_t)/h`), scaling, variación cuadrática `[W]_t = t` (derivación en L²); puente browniano.
**Demos:** (1) RW escalado → BM visual (n=10²,10³,10⁵); (2) variación cuadrática: sumas `Σ(ΔW)²` sobre particiones refinadas → t, mientras variación total diverge (gráfica log); (3) simulación exacta de paths (incrementos gaussianos) usando `plot_paths`; (4) puente browniano condicionado W_1=0.
**Validación:** asserts: media y var de W_T dentro de 3 SE (0 y T); QV media dentro de 3 SE de T.
**Referencias:** Shreve II cap. 3; Karatzas-Shreve cap. 2.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 1.2 (movimiento browniano)`

### Task 5: Notebook 1.3 — `01.3-ito-integral-lemma.ipynb`

**Teoría:** construcción del integral de Itô para procesos simples (sumas con punto izquierdo — por qué el punto importa: comparación Itô vs Stratonovich), isometría de Itô (derivación), martingalidad; lema de Itô (derivación heurística vía Taylor + `(dW)² = dt`), versión con drift; ejemplos trabajados: `d(W²) = 2W dW + dt`, `d(ln S)` para GBM, `d(e^{W - t/2})`.
**Demos:** (1) `∫₀ᵀ W dW` por sumas de Riemann-Itô vs `(W_T² − T)/2` (convergencia al refinar la partición); punto medio → Stratonovich `W_T²/2` (mostrar la diferencia); (2) isometría: `E[(∫ W dW)²]` MC vs `∫ E[W²] dt = T²/2`; (3) verificación del drift de Itô: `E[ln S_T]` MC vs `ln S0 + (μ − σ²/2)T`.
**Validación:** asserts MC dentro de 3 SE; convergencia de sumas con la partición más fina dentro de tolerancia.
**Referencias:** Shreve II cap. 4; Øksendal caps. 3-4.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 1.3 (integral y lema de Itô)`

### Task 6: Notebook 1.4 — `01.4-sdes-simulation.ipynb` (+ promoción `qflib/mc.py`)

**Teoría:** SDEs y soluciones fuertes; GBM: solución exacta vía Itô; OU: solución exacta (factor integrante), media/varianza/estacionaria; CIR: propiedades, condición de Feller (enunciado); discretización: Euler-Maruyama y Milstein (derivación del término de corrección); convergencia fuerte (orden 0.5 EM, 1.0 Milstein) y débil (orden 1.0) — definiciones.
**Demos:** (1) GBM: exacto vs EM en el mismo path (mismos shocks); (2) OU (κ=2, θ=0.04, σ=0.1): paths + media/var empíricas vs analíticas en T; (3) CIR (mismos params, x0=0.03): full truncation scheme, % de pasos truncados con y sin Feller; (4) órdenes de convergencia: log-log del error fuerte E|X_T^h − X_T| vs h para EM y Milstein en GBM (pendientes ≈ 0.5 y 1.0), y error débil |E g(X_T^h) − E g(X_T)|.
**Validación:** asserts: momentos OU dentro de 3 SE; pendientes de regresión en log-log dentro de ±0.2 del orden teórico.
**Promoción qflib:** crear `qflib/mc.py` con `gbm_paths(S0, mu, sigma, T, n_steps, n_paths, rng) -> np.ndarray (n_paths, n_steps+1)` (esquema exacto log-euler), `ou_paths(x0, kappa, theta, sigma, ...)` (exacto gaussiano), `cir_paths(x0, kappa, theta, sigma, ...)` (full truncation Euler) — TDD: tests en `tests/test_mc.py` (shapes, momentos OU analíticos con seed fijo dentro de tolerancia, positividad CIR truncado, GBM lognormal media `S0*exp(mu*T)` dentro de 3 SE). Commit separado `feat: qflib.mc (generadores de paths GBM/OU/CIR)`.
**Referencias:** Kloeden-Platen; Glasserman cap. 6.

- [ ] Notebook (deriva los esquemas desde cero) / ejecutar / READMEs / commit
- [ ] qflib/mc.py con TDD (tests primero) / `pytest tests/ -v` verde / commit separado

### Task 7: Notebook 1.5 — `01.5-girsanov-numeraire.ipynb`

**Teoría:** cambio de medida en espacio discreto (derivada de Radon-Nikodym, ejemplo dado sesgado); proceso de densidad; teorema de Girsanov (enunciado formal + esbozo vía exponencial estocástica, condición de Novikov mencionada); aplicación: eliminar el drift de GBM → medida neutral al riesgo; cambio de numerario: fórmula general del drift bajo nuevo numerario (derivación corta), preview de forward measure (M3.5/M7).
**Demos:** (1) discreto: `E_Q[X]` calculado como `E_P[Z X]` — coinciden exactamente; (2) GBM bajo P (μ=0.1) con pesos de RN `Z_T = exp(−λW_T − λ²T/2)`, λ=(μ−r)/σ: `E_P[Z_T · e^{−rT}(S_T−K)^+]` vs precio bajo Q simulando con drift r (deben coincidir dentro de 3 SE); (3) gráfica: histograma de S_T bajo P vs reponderado (≈ densidad bajo Q).
**Validación:** asserts de coincidencia (3 SE) y del caso discreto (exacto, atol 1e-12).
**Referencias:** Shreve II cap. 5; Björk caps. 11-12.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 1.5 (Girsanov y cambio de numerario)`

### Task 8: Notebook 2.1 — `02.1-one-period-model.ipynb`

**Teoría:** mercado de un periodo con dos estados: portafolio replicante (derivación completa), no-arbitraje ⇔ `d < e^{rT} < u` ⇔ existencia de `q ∈ (0,1)`, precio = esperanza descontada bajo Q (derivación), completitud; mercado trinomial: incompletitud, intervalo de precios libres de arbitraje (super/sub-replicación por LP pequeño resuelto a mano/`scipy.optimize.linprog`).
**Demos:** (1) replicación exacta de un call en modelo binomial (u=1.2, d=0.85, r=0.05, T=1): tabla payoff vs portafolio; (2) barrido de precio del call violando no-arbitraje → construcción explícita del arbitraje; (3) trinomial: intervalo `[precio_inf, precio_sup]` vía linprog, gráfica del intervalo vs strikes.
**Validación:** asserts: replicación exacta (atol 1e-12); precio Q dentro del intervalo trinomial.
**Referencias:** Shreve I cap. 1; Pliska.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 2.1 (modelo de un periodo)`

### Task 9: Notebook 2.2 — `02.2-crr-binomial-tree.ipynb`

**Teoría:** árbol CRR multi-periodo: parametrización `u = e^{σ√Δt}`, probabilidad neutral al riesgo, backward induction (derivación como esperanzas condicionales iteradas); americanas: `V = max(payoff, cont. value)`; delta en el árbol; teorema de valuación (independencia del camino para europeas → fórmula binomial cerrada).
**Demos:** implementar `crr_price(S0, K, r, sigma, T, n_steps, kind, american)` vectorizado; (1) call/put europeos (params canónicos, n=500); (2) put americano: early exercise boundary extraída del árbol y graficada; (3) delta hedging a lo largo de un path del árbol: P&L final ≈ 0.
**Validación:** vs QuantLib `BinomialVanillaEngine("crr", 500)`: europeo atol 1e-10 (misma parametrización), americano atol 1e-6; delta del árbol vs QuantLib delta atol 1e-3.
**Referencias:** Cox-Ross-Rubinstein (1979); Shreve I; Hull cap. 13.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 2.2 (árbol binomial CRR)`

### Task 10: Notebook 2.3 — `02.3-ftap.ipynb`

**Teoría:** definiciones formales en mercado finito multi-periodo (estrategia autofinanciada, arbitraje); 1er FTAP: no-arbitraje ⇔ existe medida martingala equivalente (demostración en el caso finito vía separación de hiperplanos, esbozo); 2º FTAP: completitud ⇔ unicidad de Q; conexión con lo visto en 2.1/2.2.
**Demos:** (1) búsqueda de medida martingala en un árbol binomial de 2 pasos resolviendo el sistema lineal — única; (2) mismo ejercicio en árbol trinomial — familia de soluciones parametrizada (graficar el conjunto de (q1,q2,q3) admisibles); (3) detección de arbitraje vía `linprog` en un mercado de 3 activos con precios inconsistentes: extraer la estrategia de arbitraje explícita.
**Validación:** asserts: medida binomial única coincide con la fórmula CRR (atol 1e-12); el LP detecta el arbitraje sembrado y no detecta ninguno en el mercado consistente.
**Referencias:** Harrison-Pliska (1981); Delbaen-Schachermayer (panorama); Shreve I cap. 2.

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 2.3 (teoremas fundamentales del asset pricing)`

### Task 11: Notebook 2.4 — `02.4-binomial-to-black-scholes.ipynb`

**Teoría:** convergencia CRR → BS: CLT aplicado al log-precio bajo Q (derivación de que los momentos matchean), tasa de convergencia O(1/n) y el fenómeno de oscilación par/impar (posición del strike en la malla); extrapolación de Richardson; BS como límite — enunciar la fórmula (derivación completa en 3.1).
**Demos:** (1) precio CRR vs n (n=10…2000): convergencia oscilante hacia BS con la banda del error; (2) par vs impar por separado; (3) Richardson con (n, 2n): aceleración de la convergencia (gráfica log-log del error absoluto vs n para crudo y extrapolado).
**Validación:** vs fórmula BS implementada inline (validada a su vez contra QuantLib, atol 1e-10): |CRR(2000) − BS| < 5e-3; |Richardson − BS| < |CRR − BS| para n grandes.
**Referencias:** Leisen-Reimer (1996); Diener-Diener (2004).

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 2.4 (convergencia binomial a Black-Scholes)`

### Task 12: Notebook 3.1 — `03.1-black-scholes.ipynb` (+ promoción `qflib/black.py`)

**Teoría:** derivación 1 — réplica/PDE: portafolio autofinanciado, eliminación del riesgo, PDE de BS, condiciones terminales; solución de la PDE vía reducción a la ecuación de calor (esquema de los cambios de variable); derivación 2 — martingala: `V_0 = e^{−rT} E_Q[(S_T−K)^+]`, cálculo completo de la esperanza lognormal → fórmula con `d1, d2`; put-call parity; interpretación probabilística de `N(d2)` y `N(d1)`.
**Demos:** (1) implementar `bs_price(S, K, r, q, sigma, T, kind)`; superficie precio vs (S,T); (2) verificación MC del precio (3 SE); (3) put-call parity numérica.
**Validación:** vs QuantLib `AnalyticEuropeanEngine`: atol 1e-10 sobre una malla de strikes/vencimientos (K∈{80…120}, T∈{0.25,1,5}).
**Promoción qflib:** `qflib/black.py` con `bs_price`, `bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_rho`, `implied_vol(price, S, K, r, q, T, kind)` (Brent sobre bs_price) — TDD en `tests/test_black.py` (valores de referencia QuantLib hardcodeados con atol 1e-8; implied_vol(bs_price(σ))=σ con atol 1e-8; paridad). Justificación (regla ≥3 notebooks): M3.2-3.3, M4, M8 completo y M12 la reutilizan. Commit separado `feat: qflib.black (Black-Scholes analítico + implied vol)`.
**Referencias:** Black-Scholes (1973); Shreve II cap. 5-6; Björk.

- [ ] Notebook (deriva todo desde cero) / ejecutar / READMEs / commit
- [ ] qflib/black.py con TDD / suite verde / commit separado

### Task 13: Notebook 3.2 — `03.2-greeks-delta-hedging.ipynb`

**Teoría:** griegas analíticas de BS (derivaciones: delta, gamma, vega, theta, rho); relación PDE ↔ griegas (theta + ½σ²S²gamma + rSdelta = rV); griegas por diferencias finitas (error de truncamiento vs redondeo, elección de h); P&L de delta hedging discreto: derivación del P&L ≈ ½Γ S²(σ_real² − σ_imp²)dt acumulado.
**Demos:** (1) griegas analíticas vs FD central (tabla de errores vs h — U-shape); (2) delta hedge simulado (rebalanceo diario/semanal/quincenal, 20k paths): histogramas de P&L, std vs frecuencia (≈ ∝ 1/√N); (3) hedging con vol implícita ≠ realizada (σ_imp=0.2, σ_real=0.25): P&L medio ≈ valor teórico del gamma trading.
**Validación:** asserts: FD vs analíticas (usa `qflib.black`) atol 1e-6 con h óptimo; identidad PDE atol 1e-10; P&L medio del caso σ_real>σ_imp positivo y dentro de 3 SE del teórico.
**Referencias:** Hull; Wilmott caps. sobre hedging; El Karoui et al. (1998).

- [ ] Escribir / ejecutar / READMEs / commit `feat: notebook 3.2 (griegas y delta hedging)`

### Task 14: Notebook 3.3 — `03.3-exotic-options.ipynb`

**Teoría:** digitales cash-or-nothing y asset-or-nothing (derivación desde N(d2)/N(d1)); barreras: simetría de reflexión y fórmula down-and-out call (Reiner-Rubinstein, derivación esquemática vía reflexión + Girsanov); asiáticas: geométrica (derivación completa — sigue lognormal) y aritmética (sin forma cerrada); lookback flotante (fórmula, derivación esquemática vía máximo del BM con drift); sesgo de monitoreo discreto vs continuo (corrección de Broadie-Glasserman-Kou, enunciada).
**Demos:** (1) digitales: precio y el delta explosivo cerca de K/T; (2) barrera DOC (B=90): analítico vs MC discreto — mostrar el sesgo de monitoreo y la corrección BGK; (3) asiática geométrica analítica vs MC; aritmética por MC (preview control variates M4); (4) lookback flotante analítico vs MC.
**Validación:** vs QuantLib: `CashOrNothingPayoff` + AnalyticEuropeanEngine, `AnalyticBarrierEngine`, `AnalyticContinuousGeometricAveragePriceAsianEngine`, `AnalyticContinuousFloatingLookbackEngine` — atol 1e-8 en analíticos; MC dentro de 3 SE del analítico correspondiente (con corrección BGK para barrera).
**Referencias:** Reiner-Rubinstein (1991); Kemna-Vorst (1990); Broadie-Glasserman-Kou (1997); Hull caps. 26.

- [x] Escribir / ejecutar / READMEs / commit `feat: notebook 3.3 (opciones exóticas)`

### Task 15: Notebook 3.4 — `03.4-american-options.ipynb`

**Teoría:** parada óptima y envolvente de Snell (discreto, conexión con 2.2); formulación de frontera libre de la PDE (condiciones de smooth pasting); put perpetuo: derivación analítica completa (solución de la ODE + frontera óptima); aproximación de Barone-Adesi-Whaley (esquema de la derivación); por qué el call americano sin dividendos = europeo (demostración).
**Demos:** (1) put perpetuo: fórmula vs binomial con T grande; (2) frontera de ejercicio S*(t) del binomial (n=2000) graficada; smooth pasting visual (V y payoff tangentes en S*); (3) BAW implementado vs binomial denso: error vs (K, T); (4) demostración numérica call americano = europeo (q=0).
**Validación:** binomial n=2000 vs QuantLib `BinomialVanillaEngine` atol 1e-4; BAW propio vs QuantLib `BaroneAdesiWhaleyApproximationEngine` atol 1e-6; perpetuo vs límite binomial rtol 1e-2.
**Referencias:** Barone-Adesi & Whaley (1987); Shreve I cap. 4; Wilmott.

- [x] Escribir / ejecutar / READMEs / commit `feat: notebook 3.4 (opciones americanas)`

### Task 16: Notebook 3.5 — `03.5-forward-measure-numeraires.ipynb`

**Teoría:** numerario general y el teorema del cambio de numerario (derivación del cociente de densidades); medida T-forward: bono `P(t,T)` como numerario, forwards como martingalas bajo Q^T; Black-76 para opciones sobre forwards (derivación); stock como numerario: derivación de que el asset-or-nothing digital = `S0 N(d1)` sin calcular integrales; preview: por qué esto estructura M6-M7 (annuity measure, terminal measure LMM).
**Demos:** (1) con tasas deterministas: precio bajo Q vs bajo Q^T — idénticos (sanity); (2) asset-or-nothing por cambio de numerario (MC bajo la medida del stock: simular con drift r+σ²) vs fórmula `S0 N(d1)` vs MC bajo Q — tres caminos, un precio; (3) Black-76: call sobre forward vs BS spot con carry — equivalencia numérica.
**Validación:** asserts: (2) los tres precios dentro de 3 SE / atol 1e-10 el analítico; (3) atol 1e-10; QuantLib `BlackCalculator` para Black-76 atol 1e-10.
**Referencias:** Geman-El Karoui-Rochet (1995); Shreve II cap. 9; Brigo-Mercurio cap. 2.

- [x] Escribir / ejecutar / READMEs / commit `feat: notebook 3.5 (forward measure y numerarios)`

---

Al completar las 16 tareas: review final de fase (whole-branch), push final, y continuar con el plan de Fase 2.
