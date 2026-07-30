# Fase 2 — Métodos numéricos (M4): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development o superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Completar los 6 notebooks de M4 (Monte Carlo, reducción de varianza, Longstaff-Schwartz y diferencias finitas), dejando en `qflib` las dos piezas que M6–M11 van a reutilizar.

**Architecture:** Un notebook = una tarea, con los steps fijos de la "Plantilla de tarea por notebook" del plan de Fase 0 (`docs/superpowers/plans/2026-07-29-fase-0-esqueleto.md`): escribir con nbformat → ejecutar in-place → validar con asserts → actualizar READMEs → commit. Dos tareas añaden promoción a `qflib` con TDD y commit propio. M4 es el módulo donde el currículum deja de valuar cosas nuevas y aprende a valuar *bien*: todo se contrasta contra los precios exactos que M3 ya derivó (BS, digitales, barrera, asiática geométrica, put americano binomial), así que cada método numérico tiene una referencia dura contra la cual medir su error.

**Tech Stack:** conda env `qfcurriculum`; NumPy/SciPy (`scipy.stats.qmc` para Sobol/LHS, `scipy.linalg.solve_banded` para los sistemas tridiagonales); QuantLib-Python para validación; nbformat + nbconvert.

**Spec:** `docs/superpowers/specs/2026-07-29-quant-curriculum-design.md` (§M4 y "Formato estándar de notebook").

## Global Constraints

- Prosa en español; términos técnicos, código, nombres y comentarios en inglés. LaTeX en Markdown cells (`$$…$$` display). Rigor de posgrado: derivaciones completas salvo donde se indique "enunciado + referencia".
- Estructura de todo notebook (plantilla 0.1): título+motivación → teoría (2-4 celdas MD) → setup (`apply_style()`, `rng = np.random.default_rng(42)`) → demos (pares MD+code) → validación con `assert`s → referencias.
- Parámetros canónicos (usarlos salvo que el notebook indique otros): `S0=100, K=100, r=0.05` (continuo), `q=0.0`, `sigma=0.20, T=1.0`. Referencias exactas ya derivadas en M3 y reutilizables como verdad: call europeo `10.4505835722`, put europeo `5.5735260223`, asset-or-nothing `63.6830651176`, DOC con `B=90` `8.6654716582`, asiática geométrica continua `5.5468186338`, put americano binomial (n=2000, p exacta) `6.0899899526`.
- Seed fijo `np.random.default_rng(42)`. MC: ≤200k paths por celda salvo que la tarea diga otra cosa; runtime objetivo <60 s por celda, <5 min por notebook.
- Ejecución: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <nb>` — limpio, sin errores ni warnings en outputs.
- **Regla de validación (lección de la Fase 1, no negociable):** un estimador con discretización (monitoreo discreto, paso de tiempo, malla) **no converge a la fórmula continua**, así que compararlo contra ella mide sesgo de discretización y no error muestral. Cada assert debe comparar contra la referencia *que le corresponde*: el analítico discreto cuando existe, el continuo corregido cuando hay corrección conocida, o el continuo pero con una tolerancia derivada del orden del sesgo — nunca "3 SE" contra un continuo con sesgo de por medio. Donde se afirme un orden de convergencia, verificarlo con un assert sobre `error × n^alpha` aproximadamente constante entre los extremos de la malla (rtol explícito), no con una sola corrida.
- Las tolerancias numéricas de cada tarea son **puntos de partida a verificar**: si al ejecutar no se cumplen, primero diagnosticar si es sesgo real (y entonces corregir el diseño de la validación, como se hizo en 3.3) y sólo después ajustar la tolerancia, documentando el motivo en el notebook.
- matplotlib mathtext: usar `\geq`/`\leq` (no `\ge`/`\le`) **en strings de código**; en Markdown cells `\ge`/`\le` son válidos. Todo plot con título, ejes etiquetados y legend cuando hay >1 serie.
- READMEs: al completar cada notebook, marcar ✅ en README principal; en `notebooks/04-metodos-numericos/README.md` añadir columna Estado (hoy no la tiene) y marcar ✅.
- Commits: `feat: notebook M.N (<tema corto>)`, sin Co-Authored-By. Un notebook = un commit; promociones a qflib en commit propio `feat: qflib.<mod> …`.
- Regla qflib: el notebook desarrolla su tema desde cero; sólo importa de qflib lo transversal ya derivado en módulos anteriores (`plotting`, `market`, `black`, `mc` son OK — `mc.gbm_paths` se derivó en 1.4 y `black` en 3.1).

## File Structure

**Notebooks (crear):**
- `notebooks/04-metodos-numericos/04.1-monte-carlo-fundamentals.ipynb` — error estándar, IC, sesgo de discretización vs error estadístico
- `notebooks/04-metodos-numericos/04.2-variance-reduction-i.ipynb` — antitéticas, control variates, estratificado, LHS
- `notebooks/04-metodos-numericos/04.3-variance-reduction-ii.ipynb` — importance sampling, QMC/Sobol
- `notebooks/04-metodos-numericos/04.4-longstaff-schwartz.ipynb` — LSM para americanas/bermudas
- `notebooks/04-metodos-numericos/04.5-finite-differences-i.ipynb` — explícito/implícito/θ, estabilidad von Neumann
- `notebooks/04-metodos-numericos/04.6-finite-differences-ii.ipynb` — Crank-Nicolson, fronteras, grids no uniformes, barreras, PSOR

**qflib (modificar/crear):**
- `qflib/mc.py` (modificar) — añadir `mc_estimate` y `normal_sampler`: los usan M6.4 (exposiciones), M7.4 (LMM), M11 (XVA) y todo M12, muy por encima del umbral de ≥3 notebooks del spec.
- `qflib/lsm.py` (crear) — `longstaff_schwartz`: lo reusan M7.4 (Bermudan swaption) y M11.1 (exposiciones con ejercicio), además de 4.4.
- **Decisión explícita: el solver PDE NO se promueve en esta fase.** Sólo se ve un reuso claro (M8.2, Dupire). Si M8 lo vuelve a necesitar, se promueve entonces con su propio TDD; mientras tanto vive en 4.5/4.6.

**Tests (modificar/crear):**
- `tests/test_mc.py` (modificar) — casos de `mc_estimate` y `normal_sampler`
- `tests/test_lsm.py` (crear) — casos de `longstaff_schwartz`

**Docs (modificar):** `README.md`, `notebooks/04-metodos-numericos/README.md`, este plan.

---

### Task 1: Notebook 4.1 — `04.1-monte-carlo-fundamentals.ipynb`

**Files:**
- Create: `notebooks/04-metodos-numericos/04.1-monte-carlo-fundamentals.ipynb`
- Modify: `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.mc.gbm_paths`, `qflib.plotting.apply_style`.
- Produces: nada para otras tareas (el notebook define sus helpers localmente).

**Teoría:** estimador MC `\hat{V}_n = \frac1n\sum Y_i` insesgado; LLN da consistencia y **TCL da la barra de error**: `\sqrt{n}(\hat V_n - V) \to N(0,\sigma_Y^2)`, de donde `SE = s_n/\sqrt{n}` y el IC al 95% `\hat V_n \pm 1.96\,SE` — con la advertencia de que es un IC *asintótico* y que `s_n` es a su vez aleatorio. El costo: reducir el error a la mitad cuesta 4× el trabajo, tasa `O(n^{-1/2})` **independiente de la dimensión** (el contraste con cuadratura, que motiva todo M4). Descomposición del error total en **sesgo de discretización + error estadístico**: para un payoff path-dependent simulado con `m` pasos, `\mathbb{E}[\hat V_{n,m}] - V = \underbrace{b(m)}_{\text{sesgo}} + O_p(n^{-1/2})`, y sólo el segundo término se ve en las barras de error — el primero es invisible al MC y es exactamente el error que hizo fallar 3.3. Órdenes de convergencia de Euler-Maruyama: **débil** `O(\Delta t)` (error en esperanzas) y **fuerte** `O(\sqrt{\Delta t})` (error trayectoria a trayectoria); Milstein sube el fuerte a `O(\Delta t)` añadiendo el término `\tfrac12\sigma\sigma'(\Delta W^2-\Delta t)`, que para GBM es `\tfrac12\sigma^2 S(\Delta W^2-\Delta t)`. Enunciar (sin demostrar, referencia Glasserman §6.1) por qué el orden fuerte importa sólo para payoffs path-dependent y para multilevel MC.

**Demos:** (1) convergencia del error del call europeo vs `n \in \{10^3,\dots,10^6\}` en log-log, con la recta de pendiente `-1/2` superpuesta y las bandas de ±1.96 SE; (2) **cobertura empírica del IC**: 500 réplicas independientes con `n=10^4`, contar qué fracción de los IC al 95% contiene el precio BS exacto (debe salir ≈0.95) — el experimento que convierte el IC de fórmula en afirmación verificable; (3) sesgo débil de Euler: precio del call con `m \in \{1,2,4,\dots,64\}` pasos vs el simulador exacto de GBM, con `n=10^6` y **números aleatorios comunes** entre las `m` para que la curva de sesgo no sea ruido; graficar `error × m` (debe ser constante); (4) error fuerte de Euler vs Milstein: `\mathbb{E}|S_T^{(m)} - S_T^{\text{exacto}}|` sobre las mismas normales, en log-log, pendientes `-1/2` y `-1`.

**Validación:** call MC (simulador exacto, `n=10^6`) vs `bs_price` dentro de 3 SE; pendiente del ajuste log-log de SE vs `n` en `[-0.52, -0.48]`; cobertura del IC en `[0.93, 0.97]`; sesgo débil de Euler: `|error × m|` constante entre `m=4` y `m=64` con rtol 0.15; error fuerte: pendiente de Euler en `[-0.60,-0.40]` y de Milstein en `[-1.15,-0.85]`.

**Referencias:** Glasserman (2004) caps. 1, 6; Kloeden & Platen (1992) caps. 9-10; Higham (2001) *An Algorithmic Introduction to Numerical Simulation of SDEs*.

- [x] Escribir notebook (nbformat) según la plantilla y las secciones de arriba
- [x] Ejecutar in-place limpio; los 6 asserts pasan
- [x] READMEs actualizados (añadir columna Estado al README del módulo)
- [x] Commit `feat: notebook 4.1 (fundamentos de Monte Carlo)`

---

### Task 2: Notebook 4.2 — `04.2-variance-reduction-i.ipynb`

**Files:**
- Create: `notebooks/04-metodos-numericos/04.2-variance-reduction-i.ipynb`
- Modify: `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.mc.gbm_paths`, `qflib.plotting.apply_style`.
- Produces: nada (helpers locales).

**Teoría:** el marco común — toda técnica de reducción de varianza es *el mismo estimador con menos varianza por unidad de trabajo*, así que la métrica honesta es la **eficiencia** `\text{Var}\times\text{tiempo}`, no la varianza sola. (1) **Antitéticas:** `\hat V = \frac12(f(Z)+f(-Z))`; `\text{Var} = \frac12(\text{Var}f + \text{Cov}(f(Z),f(-Z)))`, así que ayuda si y sólo si la covarianza es negativa; teorema: si `f` es monótona, la covarianza es `\leq 0` (demostración vía la desigualdad de Chebyshev para asociación / argumento de reordenamiento) — y contraejemplo explícito con un payoff no monótono (straddle) donde **empeora**. (2) **Control variates:** con `X` de media conocida `\mu_X`, `\hat V_c = \bar Y - c(\bar X - \mu_X)`; minimizando en `c` sale `c^* = \text{Cov}(X,Y)/\text{Var}(X)` y `\text{Var}(\hat V_c) = \text{Var}(\bar Y)(1-\rho^2)`: **la reducción depende sólo de la correlación**. Advertir del sesgo de estimar `c^*` con la misma muestra (`O(1/n)`, despreciable pero real; mencionar el remedio de muestra piloto). (3) **Estratificado:** partir el soporte de la normal terminal en `k` estratos equiprobables vía la inversa de la CDF, muestrear `n/k` en cada uno; la varianza pasa a ser sólo la varianza *dentro* de estratos — la varianza *entre* estratos se elimina exactamente. (4) **LHS:** estratificar cada dimensión marginalmente con una permutación aleatoria; qué garantiza (marginales perfectas) y qué no (nada sobre interacciones).

**Demos:** (1) antitéticas en el call europeo: varianza y factor de eficiencia; repetir con un straddle para exhibir el caso donde no sirve; (2) **control variate del preview de 3.3**: asiática aritmética con la geométrica como control, usando la fórmula cerrada *discreta* de 3.3 §6.1b como `\mu_X` (correlación ≈0.9996 medida en 3.3 → esperar una reducción de varianza de dos órdenes); graficar `\text{Var}` vs `c` con el mínimo en `c^*`; (3) estratificado sobre `Z_T` con `k \in \{1,10,100\}` estratos en el call, y el mismo experimento en un digital cash-or-nothing (donde la ganancia es mucho mayor por el payoff escalonado); (4) LHS en una asiática de 4 fechas de monitoreo vs MC crudo.

**Validación:** cada estimador dentro de 3 SE de su referencia exacta (call/digital: fórmulas de M3; asiática aritmética: no tiene cerrada, así que se compara el estimador con control contra el MC crudo, `|diff| \leq 3\sqrt{SE_1^2+SE_2^2}`); `Var_antithetic < Var_plain` en el call y `Var_antithetic > Var_plain` en el straddle (assert de ambos signos); `Var_control/Var_plain < 0.05` en la asiática; `Var_stratified(k=100) < Var_plain` en el digital con factor >5; `c^*` estimado vs el óptimo teórico `\rho\sigma_Y/\sigma_X` con rtol 0.05.

**Referencias:** Glasserman (2004) cap. 4; Boyle, Broadie & Glasserman (1997) *Monte Carlo Methods for Security Pricing*; McKay, Beckman & Conover (1979) (LHS).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 4.2 (reducción de varianza I)`

---

### Task 3: Notebook 4.3 — `04.3-variance-reduction-ii.ipynb` (+ promoción `qflib/mc.py`)

**Files:**
- Create: `notebooks/04-metodos-numericos/04.3-variance-reduction-ii.ipynb`
- Modify: `qflib/mc.py`, `tests/test_mc.py`, `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.plotting.apply_style`, `scipy.stats.qmc.Sobol`.
- Produces (usados por M6.4, M7.4, M11, M12):
  ```python
  def mc_estimate(payoffs, confidence=0.95):
      """Return (mean, standard_error, (ci_low, ci_high)) for a 1-D array of iid discounted payoffs."""

  def normal_sampler(n_paths, dim, rng, method="pseudo", antithetic=False, scramble=True):
      """Return an (n_paths, dim) array of N(0,1) draws.

      method: "pseudo" (rng.standard_normal) or "sobol" (scipy.stats.qmc.Sobol -> inverse CDF).
      antithetic=True returns n_paths/2 draws and their negatives (n_paths must be even).
      scramble applies only to "sobol" (Owen scrambling, so the QMC error is estimable).
      """
  ```

**Teoría:** (1) **Importance sampling:** cambiar la medida de muestreo `\mathbb{E}_P[f] = \mathbb{E}_Q[f\,\frac{dP}{dQ}]` y elegir `Q` que ponga masa donde el payoff no es cero; el estimador de varianza mínima es el que hace `f\cdot\frac{dP}{dQ}` constante (inalcanzable, exige conocer la respuesta), y en la práctica se usa un **desplazamiento de deriva** `Z \to Z+\mu` con razón de verosimilitud `e^{-\mu Z-\mu^2/2}`. Regla práctica: poner `\mu` en el punto donde el integrando `f\cdot\phi` es máximo (el *mode matching* de Glasserman-Heidelberger-Shahabuddin); para un call con strike `K` eso da `\mu^* = \frac{\ln(K/S_0)-(r-q-\sigma^2/2)T}{\sigma\sqrt{T}}` cuando el strike está lejos. **Advertencia obligatoria:** un `\mu` mal elegido *aumenta* la varianza, y puede volverla infinita si la razón de verosimilitud tiene cola pesada — es la técnica con más filo de todas. Conectar con 3.5: valuar bajo la medida del stock ya era importance sampling disfrazado (y ahí se midió 1.2× de reducción). (2) **QMC:** las secuencias de baja discrepancia llenan el cubo más uniformemente que el azar; Koksma-Hlawka acota el error por `V(f)\cdot D_n^*` con `D_n^* = O((\log n)^d/n)`, es decir **casi `O(1/n)`** en vez de `O(n^{-1/2})` — pero (i) la cota es inútil cuantitativamente (`V(f)` suele ser infinita para payoffs con kink) y (ii) el determinismo destruye la barra de error. La solución práctica es el **scrambling de Owen**: aleatoriza manteniendo la baja discrepancia, así que `R` réplicas scrambled dan una estimación honesta del error. Mencionar la dimensión efectiva y el *Brownian bridge* como la razón de que QMC funcione en dimensión alta (se usa en M7).

**Demos:** (1) call deep OTM `K=180` (precio exacto `0.0286428581`, con `\mu^* = 2.7889`): MC crudo vs IS con `\mu^*`, reportando factor de reducción de varianza (esperar >20×) y el número de trayectorias ITM en cada uno — con `n=10^5` el MC crudo produce del orden de 200 trayectorias ITM, que es la imagen del problema; (2) barrido de `\mu \in [0,4]` graficando la varianza del estimador IS: mínimo cerca de `\mu^*` y crecimiento explosivo a la derecha — la demostración visual del filo; (3) Sobol scrambled vs pseudoaleatorio en el call europeo: RMSE sobre `R=32` réplicas vs `n \in \{2^8,\dots,2^{16}\}` en log-log, con las pendientes ajustadas; (4) el mismo experimento en una asiática de 32 fechas (dimensión 32) para mostrar la degradación de QMC con la dimensión, y cómo el Brownian bridge la recupera parcialmente.

**Validación:** IS vs `bs_price(K=180)` dentro de 3 SE; `Var_plain/Var_IS > 20`; el mínimo del barrido de `\mu` cae dentro de `\mu^* \pm 0.4`; Sobol scrambled vs `bs_price` dentro de 3 SE de su error entre réplicas; pendiente log-log de RMSE de Sobol `< -0.7` y la de pseudoaleatorio en `[-0.6,-0.4]`; `mc_estimate` y `normal_sampler` cubiertos por pytest (ver steps).

**Referencias:** Glasserman (2004) caps. 4.6 y 5; Glasserman, Heidelberger & Shahabuddin (1999); Owen (1997) *Scrambled Net Variance*; Caflisch, Morokoff & Owen (1997) (dimensión efectiva).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 4.3 (reducción de varianza II)`
- [x] Tests primero en `tests/test_mc.py`: `mc_estimate` sobre una muestra determinista conocida (media y SE calculados a mano); cobertura del IC (`0.95` nominal sobre 2000 réplicas de `n=200`, esperar `[0.93,0.97]`); `normal_sampler` con `antithetic=True` suma exactamente 0 por columna y tiene la forma pedida; `method="sobol"` con `scramble=False` reproduce la secuencia de `scipy` y con `scramble=True` cambia con el seed pero mantiene media `≈0` y varianza `≈1` (atol 0.05 con `n=2^14`); error si `antithetic=True` con `n_paths` impar
- [x] Implementar en `qflib/mc.py`; `conda run -n qfcurriculum pytest tests/ -v` verde
- [x] Commit separado `feat: qflib.mc (mc_estimate y normal_sampler para QMC/antitéticas)`

---

### Task 4: Notebook 4.4 — `04.4-longstaff-schwartz.ipynb` (+ promoción `qflib/lsm.py`)

**Files:**
- Create: `notebooks/04-metodos-numericos/04.4-longstaff-schwartz.ipynb`, `qflib/lsm.py`, `tests/test_lsm.py`
- Modify: `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.mc.gbm_paths`, `qflib.mc.mc_estimate` (Task 3), `qflib.plotting.apply_style`.
- Produces (usados por M7.4 y M11.1):
  ```python
  def longstaff_schwartz(paths, payoff_fn, discount_factors, basis="poly", degree=3, itm_only=True):
      """Least-squares Monte Carlo price of a Bermudan option.

      paths: (n_paths, n_dates+1) array of underlying values, column 0 = t=0.
      payoff_fn: callable S -> intrinsic value, applied elementwise.
      discount_factors: (n_dates,) array, discount from t_i to t_{i-1}.
      basis: "poly" (monomials) or "laguerre"; degree = highest basis index.
      itm_only: regress only on in-the-money paths (Longstaff-Schwartz's own recommendation).

      Returns (price, standard_error, exercise_dates_used).
      """
  ```

**Teoría:** el problema es el de 3.4 (envolvente de Snell) pero en un mundo donde no hay retícula: en Monte Carlo las trayectorias van *hacia adelante* y la parada óptima necesita el valor de continuación, que mira *hacia atrás* — la razón de que el MC ingenuo no sirva para americanas. La idea de Longstaff-Schwartz: el valor de continuación es una **esperanza condicional** `C_i(S) = \mathbb{E}[D_{i+1}V_{i+1}\mid S_i=S]`, y una esperanza condicional es la **proyección `L^2`** sobre las funciones medibles de `S_i`; aproximarla por la proyección sobre un subespacio finito (polinomios de grado `\leq k`) es una regresión por mínimos cuadrados sobre las trayectorias. Detalles que importan y hay que justificar: (i) regresar **sólo sobre las trayectorias ITM**, porque son las únicas donde la decisión es no trivial y así el ajuste no gasta grados de libertad donde el payoff es 0; (ii) usar la regresión **sólo para decidir**, y acumular el flujo realizado — no el valor ajustado — para no heredar el error de regresión en el precio; (iii) el estimador resulta **sesgado a la baja** (la política estimada es subóptima) y además hay un sesgo *al alza* por usar la misma muestra para estimar la política y valuarla: enunciar la solución estándar (muestras independientes para política y valuación; cotas duales de Andersen-Broadie) con referencia, sin implementar el dual. Bermudan → americana: el precio crece con el número de fechas de ejercicio y converge al americano.

**Demos:** (1) put americano canónico con `m=50` fechas, `n=10^5`, base polinomial de grado 3: precio vs el binomial de 3.4 (`6.0899899526`) y vs el europeo — debe caer entre ambos; (2) **la frontera de ejercicio recuperada de la regresión** graficada sobre la frontera binomial de 3.4, que es la comprobación visual de que la regresión está aprendiendo lo correcto; (3) sensibilidad al grado del polinomio (`1..6`) y a la base (monomios vs Laguerre): tabla de precios y comentario sobre por qué es tan robusto; (4) convergencia Bermudan → americana con `m \in \{1,2,4,\dots,64\}` fechas: `m=1` debe reproducir el europeo exacto dentro de 3 SE (control duro), y `m=64` acercarse al binomial.

**Validación:** `m=1` vs `bs_price(put)` dentro de 3 SE; `europeo \leq LSM \leq binomial + 3SE` en el caso canónico; `|LSM - binomial|/binomial < 0.01` con `m=50, n=10^5`; precios con grados 3..6 dentro de 1% entre sí; QuantLib `MCAmericanEngine` (mismas fechas y trayectorias nominales) dentro de 3 SE combinados; tests pytest de `longstaff_schwartz`.

**Referencias:** Longstaff & Schwartz (2001); Tsitsiklis & Van Roy (2001); Andersen & Broadie (2004) (cotas duales); Glasserman (2004) cap. 8.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 4.4 (Longstaff-Schwartz)`
- [ ] Tests primero en `tests/test_lsm.py`: con una sola fecha de ejercicio el resultado iguala el payoff descontado europeo simulado (atol 1e-12 sobre trayectorias fijas); sobre trayectorias deterministas de 2 estados construidas a mano, el precio iguala el valor de Snell calculado a mano (atol 1e-12); `itm_only=False` y `True` coinciden cuando todas las trayectorias están ITM; error si `discount_factors` no tiene largo `n_dates`
- [ ] Implementar `qflib/lsm.py`; `pytest tests/ -v` verde
- [ ] Commit separado `feat: qflib.lsm (Longstaff-Schwartz reusable en M7 y M11)`

---

### Task 5: Notebook 4.5 — `04.5-finite-differences-i.ipynb`

**Files:**
- Create: `notebooks/04-metodos-numericos/04.5-finite-differences-i.ipynb`
- Modify: `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.plotting.apply_style`, `scipy.linalg.solve_banded`.
- Produces: nada fuera del notebook (ver decisión de no promover el solver en File Structure).

**Teoría:** de la PDE de Black-Scholes (3.1) a la **ecuación de calor** por el cambio de variables `x=\ln S`, `\tau=T-t`, `u=e^{\alpha x+\beta\tau}V` — hacer la derivación completa y decir por qué se hace: con `x=\ln S` los coeficientes dejan de depender del espacio y la malla queda uniforme en log-precio, que es donde el problema realmente es homogéneo. Discretización: diferencias centradas en espacio (`O(\Delta x^2)`), y en tiempo el **θ-scheme** `u^{n+1} = u^n + \Delta\tau[\theta L u^{n+1} + (1-\theta)Lu^n]`, que interpola explícito (`θ=0`, `O(\Delta\tau)`), implícito (`θ=1`, `O(\Delta\tau)`) y Crank-Nicolson (`θ=1/2`, `O(\Delta\tau^2)`). **Estabilidad por von Neumann:** sustituir el modo `u_j^n = \xi^n e^{ikj\Delta x}$ y obtener el factor de amplificación `\xi(k) = \frac{1-(1-\theta)\lambda\kappa}{1+\theta\lambda\kappa}` con `\kappa = 4\sin^2(k\Delta x/2)` y `\lambda = \Delta\tau/\Delta x^2`; de `|\xi|\leq1` sale que `θ\geq1/2` es **incondicionalmente estable** y que para `θ<1/2` hace falta la condición CFL `\lambda \leq \frac{1}{2(1-2\theta)}` — para el explícito, `\Delta\tau \leq \Delta x^2/2`, que es la razón de que nadie use el explícito en producción. Estructura tridiagonal del sistema y por qué `solve_banded` lo resuelve en `O(N)` en vez de `O(N^3)`.

**Demos:** (1) call europeo por los tres esquemas contra `bs_price` sobre toda la malla de `S`, con el gráfico del error vs `S` (máximo cerca del strike, donde vive la convexidad); (2) **la inestabilidad en vivo**: explícito con `\lambda` justo por debajo y justo por encima de `1/2`, graficando la solución oscilando y explotando — con el assert correspondiente de que efectivamente explota; (3) órdenes de convergencia: error vs `\Delta\tau` con `\Delta x` fijo y muy fino para aislar el tiempo, log-log con las pendientes de implícito (`≈1`) y Crank-Nicolson (`≈2`); y error vs `\Delta x` con `\Delta\tau` fino (pendiente `≈2`); (4) el factor de amplificación `|\xi(k)|` graficado vs `k\Delta x` para varios `θ` y `\lambda`, superponiendo la frontera de estabilidad — la teoría de von Neumann hecha figura.

**Validación:** implícito y CN vs `bs_price` en `S_0=100` con malla `N_x=800, N_\tau=400` en `atol=5e-3`; explícito con `\lambda=0.6` produce `\max|u| > 10^3` (assert de la explosión) y con `\lambda=0.4` queda en `atol=5e-3`; pendiente temporal de CN en `[1.8, 2.2]` y de implícito en `[0.85, 1.15]`; pendiente espacial en `[1.8, 2.2]`; `|\xi|` numérico vs la fórmula analítica en `atol=1e-12`.

**Referencias:** Wilmott, Howison & Dewynne (1995) caps. 8-9; Duffy (2006) *Finite Difference Methods in Financial Engineering*; Tavella & Randall (2000); Smith (1985) (von Neumann).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 4.5 (diferencias finitas I)`

---

### Task 6: Notebook 4.6 — `04.6-finite-differences-ii.ipynb`

**Files:**
- Create: `notebooks/04-metodos-numericos/04.6-finite-differences-ii.ipynb`
- Modify: `README.md`, `notebooks/04-metodos-numericos/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price`, `qflib.plotting.apply_style`, `scipy.linalg.solve_banded`. Reimplementa el θ-scheme de 4.5 en el mismo notebook (regla qflib: no se promovió).
- Produces: nada.

**Teoría:** (1) **Por qué Crank-Nicolson falla con payoffs discontinuos:** su factor de amplificación tiende a `-1` para modos de alta frecuencia, así que el error inicial no se amortigua sino que **oscila**; un payoff digital (o el kink de un vanilla) excita exactamente esos modos. El remedio estándar es **Rannacher**: arrancar con 2 (o 4) medios pasos completamente implícitos (`θ=1`, que sí amortigua) y seguir con CN — recuperando segundo orden con las oscilaciones muertas. (2) **Condiciones de frontera:** Dirichlet a partir del comportamiento asintótico (`V\to0` para `S\to0` en un call, `V\to S e^{-qT}-Ke^{-r\tau}` para `S\to\infty`), Neumann/lineales (`\partial^2V/\partial S^2\to0`), y el criterio para poner el borde lo bastante lejos (típico `\ln S \pm 5\sigma\sqrt{T}`); qué error introduce truncar cerca. (3) **Grids no uniformes:** concentrar nodos donde la solución tiene curvatura (el strike, la barrera) mediante el mapeo `\sinh` de Tavella-Randall; el precio a pagar es que las diferencias centradas pierden un orden salvo que se usen las fórmulas ponderadas correctas — darlas explícitamente. (4) **Barreras:** poner la barrera **exactamente sobre un nodo** (si no, el error es `O(\Delta x)` en vez de `O(\Delta x^2)`, el análogo en PDE del sesgo de monitoreo de 3.3). (5) **Americanas por PSOR:** la desigualdad variacional de 3.4 discretizada es un LCP; PSOR es Gauss-Seidel con sobre-relajación proyectando `\max(\cdot, \text{payoff})` en cada barrido; enunciar convergencia para matrices M (referencia) y comparar con el operator splitting.

**Demos:** (1) digital cash-or-nothing por CN puro vs CN+Rannacher: el gráfico de las oscilaciones alrededor del strike y su desaparición, más la tabla de error; (2) impacto de la frontera: mismo call con el borde a `2\sigma\sqrt T`, `3\sigma\sqrt T`, `5\sigma\sqrt T` — error vs distancia del borde; (3) DOC con `B=90` sobre grid uniforme con barrera *entre* nodos vs grid alineado a la barrera vs grid `sinh` concentrado en la barrera, contra el analítico de 3.3 (`8.6654716582`): tabla de error y órdenes; (4) put americano por PSOR vs el binomial de 3.4 (`6.0899899526`), con la frontera de ejercicio extraída del grid superpuesta a la de 3.4 y el número de iteraciones PSOR vs el parámetro de relajación `\omega \in [1.0,1.9]`.

**Validación:** digital CN+Rannacher vs analítico `atol=1e-3` y `\max` oscilación de CN puro >5× la de Rannacher (assert de ambos); call con borde a `5\sigma\sqrt T` vs `bs_price` `atol=5e-3`; DOC con grid alineado vs analítico `atol=1e-2` y error del grid desalineado >3× el alineado; PSOR vs binomial `rtol=2e-3`; PSOR vs QuantLib `FdBlackScholesVanillaEngine` (mismo tamaño de malla) `atol=5e-3`.

**Referencias:** Rannacher (1984); Giles & Carter (2006) *Convergence Analysis of Crank-Nicolson and Rannacher Time-Marching*; Tavella & Randall (2000) caps. 4-6; Wilmott, Howison & Dewynne (1995) cap. 9 (PSOR); Duffy (2006).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 4.6 (diferencias finitas II)`

---

## Cierre de fase

- [ ] Re-ejecutar **todos** los notebooks del repo desde cero a un directorio temporal (no in-place) y confirmar 23/23 OK — el gate que en la Fase 1 fue la única verificación real de reproducibilidad
- [ ] `conda run -n qfcurriculum pytest tests/ -v` verde (64 tests actuales + los nuevos de `mc` y `lsm`)
- [ ] Confirmar que todas las filas de M0–M4 están en ✅ en `README.md` y en los READMEs de módulo
- [ ] Push final y continuar con el plan de Fase 3 (M5–M7)
