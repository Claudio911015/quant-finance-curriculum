# Fase 3 — Curvas de tasas, modelos de tasa corta, HJM/LMM (M5-M7): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development o superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Completar los 13 notebooks de M5 (curvas de tasas), M6 (modelos de tasa corta) y M7 (HJM/LMM), promoviendo a `qflib` la clase de curva de descuento y los generadores de cotizaciones sintéticas de swaps/caps/swaptions que el spec reserva para este tramo.

**Architecture:** Un notebook = una tarea, con los steps fijos de la "Plantilla de tarea por notebook" del plan de Fase 0 (`docs/superpowers/plans/2026-07-29-fase-0-esqueleto.md`). Esta fase tiene una dependencia secuencial fuerte que las anteriores no tenían: M5 construye la curva de descuento que M6 usa como dato de calibración, y M6 (Hull-White) es el caso Markoviano de una dimensión que M7 (HJM/LMM) generaliza — cada módulo literalmente reutiliza el resultado matemático del anterior, no sólo el código. `qflib.curves.DiscountCurve` se promueve en 5.2 (la primera vez que hace falta un objeto curva reusable) y las cotizaciones sintéticas de swaps/caps/swaptions se añaden a `qflib.market` en los puntos donde el spec las pide explícitamente (≥3 notebooks las consumen: bootstrapping, calibración HW, calibración LMM).

**Tech Stack:** conda env `qfcurriculum`; NumPy/SciPy (`scipy.optimize.least_squares` para calibración, `scipy.interpolate` sólo como referencia — los interpoladores de curva se derivan desde cero); QuantLib-Python para validación (`YieldTermStructure`, `HullWhite`, `TreeSwaptionEngine`, `BlackCapFloorEngine`); nbformat + nbconvert.

**Spec:** `docs/superpowers/specs/2026-07-29-quant-curriculum-design.md` (§M5-M7 y "Formato estándar de notebook").

## Global Constraints

- Prosa en español; términos técnicos, código, nombres y comentarios en inglés. LaTeX en Markdown cells. Rigor de posgrado: derivaciones completas salvo donde se indique "enunciado + referencia".
- Estructura de todo notebook (plantilla 0.1): título+motivación → teoría (2-4 celdas MD) → setup (`apply_style()`, `rng = np.random.default_rng(42)` donde haya MC) → demos (pares MD+code) → validación con `assert`s → referencias.
- Fecha de valuación fija en todo el módulo: `eval_date = ql.Date(29, 7, 2026)`, `Actual365Fixed()` para QuantLib, `NullCalendar()` salvo que 5.1 discuta calendarios explícitamente.
- Curva "verdadera" sintética por defecto: `qflib.market.nelson_siegel_zero`/`nelson_siegel_df` con los parámetros por defecto del módulo (`beta0=0.045, beta1=-0.01, beta2=0.01, tau=2.0`) — la misma curva subyacente en M5, M6 y M7, así que un notebook posterior puede reusar los nodos de mercado de uno anterior sin recalcular.
- Seed fijo `np.random.default_rng(42)` en toda simulación. MC de tasas: pasos de tiempo mensuales o más finos según el producto; runtime objetivo <60s por celda, <5 min por notebook.
- Ejecución: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <nb>` (900s en 6.2/7.3/7.4 por la calibración) — limpio, sin errores ni warnings en outputs.
- **Regla de validación (lecciones de las Fases 1-2, no negociable):** (a) un estimador con discretización no converge a la fórmula continua — comparar contra la referencia que corresponde (discreta si existe, o continua con la corrección conocida), nunca con una tolerancia holgada que esconda el sesgo; (b) antes de usar un estadístico como diagnóstico (varianza de un estimador no-iid, n_eff, orden de convergencia en un punto singular), verificar que mide lo que se cree que mide — la Fase 2 encontró tres casos donde la métrica obvia daba el veredicto invertido; (c) al comparar un valor de malla/grid contra una fórmula cerrada, evaluar la fórmula en el nodo exacto, no en un valor nominal.
- **Las tolerancias numéricas de cada tarea son puntos de partida a verificar, no números finales.** Antes de escribir el notebook, prototipar la numérica crítica de la tarea en un script suelto (como en las Fases 1-2) y ajustar tolerancias a lo que el prototipo mide, documentando el porqué en el notebook si difiere del punto de partida de este plan.
- matplotlib mathtext: `\geq`/`\leq` en código; `\ge`/`\le` válidos en Markdown. Todo plot con título, ejes etiquetados y legend cuando hay >1 serie.
- READMEs: al completar cada notebook, marcar ✅ en README principal y en el README del módulo correspondiente (`notebooks/05-curvas/README.md`, `06-tasa-corta/README.md`, `07-hjm-lmm/README.md` — ya tienen columna Estado).
- Commits: `feat: notebook M.N (<tema corto>)`, sin Co-Authored-By. Un notebook = un commit; promociones a qflib en commit propio `feat: qflib.<mod> …`.
- Regla qflib: el notebook desarrolla su tema desde cero; sólo importa de qflib lo transversal ya derivado en módulos anteriores.

## File Structure

**Notebooks (crear), en orden de dependencia:**
- `notebooks/05-curvas/05.1-instruments-conventions.ipynb`
- `notebooks/05-curvas/05.2-single-curve-bootstrapping.ipynb`
- `notebooks/05-curvas/05.3-multi-curve-ois-discounting.ipynb`
- `notebooks/05-curvas/05.4-interpolation-methods.ipynb`
- `notebooks/05-curvas/05.5-sensitivities-dv01-krd.ipynb`
- `notebooks/06-tasa-corta/06.1-vasicek-hull-white.ipynb`
- `notebooks/06-tasa-corta/06.2-hull-white-calibration-trinomial-tree.ipynb`
- `notebooks/06-tasa-corta/06.3-cir-cir-plus-plus.ipynb`
- `notebooks/06-tasa-corta/06.4-rate-exposure-simulation.ipynb`
- `notebooks/07-hjm-lmm/07.1-hjm-framework-drift-condition.ipynb`
- `notebooks/07-hjm-lmm/07.2-lmm-forward-dynamics.ipynb`
- `notebooks/07-hjm-lmm/07.3-lmm-calibration-correlation.ipynb`
- `notebooks/07-hjm-lmm/07.4-lmm-simulation-bermudan-swaption.ipynb`

**qflib (crear/modificar):**
- `qflib/curves.py` (crear en Tarea 2) — clase `DiscountCurve`: construida desde nodos `(t, df)`, expone `df(t)`, `zero_rate(t)`, `forward_rate(t1, t2)`; interpolador log-lineal en discount factor por defecto, con parámetro `interpolator` para 5.4. Reusada por M5.3-5.5, M6, M7 y (más adelante) M11-M12.
- `qflib/market.py` (modificar en Tareas 2, 7 y 12) — añadir `par_swap_rate(curve, schedule)` (2), `synthetic_cap_vols(curve, strikes, tenors, ...)` y `synthetic_swaption_vols(curve, expiries, tenors, ...)` (7 y 12): cotizaciones sintéticas coherentes con la curva "verdadera", reusadas por ≥3 notebooks (6.2, 7.2, 7.3) según el propio alcance del spec para `market.py`.

**Tests (crear/modificar):**
- `tests/test_curves.py` (crear) — casos de `DiscountCurve`.
- `tests/test_market.py` (modificar) — casos de `par_swap_rate`, `synthetic_cap_vols`, `synthetic_swaption_vols`.

**Docs (modificar):** `README.md`, `notebooks/05-curvas/README.md`, `notebooks/06-tasa-corta/README.md`, `notebooks/07-hjm-lmm/README.md`, este plan.

---

### Task 1: Notebook 5.1 — `05.1-instruments-conventions.ipynb`

**Files:**
- Create: `notebooks/05-curvas/05.1-instruments-conventions.ipynb`
- Modify: `README.md`, `notebooks/05-curvas/README.md`

**Interfaces:**
- Consumes: `qflib.plotting.apply_style`. Ningún objeto de curva todavía — este notebook define instrumentos, no los bootstrapea.
- Produces: nada para otras tareas (definiciones y fórmulas locales, reusadas conceptualmente pero no importadas como código).

**Teoría:** convenciones de conteo de días (`Actual/360`, `Actual/365 Fixed`, `30/360 Bond Basis`) y por qué existen — cada mercado fija el día base por convención histórica, no por elegancia matemática; el efecto numérico de usar la convención equivocada en una misma tasa cotizada. Composición simple vs compuesta vs continua y las fórmulas de conversión entre ellas. **Depósitos**: tasa simple sobre `Actual/360`, precio = 1 nocional descontado. **FRA**: payoff `N·(L(T_1,T_2)-K)·\tau/(1+L\tau)` liquidado en `T_1` (descontado un período) vs `N·(L-K)\tau` liquidado en `T_2`; derivar la tasa forward implícita `L(T_1,T_2)=\frac{1}{\tau}\left(\frac{P(0,T_1)}{P(0,T_2)}-1\right)` desde no-arbitraje (réplica con dos depósitos). **Futuros de tasa** (ED/SOFR futures): el ajuste de convexidad futuro-forward, `\text{futures rate} - \text{forward rate} \approx \frac{1}{2}\sigma^2 T_1T_2`, enunciado con la derivación esquemática (el futuro se marca a mercado diario, así que bajo la medida spot su valor esperado es distinto al forward bajo la medida forward — conexión directa con 3.5) y referencia a Hull-White (1990) para la fórmula cerrada bajo su modelo. **Swaps** (fijo-flotante) y **OIS** (overnight indexado, con la fórmula de capitalización del promedio geométrico del overnight compounding). Calendarios y convenciones de ajuste de fecha de negocio (`Following`, `Modified Following`) — enunciado breve, sin implementar un calendario festivo completo.

**Demos:** (1) tabla comparando el mismo flujo de caja anualizado bajo `Act/360`, `Act/365F` y `30/360` para ver la magnitud de la diferencia; (2) FRA: replicar la tasa forward con dos depósitos sintéticos (sobre la curva Nelson-Siegel) y verificar que el payoff del FRA a la tasa forward es exactamente cero en valor presente (por construcción); (3) el ajuste de convexidad futuro-forward evaluado numéricamente para un par de vencimientos, mostrando que crece con $T_1T_2$; (4) una pata OIS con compounding diario vs una tasa simple equivalente, mostrando la pequeña diferencia de convexidad del compounding.

**Validación:** día-conteo propio vs `ql.Actual360()`, `ql.Actual365Fixed()`, `ql.Thirty360(ql.Thirty360.BondBasis)` sobre 10 pares de fechas aleatorias, `atol=1e-12` (son fracciones racionales, deben coincidir exactamente salvo redondeo de floats); tasa forward implícita de FRA vs `ql.Actual360` + descuento manual de la curva Nelson-Siegel, `atol=1e-10`; el payoff del FRA a la tasa par es cero, `atol=1e-10`.

**Referencias:** Hull, J.C. *Options, Futures, and Other Derivatives*, 10th ed., caps. 4, 6, 33 (convexidad futuros-forward); Brigo & Mercurio (2006) cap. 1 (convenciones); ISDA 2006 Definitions (day count, business day conventions, referencia).

- [x] Escribir notebook (nbformat) según la plantilla y las secciones de arriba
- [x] Ejecutar in-place limpio; los asserts pasan
- [x] READMEs actualizados
- [x] Commit `feat: notebook 5.1 (instrumentos y convenciones)`

---

### Task 2: Notebook 5.2 — `05.2-single-curve-bootstrapping.ipynb` (+ promoción `qflib/curves.py`, `qflib.market.par_swap_rate`)

**Files:**
- Create: `notebooks/05-curvas/05.2-single-curve-bootstrapping.ipynb`, `qflib/curves.py`, `tests/test_curves.py`
- Modify: `qflib/market.py`, `tests/test_market.py`, `README.md`, `notebooks/05-curvas/README.md`

**Interfaces:**
- Consumes: `qflib.market.nelson_siegel_df`, `qflib.plotting.apply_style`.
- Produces (usados por 5.3, 5.4, 5.5, M6, M7):
  ```python
  # qflib/curves.py
  class DiscountCurve:
      def __init__(self, times, discount_factors, interpolator="log_linear"):
          """times: increasing array starting > 0. interpolator: "log_linear" (5.2) or
          "log_cubic"/"monotone_convex" (5.4, must already be registered — see 5.4)."""

      def df(self, t):
          """Discount factor P(0,t), scalar or array."""

      def zero_rate(self, t):
          """Continuously-compounded zero rate z(t) = -ln(df(t))/t."""

      def forward_rate(self, t1, t2):
          """Simple forward rate over [t1,t2]: (df(t1)/df(t2) - 1) / (t2 - t1)."""

  # qflib/market.py (nueva función)
  def par_swap_rate(curve, payment_times):
      """Par fixed rate of a vanilla swap with annual fixed leg at payment_times (float years),
      floating leg = 1 - df(payment_times[-1]) on the same curve (single-curve assumption).
      Returns df(0)-normalized par rate: (1 - df(T_n)) / sum(tau_i * df(T_i))."""
  ```

**Teoría:** el problema del bootstrapping: dado un conjunto de instrumentos cotizados a la par (depósitos cortos, swaps largos), encontrar la curva de descuento que los reprecia exactamente. Derivar el bootstrapping secuencial: los depósitos fijan los primeros nodos directamente (`df(t)=1/(1+L\cdot t)`); cada swap sucesivo, con todos los nodos anteriores ya conocidos, deja **una sola incógnita** (el discount factor en su propio vencimiento) porque el spec de la pata fija es `\sum_i\tau_i\,\text{df}(T_i)\cdot K = 1-\text{df}(T_n)$ y todos los `df(T_i)` con `i<n` ya se conocen — de ahí sale `df(T_n)` en forma cerrada, nodo por nodo, sin resolver un sistema. Interpolación **log-lineal en discount factor** entre nodos (equivalente a tasa forward constante a trozos) como el caso base, señalando ya el defecto que resuelve 5.4: la tasa forward instantánea es discontinua en cada nodo.

**Demos:** (1) generar cotizaciones sintéticas (depósitos a 1M/3M/6M, swaps anuales a 1Y..10Y) evaluando exactamente la curva Nelson-Siegel "verdadera" con `par_swap_rate`; (2) bootstrapear `DiscountCurve` a partir de esas cotizaciones y verificar que reproduce la curva verdadera en los nodos (por construcción) y **entre** nodos (por la interpolación, con algún error); (3) graficar la curva zero bootstrapeada vs la verdadera, y la curva forward instantánea (discontinua) que resulta de la interpolación log-lineal.

**Validación:** cada cotización sintética se reprecia con la curva bootstrapeada a `atol=1e-10` (por construcción del algoritmo); la curva bootstrapeada vs `ql.PiecewiseLogLinearDiscount` construida con las mismas cotizaciones (mismos `ql.DepositRateHelper`/`ql.SwapRateHelper`), discount factors en los nodos, `atol=1e-8`; error entre nodos vs la curva Nelson-Siegel verdadera, `rtol=1e-3` (hay error de interpolación real entre nodos, no debe forzarse a cero).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 5.2 (bootstrapping de curva única)`
- [x] Tests primero en `tests/test_curves.py`: `DiscountCurve` con 2 nodos conocidos reproduce `df`/`zero_rate`/`forward_rate` a mano (atol 1e-12); `df(t)` en un nodo exacto iguala el discount factor dado; error si `times` no es estrictamente creciente o no empieza `>0`
- [x] Tests en `tests/test_market.py`: `par_swap_rate` sobre una curva plana (df exponencial con tasa constante `y`) da par rate `\approx y` (con compounding anual, rtol 1e-6 para tenors largos donde la aproximación discreta-continua converge)
- [x] Implementar `qflib/curves.py` y la función en `qflib/market.py`; `pytest tests/ -v` verde
- [x] Commit separado `feat: qflib.curves (DiscountCurve) y qflib.market.par_swap_rate`

---

### Task 3: Notebook 5.3 — `05.3-multi-curve-ois-discounting.ipynb`

**Files:**
- Create: `notebooks/05-curvas/05.3-multi-curve-ois-discounting.ipynb`
- Modify: `README.md`, `notebooks/05-curvas/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.market.nelson_siegel_df`, `qflib.market.par_swap_rate`, `qflib.plotting.apply_style`.
- Produces: nada nuevo en qflib (reusa `DiscountCurve` con dos instancias).

**Teoría:** por qué una sola curva deja de alcanzar tras 2008: la tasa de descuento de mercado (colateral, típicamente overnight/OIS) y la tasa de proyección de los flujos flotantes (Libor/Euribor a 3M o 6M, con riesgo de crédito bancario embebido) dejan de ser la misma curva — el **tenor basis** entre ellas es observable en el mercado (basis swaps) y no converge a cero. El bootstrapping se separa en dos etapas: (1) la curva OIS se bootstrapea de instrumentos OIS con la fórmula estándar (ahora la "curva de descuento" $P_{\text{OIS}}(0,t)$); (2) la curva de proyección se bootstrapea de swaps par cuya pata flotante paga Libor-3M pero cuyo **descuento** usa $P_{\text{OIS}}$, no $P_{\text{proj}}$ — la ecuación par cambia a $\sum_i\tau_i F_i(0)P_{\text{OIS}}(0,T_i) = K\sum_i\tau_iP_{\text{OIS}}(0,T_i)$ con $F_i(0)$ la tasa forward de la curva de proyección, que ahora sí es la única incógnita en cada paso. Mostrar la fórmula del basis swap tenor-tenor y cómo un basis positivo desplaza la curva de proyección por encima de la OIS.

**Demos:** (1) construir dos curvas Nelson-Siegel "verdaderas" con parámetros distintos (proyección con `beta0` ligeramente mayor, simulando el basis); (2) bootstrapear ambas por separado con sus propios instrumentos sintéticos (OIS con descuento en sí misma; swaps 3M con descuento en la curva OIS ya bootstrapeada); (3) graficar el tenor basis implícito $F_{\text{proj}}(t)-F_{\text{OIS}}(t)$ recuperado, comparado contra el basis verdadero de las dos curvas Nelson-Siegel; (4) mostrar el error de valuación (típicamente pequeño pero no cero) de ignorar el multi-curva y descontar con una sola curva.

**Validación:** curva OIS bootstrapeada reproduce sus cotizaciones a `atol=1e-10`; curva de proyección (con descuento OIS) reproduce sus cotizaciones swap a `atol=1e-10`; basis implícito recuperado vs el basis verdadero de las dos Nelson-Siegel, `rtol=5e-3`; contraste contra QuantLib con `ql.OISRateHelper` para la curva de descuento y `ql.SwapRateHelper` con `discountingTermStructure` apuntando a la curva OIS ya construida, discount factors en los nodos comunes `atol=1e-7`.

**Referencias:** Hull, J.C. cap. 9 (OIS discounting); Mercurio, F. (2010) *A LIBOR Market Model with Stochastic Basis*, o su versión determinista más simple: Ametrano & Bianchetti (2009) *Bootstrapping the Illiquidity*.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 5.3 (multi-curva y OIS discounting)`

---

### Task 4: Notebook 5.4 — `05.4-interpolation-methods.ipynb`

**Files:**
- Create: `notebooks/05-curvas/05.4-interpolation-methods.ipynb`
- Modify: `qflib/curves.py`, `README.md`, `notebooks/05-curvas/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve` (extendida aquí), `qflib.market.nelson_siegel_df`, `par_swap_rate`, `qflib.plotting.apply_style`.
- Produces (extiende `qflib.curves.DiscountCurve`, usado desde M6 en adelante donde importe una curva suave):
  ```python
  # DiscountCurve gana los interpoladores "log_cubic" y "monotone_convex" (Section 3),
  # seleccionables via el parametro interpolator ya definido en la Tarea 2.
  ```

**Teoría:** el defecto de log-lineal (5.2): la tasa forward instantánea $f(t)=-\partial_t\ln P(0,t)$ es **constante a trozos y discontinua** en los nodos — inaceptable para instrumentos sensibles a la forma de la curva forward (caps, cualquier cosa con fixing entre nodos). **Spline cúbico natural en $\ln P(0,t)$**: interpola suavemente (forward continua) pero puede producir **oscilaciones** (overshoot de Runge) y, peor, tramos donde el forward implícito se vuelve **negativo** aunque los nodos sean crecientes en tasa — derivar la condición de spline natural (segunda derivada nula en los extremos) y mostrar el fenómeno con un ejemplo concreto de la curva Nelson-Siegel con un salto de pendiente entre nodos adyacentes. **Monotone convex de Hagan-West (2006)**: el método que domina en la práctica porque garantiza forwards positivos y localmente monótonos entre nodos por construcción, a costa de una interpolación más intrincada (parabólica por tramos con ajuste de los valores en los nodos para preservar monotonía) — implementar el algoritmo completo desde el paper (sección de construcción de $f_i$ en los nodos y la interpolación parabólica con los tres casos de la prueba de monotonía), no una aproximación.

**Demos:** (1) las tres curvas forward instantáneas (log-lineal, spline cúbico, monotone convex) superpuestas sobre los mismos nodos de la Tarea 2, con el forward negativo del spline marcado explícitamente si aparece con estos nodos (si no aparece con la curva suave de Nelson-Siegel, forzar un ejemplo con un nodo perturbado a mano que sí lo produzca — el punto pedagógico es mostrar el fallo, no solo mencionarlo); (2) tabla de error de reprecio de las cotizaciones originales para las tres (deben ser $\approx0$ en todas, por construcción, ya que las tres pasan por los nodos exactamente); (3) sensibilidad de un forward de 5Y5Y (`forward_rate` entre nodos) a un shock de 1bp en el nodo adyacente, comparando cuán localizado queda el efecto en cada interpolador (monotone convex debe ser el más localizado).

**Validación:** las tres interpolaciones reproducen los discount factors en los nodos exactos a `atol=1e-12` (identidad de interpolación); el forward instantáneo de monotone convex es `\geq0$` en toda la malla fina de prueba (assert de la propiedad que motiva el método) incluso en el ejemplo con el nodo perturbado, mientras el spline cúbico natural sí produce al menos un forward negativo ahí (assert de que el fallo es real, no hipotético); monotone convex propio vs `ql.MonotonicLogCubicNaturalCubic` (o el interpolador monótono más cercano disponible en QuantLib-Python para la curva) en los mismos nodos, `atol=1e-6` en discount factors sobre la malla fina.

**Referencias:** Hagan, P.S. & West, G. (2006). *Interpolation Methods for Curve Construction*. Applied Mathematical Finance, 13(2), 89-129.; Hagan & West (2008) *Methods for Constructing a Yield Curve*, WILMOTT Magazine.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 5.4 (métodos de interpolación)`

---

### Task 5: Notebook 5.5 — `05.5-sensitivities-dv01-krd.ipynb`

**Files:**
- Create: `notebooks/05-curvas/05.5-sensitivities-dv01-krd.ipynb`
- Modify: `README.md`, `notebooks/05-curvas/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.market.par_swap_rate`, `qflib.plotting.apply_style`.
- Produces: nada nuevo en qflib.

**Teoría:** **DV01** (dollar value of a basis point): sensibilidad del valor de un instrumento a un desplazamiento paralelo de 1bp en toda la curva de tasas — derivar la fórmula cerrada de un swap par ($\text{DV01}\approx\sum_i\tau_iP(0,T_i)$, la anualidad, hasta primer orden) y contrastarla contra la diferencia finita central bump-and-reprice, mostrando que coinciden hasta $O(\Delta^2)$. **Key-rate durations (KRD)**: en vez de un shock paralelo, perturbar **un solo nodo** de la curva bootstrapeada (manteniendo los demás fijos) y medir el cambio de valor — la suma de las KRD debe reproducir el DV01 paralelo (identidad de descomposición), y el perfil de KRDs muestra a qué parte de la curva es sensible cada instrumento (un swap 10Y concentra su riesgo en el nodo 10Y, con "leakage" hacia nodos vecinos por la interpolación). **Jacobiano de calibración**: la matriz $\partial(\text{tasa zero en nodo } j)/\partial(\text{cotización de mercado } i)$, que en un bootstrapping secuencial (5.2) es **triangular** (el nodo $j$ sólo depende de cotizaciones con vencimiento $\leq T_j$) — derivar esa estructura y calcularla por diferencias finitas, verificando que los elementos fuera de la triangular son cero.

**Demos:** (1) DV01 analítico vs bump-and-reprice de un swap 10Y, con el bump en $\pm1$bp y $\pm5$bp para ver la convergencia de la diferencia finita; (2) perfil completo de KRDs de swaps 2Y, 5Y y 10Y sobre la malla de nodos de 5.2, y verificación de que $\sum_j\text{KRD}_j=\text{DV01}$ para cada uno; (3) el jacobiano de calibración completo como un mapa de calor, mostrando la estructura triangular; (4) el mismo jacobiano usado para propagar un error de cotización de mercado (p.ej. una cotización swap mal capturada por 2bp) al error resultante en la curva zero.

**Validación:** DV01 analítico vs bump-and-reprice (bump $\pm1$bp), `atol=1e-6` en unidades de valor presente por unidad nocional; $\sum_j\text{KRD}_j$ vs DV01 paralelo, `rtol=1e-3` (la descomposición no es exacta si la interpolación no es lineal en los shocks, pero debe ser muy cercana); elementos del jacobiano fuera de la estructura triangular, `atol=1e-8` (deben ser exactamente cero salvo ruido de diferencia finita); DV01 propio vs QuantLib (`ql.BasisPointSensitivity` o bump-and-reprice manual de un `ql.VanillaSwap` con la curva de 5.2 reconstruida en QuantLib), `atol=1e-5`.

**Referencias:** Ho, T.S.Y. (1992). *Key Rate Durations: Measures of Interest Rate Risks*. Journal of Fixed Income, 2(2), 29-44.; Tuckman, B. & Serrat, A. (2011). *Fixed Income Securities*, 3rd ed., Wiley, cap. 5.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 5.5 (sensibilidades: DV01, KRD, jacobiano)`

---

### Task 6: Notebook 6.1 — `06.1-vasicek-hull-white.ipynb`

**Files:**
- Create: `notebooks/06-tasa-corta/06.1-vasicek-hull-white.ipynb`
- Modify: `README.md`, `notebooks/06-tasa-corta/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.market.nelson_siegel_zero`/`nelson_siegel_df`, `qflib.mc.ou_paths` (Vasicek es exactamente un proceso OU en $r_t$, ya derivado en 1.4), `qflib.plotting.apply_style`.
- Produces: nada en qflib (las fórmulas de bono son locales al notebook; se reusan conceptualmente en 6.2-6.4 pero cada notebook las redefine según la regla de dependencia, salvo lo que se decida promover explícitamente — aquí no se promueve nada porque cada notebook posterior necesita variantes distintas de la fórmula, ver Tareas 7-9).

**Teoría:** **Vasicek**: $dr_t=\kappa(\theta-r_t)dt+\sigma dW_t$ bajo $\mathbb{Q}$ — el mismo proceso OU de 1.4, ahora como tasa corta. El precio de bono $P(t,T)=\mathbb{E}^{\mathbb{Q}}_t[e^{-\int_t^Tr_sds}]$ se deriva resolviendo la PDE de tasa corta (análoga a Black-Scholes con $S\to r$) con el *ansatz* afín $P(t,T)=A(t,T)e^{-B(t,T)r_t}$, sustituyendo y separando en una ODE para $B$ (Riccati lineal, resoluble en cerrado) y una para $A$ — dar la derivación completa hasta las fórmulas cerradas de $A,B$. El defecto estructural de Vasicek con parámetros constantes: la curva de tasas que genera es una familia de dos-tres parámetros, **no puede ajustar exactamente** una curva de mercado arbitraria (es sobre-restrictiva). **Hull-White** ($\theta\to\theta(t)$ determinista) resuelve exactamente eso: derivar $\theta(t)$ en función de la curva forward instantánea observada $f(0,t)$ y sus derivadas, de modo que $P^{HW}(0,T)$ reproduce $P^{\text{mercado}}(0,T)$ para **todo** $T$ simultáneamente — el "exact fit" del título. Mostrar la fórmula de bono de Hull-White en términos de la curva de mercado (sin pasar por $\theta(t)$ explícito), que es la forma que de verdad se usa en producción: $P^{HW}(t,T)=\frac{P^M(0,T)}{P^M(0,t)}\exp\left(B(t,T)f^M(0,t)-\frac{\sigma^2}{4a}B(t,T)^2(1-e^{-2at})\right)$.

**Demos:** (1) Vasicek: bono cerrado vs integración numérica de la esperanza por MC (usando `ou_paths` de 1.4/qflib.mc con $10^5$ trayectorias) para verificar la fórmula antes de usarla como referencia en el resto del notebook; (2) mostrar explícitamente que Vasicek con parámetros fijos **no puede** encajar la curva Nelson-Siegel de 5.2 (graficar la curva de mercado vs la mejor curva Vasicek por mínimos cuadrados, con el error residual visible); (3) Hull-White ajustado a la misma curva Nelson-Siegel: la curva de bonos $P^{HW}(0,T)$ reproduce la de mercado exactamente en toda una malla de vencimientos; (4) simulación de trayectorias de tasa corta bajo Hull-White (con `theta(t)` tabulado) y verificación de que el bono promedio simulado (MC) coincide con la fórmula cerrada.

**Validación:** bono Vasicek cerrado vs MC (mismos parámetros), dentro de 3 SE con $n=10^5$; Hull-White ajustado reproduce $P^M(0,T)$ en 20 vencimientos de la malla Nelson-Siegel, `atol=1e-8` (es una identidad algebraica, no una aproximación); Hull-White propio vs `ql.HullWhite` model con la curva de 5.2 reconstruida como `ql.YieldTermStructureHandle` en QuantLib, `discountBond(t,T,r_t)` para varios `(t,T,r_t)`, `atol=1e-8`.

**Referencias:** Vasicek, O. (1977). *An Equilibrium Characterization of the Term Structure*. Journal of Financial Economics, 5(2), 177-188.; Hull, J.C. & White, A. (1990). *Pricing Interest-Rate-Derivative Securities*. Review of Financial Studies, 3(4), 573-592.; Brigo, D. & Mercurio, F. (2006). *Interest Rate Models*, 2nd ed., Springer, cap. 3.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 6.1 (Vasicek y Hull-White)`

---

### Task 7: Notebook 6.2 — `06.2-hull-white-calibration-trinomial-tree.ipynb` (+ promoción `qflib.market.synthetic_cap_vols`, `synthetic_swaption_vols`)

**Files:**
- Create: `notebooks/06-tasa-corta/06.2-hull-white-calibration-trinomial-tree.ipynb`
- Modify: `qflib/market.py`, `tests/test_market.py`, `README.md`, `notebooks/06-tasa-corta/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.black.bs_price` (Black-76 para caps/swaptions, derivado en 3.5 §2), `qflib.plotting.apply_style`, la fórmula de bono Hull-White de 6.1 (redefinida localmente).
- Produces (usados por 7.2 y 7.3):
  ```python
  # qflib/market.py
  def synthetic_cap_vols(strikes, tenors, base_vol=0.20, skew=-0.02, term_slope=-0.01):
      """Synthetic Black cap-vol surface: base_vol + skew*(K - atm_rate) + term_slope*log(tenor),
      coherent in shape with a typical downward-sloping term structure and negative skew.
      Returns a (len(tenors), len(strikes)) array of Black vols."""

  def synthetic_swaption_vols(expiries, tenors, base_vol=0.18, expiry_slope=-0.015):
      """Synthetic Black swaption-vol surface, base_vol + expiry_slope*log(expiry).
      Returns a (len(expiries), len(tenors)) array of Black vols."""
  ```

**Teoría:** precio de **caplet** bajo Hull-White en forma cerrada (Jamshidian, 1989): un caplet es una opción sobre un bono cupón cero, y bajo un modelo de un factor la opción sobre bono tiene fórmula cerrada tipo Black con volatilidad $\sigma_P(t,T_1,T_2)=\frac{\sigma}{a}(1-e^{-a(T_2-T_1)})\sqrt{\frac{1-e^{-2at}}{2a}}$ — derivar de dónde sale esa volatilidad (la varianza de $\ln P(t,T)$ bajo Hull-White, afín en $r_t$) y la fórmula de Jamshidian para una swaption como opción sobre un portafolio de bonos, usando que en un modelo de un factor el precio del bono es monótono en $r_t$ así que el ejercicio óptimo tiene un único punto de corte $r^*$ (el "truco de Jamshidian"). **Calibración**: minimizar la suma de errores cuadráticos entre los precios Black de mercado (de la superficie sintética) y los precios Hull-White cerrados, sobre $(a,\sigma)$ — con `scipy.optimize.least_squares`. **Árbol trinomial de Hull-White** (Hull-White, 1994): discretización del proceso en una retícula recombinante con tres ramas por nodo, probabilidades elegidas para matchear media y varianza del proceso continuo, y un desplazamiento aditivo por nivel de tiempo para el exact-fit a la curva (el análogo discreto de $\theta(t)$) — derivar las probabilidades estándar del árbol (caso central, y los dos casos de rama especiales cuando la reversión obliga a un patrón de ramificación distinto cerca de los bordes).

**Demos:** (1) superficie de vols de cap sintética generada con `synthetic_cap_vols`, convertida a precios Black usando la curva Hull-White ajustada de 6.1 como curva de descuento/proyección; (2) calibración de $(a,\sigma)$ por mínimos cuadrados contra esos precios, con la superposición de vols de mercado vs vols implicadas por el Hull-White calibrado; (3) construcción del árbol trinomial con los parámetros calibrados y re-precio de los mismos caps sobre el árbol, comparando contra la fórmula cerrada de Jamshidian (deben casi coincidir: el árbol converge a la fórmula continua); (4) precio de una swaption (usando la fórmula de Jamshidian) con los parámetros calibrados, comparado con la vol de swaption sintética convertida a precio Black.

**Validación:** caplet Jamshidian vs `ql.BlackCapFloorEngine`/`ql.Gaussian1dSwaptionEngine`... en la práctica, contra `ql.HullWhite` + `ql.TreeCapFloorEngine` con los mismos $(a,\sigma)$ calibrados, `atol=1e-6` en precio; parámetros calibrados $(a,\sigma)$ propios vs los que produce `ql.HullWhite` calibrado con `ql.HullWhite.calibrate` sobre los mismos `ql.CapHelper`, `rtol=5e-2` (la calibración numérica puede converger a óptimos ligeramente distintos con solvers distintos — se verifica que ambos reprecian el mercado igual de bien, no que los parámetros coincidan exactamente); árbol trinomial vs Jamshidian cerrado, `rtol=1e-3` con un árbol de al menos 100 pasos por año.

**Referencias:** Jamshidian, F. (1989). *An Exact Bond Option Formula*. Journal of Finance, 44(1), 205-209.; Hull, J.C. & White, A. (1994). *Numerical Procedures for Implementing Term Structure Models I: Single-Factor Models*. Journal of Derivatives, 2(1), 7-16.; Brigo & Mercurio (2006) caps. 3-4.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 6.2 (calibración Hull-White y árbol trinomial)`
- [x] Tests primero en `tests/test_market.py`: `synthetic_cap_vols`/`synthetic_swaption_vols` devuelven arrays con la forma pedida, todos los valores `>0`, y el vol ATM (strike = tasa par) es exactamente `base_vol` en el primer tenor/expiry (por construcción de la fórmula)
- [x] Implementar en `qflib/market.py`; `pytest tests/ -v` verde
- [x] Commit separado `feat: qflib.market (superficies sinteticas de cap y swaption vols)`

---

### Task 8: Notebook 6.3 — `06.3-cir-cir-plus-plus.ipynb`

**Files:**
- Create: `notebooks/06-tasa-corta/06.3-cir-cir-plus-plus.ipynb`
- Modify: `README.md`, `notebooks/06-tasa-corta/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.mc.cir_paths` (derivado en 1.4/promovido en Fase 0), `qflib.plotting.apply_style`.
- Produces: nada en qflib.

**Teoría:** **CIR** ($dr_t=\kappa(\theta-r_t)dt+\sigma\sqrt{r_t}dW_t$, ya simulado en 1.4 con full-truncation Euler) tiene precio de bono cerrado con la misma estructura afín $P(t,T)=A(t,T)e^{-B(t,T)r_t}$ pero $A,B$ salen de una ODE de Riccati **no lineal** (a diferencia de Vasicek); dar la solución cerrada completa en términos de $\gamma=\sqrt{\kappa^2+2\sigma^2}$. La condición de **Feller** $2\kappa\theta\geq\sigma^2$ garantiza que $r_t$ nunca toca cero — conectar con la discusión de 1.4 sobre el esquema full-truncation cuando Feller falla. El mismo defecto de Vasicek reaparece: CIR con parámetros constantes no ajusta una curva de mercado arbitraria. **CIR++** (Brigo-Mercurio, 2001): $r_t=x_t+\varphi(t)$ con $x_t$ un CIR "puro" y $\varphi(t)$ un desplazamiento determinista elegido para el exact-fit, análogo al $\theta(t)$ de Hull-White pero aditivo sobre el proceso en vez de sobre el drift — derivar $\varphi(t)=f^M(0,t)-f^{CIR}(0,t;x_0,\kappa,\theta,\sigma)$ y notar la ventaja sobre Hull-White puro: $r_t$ puede mantenerse no-negativo (si Feller se cumple para $x_t$) incluso con el exact fit, mientras que Hull-White con volatilidad constante siempre permite tasas negativas.

**Demos:** (1) bono CIR cerrado vs `qflib.mc.cir_paths` + MC, para dos parametrizaciones (Feller cumplida y Feller violada) verificando que la fórmula cerrada sigue siendo válida en ambos casos (la fórmula no depende de Feller, sólo la interpretación de la trayectoria); (2) CIR con parámetros fijos no ajusta la curva Nelson-Siegel (igual que 6.1); (3) CIR++ ajustado a la misma curva, exact-fit verificado en una malla de vencimientos; (4) simulación de trayectorias de $r_t=x_t+\varphi(t)$ bajo CIR++ con Feller cumplida para $x_t$, mostrando que las tasas simuladas permanecen no-negativas, y comparación de la fracción de tiempo con $r_t<0$ bajo CIR++ vs bajo un Hull-White con volatilidad calibrada al mismo nivel (mostrando la ventaja práctica).

**Validación:** bono CIR cerrado vs MC dentro de 3 SE, en ambos regímenes de Feller; CIR++ ajustado reproduce $P^M(0,T)$ en 20 vencimientos, `atol=1e-8`; bono CIR cerrado propio vs una implementación de referencia (QuantLib no expone CIR de forma directamente comparable — usar como referencia la fórmula cerrada evaluada con `mpmath`/precisión extendida o simplemente el MC de altísima precisión, $n=10^6$, dentro de 3 SE, documentando por qué no hay comparación directa contra QuantLib aquí).

**Referencias:** Cox, J.C., Ingersoll, J.E. & Ross, S.A. (1985). *A Theory of the Term Structure of Interest Rates*. Econometrica, 53(2), 385-407.; Brigo, D. & Mercurio, F. (2001). *A Deterministic-Shift Extension of Analytically-Tractable and Time-Homogeneous Short-Rate Models*. Finance and Stochastics, 5(3), 369-387.; Brigo & Mercurio (2006) cap. 3.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 6.3 (CIR y CIR++)`

---

### Task 9: Notebook 6.4 — `06.4-rate-exposure-simulation.ipynb`

**Files:**
- Create: `notebooks/06-tasa-corta/06.4-rate-exposure-simulation.ipynb`
- Modify: `README.md`, `notebooks/06-tasa-corta/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.mc.mc_estimate` (promovido en 4.3), la fórmula de bono/simulación de Hull-White de 6.1, `qflib.plotting.apply_style`.
- Produces: nada en qflib (el bridge conceptual a M11 no requiere promoción — M11 construirá su propio motor de exposición, más general, sobre estas ideas).

**Teoría:** el puente a XVA (M11): valuar un swap **hoy** da un número; gestionar el riesgo de contraparte exige conocer la **distribución** del valor del swap en cada fecha futura, bajo el modelo de tasas que ya se tiene (Hull-White de 6.1-6.2 calibrado). Definir **Expected Exposure** $EE(t)=\mathbb{E}^{\mathbb{Q}}[\max(V(t),0)]$ y **Potential Future Exposure** $PFE_\alpha(t)=\inf\{x:\mathbb{Q}(V(t)\leq x)\geq\alpha\}$ (el cuantil $\alpha$, típicamente 95%) — la asimetría $\max(\cdot,0)$ es la razón estructural de por qué la exposición de un swap sin colateral no es cero aunque su valor a mercado hoy sí lo sea: la opcionalidad de que la contraparte incumpla sólo cuando el swap está a favor de quien mide la exposición. Valuar el swap en cada fecha futura simulada requiere el valor **condicional** de la pata flotante y fija dado el estado del modelo en ese instante — con Hull-White esto tiene fórmula cerrada (reusando el bono $P^{HW}(t,T;r_t)$ de 6.1 para descontar cada flujo futuro desde la fecha de simulación).

**Demos:** (1) simular $10^4$ trayectorias de tasa corta bajo Hull-White calibrado (6.2) hasta el vencimiento de un swap 5Y; (2) en cada fecha de simulación, valuar el swap remanente usando los bonos Hull-White cerrados (no otra simulación anidada); (3) el perfil de $EE(t)$ y $PFE_{95\%}(t)$ a lo largo de la vida del swap, con la forma característica de "joroba" de un swap par (exposición crece mientras hay muchos flujos futuros inciertos, decrece al acercarse al vencimiento); (4) verificación de martingala: el valor descontado del swap remanente promediado sobre las trayectorias en $t=0$ debe reproducir el valor de mercado en $t=0$ (control de que la simulación no tiene error de deriva).

**Validación:** valor del swap en $t=0$ (promedio MC de las trayectorias, descontado) vs el valor de mercado calculado directamente de la curva Hull-White de 6.1, dentro de 3 SE; $EE(0)=\max(V(0),0)$ exactamente (assert de borde, sin ruido MC porque en $t=0$ no hay incertidumbre); $EE(t)\geq0$ en toda la malla de tiempo (por construcción del `max`, debe cumplirse siempre); pico de la joroba de $EE(t)$ ocurre estrictamente antes del vencimiento del swap (propiedad cualitativa verificable con un assert sobre el argmax).

**Referencias:** Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*, Wiley, cap. 6 (definiciones de exposición). Gregory, J. (2015). *The xVA Challenge*, 3rd ed., Wiley, caps. 6-8.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 6.4 (simulación de exposiciones de tasas)`

---

### Task 10: Notebook 7.1 — `07.1-hjm-framework-drift-condition.ipynb`

**Files:**
- Create: `notebooks/07-hjm-lmm/07.1-hjm-framework-drift-condition.ipynb`
- Modify: `README.md`, `notebooks/07-hjm-lmm/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.mc.normal_sampler` (promovido en 4.3), la fórmula de bono Hull-White de 6.1, `qflib.plotting.apply_style`.
- Produces: nada en qflib (el simulador HJM de este notebook es el ancestro conceptual de 7.2, pero LMM re-deriva su propia dinámica discreta desde cero según la regla de dependencia).

**Teoría:** en vez de modelar la tasa corta y derivar toda la curva de ella (6.1-6.3), HJM modela **directamente** la evolución de toda la curva forward $f(t,T)$ bajo $\mathbb{Q}$: $df(t,T)=\alpha(t,T)dt+\sigma(t,T)dW_t$ — y el precio no-arbitraje de un bono impone una restricción muy fuerte entre $\alpha$ y $\sigma$: la **condición de drift de HJM**. Derivarla completa: partiendo de $P(t,T)=\exp(-\int_t^Tf(t,u)du)$, aplicar Itô a $\ln P(t,T)$, usar que $P(t,T)/B_t$ debe ser martingala bajo $\mathbb{Q}$ (FTAP, 2.3/3.5), e igualar el drift resultante a cero — sale $\alpha(t,T)=\sigma(t,T)\int_t^T\sigma(t,u)du$ (en el caso de un factor; enunciar la versión multi-factor con el producto interno de vectores de volatilidad). El punto central: **una vez fijada $\sigma(t,T)$, el drift queda completamente determinado** — no es libre, es la condición que hace el modelo consistente con no-arbitraje. Mostrar que Hull-White es el caso especial $\sigma(t,T)=\sigma e^{-a(T-t)}$ (volatilidad exponencialmente decreciente, Markoviana en un factor) y verificar que sustituyendo esa $\sigma$ en la condición de drift de HJM se recupera exactamente el drift de Hull-White de 6.1 — la razón por la que HJM con esta $\sigma$ colapsa a un proceso Markoviano de baja dimensión mientras que una $\sigma(t,T)$ genérica no lo hace (necesita toda la historia de la curva, de ahí que LMM en 7.2 simule discretamente en vez de buscar una EDP de baja dimensión).

**Demos:** (1) simular la curva forward completa bajo HJM de un factor con $\sigma(t,T)=\sigma e^{-a(T-t)}$ (discretizando $f(t,T)$ en una malla de $T$, con Euler en $t$) y verificar que el bono resultante coincide con el bono Hull-White cerrado de 6.1 — la comprobación de que el caso especial de HJM *es* Hull-White, no una aproximación; (2) verificar numéricamente la condición de drift directamente: para una $\sigma(t,T)$ dada, calcular $\alpha(t,T)$ por la fórmula y por diferenciación numérica de la deriva simulada, deben coincidir; (3) HJM con una $\sigma(t,T)$ genérica no separable (p.ej. Gaussiana en $T-t$, no exponencial) para mostrar un modelo que **no** colapsa a Hull-White, con la curva forward simulada desarrollando una forma que un modelo de un factor Markoviano no podría producir con la misma parsimonia.

**Validación:** bono del HJM de un factor (caso Hull-White) vs bono Hull-White cerrado de 6.1 **evaluado trayectoria por trayectoria** en el $r_t$ simulado de cada una y promediado después (no promediar $r_t$ primero: $P(t,T)$ es convexa en $r_t$, así que $\mathbb{E}[P(t,T;r_t)]\neq P(t,T;\mathbb{E}[r_t])$ por Jensen — comparar mal esto da un z-score inflado que parece un error de la fórmula y no lo es), dentro de 3 SE con $n\geq2\times10^4$; condición de drift verificada: $\alpha(t,T)$ por fórmula vs por diferenciación numérica de la simulación, `rtol=5e-2` (la diferenciación numérica de una trayectoria simulada tiene su propio error, no es una identidad exacta); la curva forward simulada en $t=0$ coincide con la curva de mercado inicial (por construcción, la condición inicial del HJM), `atol=1e-8`.

**Referencias:** Heath, D., Jarrow, R. & Morton, A. (1992). *Bond Pricing and the Term Structure of Interest Rates: A New Methodology*. Econometrica, 60(1), 77-105.; Brigo & Mercurio (2006) cap. 5.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 7.1 (marco HJM y condición de drift)`

---

### Task 11: Notebook 7.2 — `07.2-lmm-forward-dynamics.ipynb`

**Files:**
- Create: `notebooks/07-hjm-lmm/07.2-lmm-forward-dynamics.ipynb`
- Modify: `README.md`, `notebooks/07-hjm-lmm/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve`, `qflib.mc.normal_sampler`, `qflib.black.bs_price` (Black-76 vía la analogía de 3.5, para el caplet de referencia), `qflib.market.synthetic_cap_vols` (promovido en 6.2), `qflib.plotting.apply_style`.
- Produces: nada en qflib todavía (la simulación completa se promueve implícitamente al reusarse en 7.3-7.4 vía el propio notebook, no vía qflib — el spec reserva `qflib/mc.py` para paths *genéricos*, y el simulador LMM es demasiado específico del producto para promover).

**Teoría:** HJM modela la curva continua; LMM modela directamente el vector de tasas forward **simples** observables $L_i(t)=L(t;T_i,T_{i+1})$ — la razón práctica de su dominio: son las tasas que de verdad se cotizan (Libor/Euribor forward), y su dinámica lognormal es consistente con Black-76 para caplets, que es como el mercado cotiza. Bajo su propia **medida forward** $\mathbb{Q}^{T_{i+1}}$ (3.5), $L_i(t)$ es martingala por construcción — es el forward de un activo negociable (bono de $T_i$ a $T_{i+1}$) bajo el numerario correcto, exactamente el resultado de 3.5 §2 aplicado tasa por tasa. El problema es que un producto real (un swap, un cap) involucra **todas** las tasas simultáneamente, y no pueden vivir todas bajo su propia medida a la vez: hay que elegir **una** medida común, típicamente la **medida terminal** $\mathbb{Q}^{T_N}$ (numerario $P(t,T_N)$) o la **medida spot** (numerario la cuenta bancaria discretamente reinvertida cada período, "rolling"). Derivar el drift que cada $L_i$ adquiere bajo la medida terminal vía el teorema de cambio de numerario en cadena (3.5 §1, aplicado repetidamente entre medidas forward consecutivas): $dL_i=\sigma_iL_i\left(-\sum_{j=i+1}^{N-1}\frac{\tau_j\rho_{ij}\sigma_jL_j}{1+\tau_jL_j}\right)dt+\sigma_iL_idW_i^{T_N}$, con $\rho_{ij}$ la correlación instantánea entre $dW_i$ y $dW_j$. Notar la propiedad crucial que hace el modelo tratable pese a no tener solución cerrada: **cada $L_i$ es individualmente lognormal a primer orden** (aproximación *frozen drift*), lo bastante buena para que el caplet valga aproximadamente Black-76 con la propia volatilidad de $L_i$.

**Demos:** (1) simular un conjunto de 10 tasas forward semestrales bajo la medida terminal, con correlación de ejemplo (matriz de correlación exponencial simple, adelantando la parametrización completa de 7.3) y volatilidades tomadas de `synthetic_cap_vols`; (2) verificar que la tasa **más cercana al vencimiento terminal** ($L_{N-1}$, cuya propia medida *es* la terminal) sale exactamente martingala (drift nulo) en la simulación, mientras las demás muestran deriva no nula — la comprobación directa de la teoría; (3) precio de un caplet vía MC bajo la medida terminal (descontando con el bono simulado hasta $T_N$ y reponderando por el numerario) vs Black-76 con la vol de mercado, para verificar que la aproximación lognormal es consistente con el precio de mercado que la originó; (4) mostrar el efecto del *frozen drift*: comparar el drift "congelado" (evaluado en $t=0$ y mantenido fijo) contra el drift recalculado en cada paso, cuantificando el error que introduce la aproximación estándar de la industria.

**Validación:** $\mathbb{E}[L_{N-1}(T_{N-1})]$ vs $L_{N-1}(0)$ (martingala bajo su propia medida terminal), dentro de 3 SE; caplet MC bajo medida terminal vs Black-76 de mercado, dentro de 3 SE; drift de cada $L_i$ ($i<N-1$) bajo la medida terminal es no nulo con el signo que predice la fórmula (mismo signo que $-\sum\tau_j\rho_{ij}\sigma_jL_j/(1+\tau_jL_j)$, verificado con `np.sign`); error del frozen-drift vs drift recalculado, `rtol=1e-2` en el precio del caplet más lejano (donde el error acumulado es mayor) — cuantificado, no forzado a cero.

**Referencias:** Brace, A., Gatarek, D. & Musiela, M. (1997). *The Market Model of Interest Rate Dynamics*. Mathematical Finance, 7(2), 127-155.; Jamshidian, F. (1997). *LIBOR and Swap Market Models and Measures*. Finance and Stochastics, 1(4), 293-330.; Brigo & Mercurio (2006) cap. 6.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 7.2 (dinámica de forwards en LMM)`

---

### Task 12: Notebook 7.3 — `07.3-lmm-calibration-correlation.ipynb` (+ promoción `qflib.market.synthetic_swaption_vols` si no quedó cubierta en 6.2)

**Files:**
- Create: `notebooks/07-hjm-lmm/07.3-lmm-calibration-correlation.ipynb`
- Modify: `README.md`, `notebooks/07-hjm-lmm/README.md` (y `qflib/market.py`/`tests/test_market.py` sólo si 6.2 no cubrió ya `synthetic_swaption_vols` con la generalidad que aquí hace falta — verificar al ejecutar la Tarea 7 y ajustar esta tarea si sobra)

**Interfaces:**
- Consumes: `qflib.market.synthetic_cap_vols`, `synthetic_swaption_vols` (6.2), `qflib.curves.DiscountCurve`, `qflib.plotting.apply_style`. La dinámica de forwards de 7.2 (redefinida localmente, con la calibración añadida).
- Produces: nada nuevo en qflib (la calibración es específica del notebook).

**Teoría:** dos calibraciones separadas y su interacción. **Volatilidad instantánea**: con vol constante a trozos por tasa forward $\sigma_i(t)=\sigma_i^{(k)}$ en el intervalo $k$-ésimo, calibrar directamente a las vols de cap de mercado (`synthetic_cap_vols`) usando que la vol Black de un caplet iguala aproximadamente la vol RMS de $L_i$ hasta su propio vencimiento: $(\sigma_i^{\text{Black}})^2T_i=\int_0^{T_i}\sigma_i(t)^2dt$ — con vol constante a trozos, esto da una ecuación lineal en $\sigma_i^{(k)2}$, resoluble exactamente sin optimización si sólo se calibra a caps (que es lo que garantiza la estructura triangular del problema, análoga al bootstrapping de 5.2). **Correlación**: parametrización de Rebonato de dos parámetros $\rho_{ij}=\rho_\infty+(1-\rho_\infty)e^{-\beta|T_i-T_j|}$ — de rango completo (positiva definida por construcción, decae con la distancia entre vencimientos, con una correlación de largo plazo $\rho_\infty$ que evita que tasas muy separadas se vuelvan independientes de forma poco realista). **Consistencia con swaptions**: la fórmula de aproximación de Rebonato relaciona la vol de swaption implícita por el LMM calibrado con las vols de caplet y la matriz de correlación — $(\sigma_{\text{swpn}})^2\approx\sum_{i,j}\frac{w_i(0)w_j(0)L_i(0)L_j(0)\rho_{ij}\sigma_i\sigma_j}{S(0)^2}T_{\alpha}$ con $w_i$ los pesos de la tasa swap como combinación de forwards — derivarla (esquemáticamente, con referencia para el detalle de los pesos) y usarla para verificar si la calibración de caplets + correlación paramétrica reproduce razonablemente la superficie de swaption de mercado, o si hace falta ajustar $\beta$ para mejorar el fit (el trade-off clásico: calibrar exacto a caps deja la superficie de swaptions como diagnóstico, no como otro target exacto).

**Demos:** (1) calibración exacta de $\sigma_i^{(k)}$ a la superficie `synthetic_cap_vols`, verificando reprecio exacto de los caps; (2) matriz de correlación de Rebonato para un par $(\rho_\infty,\beta)$ de ejemplo, graficada como mapa de calor, mostrando el decaimiento con la distancia; (3) vols de swaption aproximadas (Rebonato) vs las de `synthetic_swaption_vols`, con el error de ajuste; (4) barrido de $\beta$ (con $\rho_\infty$ fijo) mostrando cómo cambia el error de swaption — encontrar el $\beta$ que minimiza ese error por una búsqueda simple en malla, sin re-optimizar los caplets (que ya están exactos).

**Validación:** vols de caplet calibradas reprecian `synthetic_cap_vols` a `atol=1e-6` en vol (por construcción, sistema lineal exacto); matriz de correlación de Rebonato es simétrica, con diagonal $1$, y semi-definida positiva (`np.linalg.eigvalsh(...) >= -1e-10`) para el rango de $(\rho_\infty,\beta)$ usado — assert de la propiedad que motiva la parametrización; vol de swaption aproximada de Rebonato vs una simulación LMM completa (Monte Carlo del propio notebook, reprecio de la swaption vía el precio de la tasa swap simulada) para al menos un par (expiry, tenor), dentro de `rtol=0.10` (la aproximación de Rebonato es de primer orden, no exacta — se verifica que es razonable, no idéntica).

**Referencias:** Rebonato, R. (1999). *On the Simultaneous Calibration of Multifactor Lognormal Interest Rate Models to Black Volatilities and to the Correlation Matrix*. Journal of Computational Finance, 2(4), 5-27.; Brigo & Mercurio (2006) cap. 6-7.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 7.3 (calibración LMM: caplets y correlación)`

---

### Task 13: Notebook 7.4 — `07.4-lmm-simulation-bermudan-swaption.ipynb`

**Files:**
- Create: `notebooks/07-hjm-lmm/07.4-lmm-simulation-bermudan-swaption.ipynb`
- Modify: `README.md`, `notebooks/07-hjm-lmm/README.md`

**Interfaces:**
- Consumes: `qflib.lsm.longstaff_schwartz` (promovido en 4.4), `qflib.mc.normal_sampler`, `qflib.market.synthetic_cap_vols`/`synthetic_swaption_vols`, `qflib.curves.DiscountCurve`, la dinámica LMM y calibración de 7.2-7.3 (redefinidas localmente). `qflib.mc.mc_estimate` para reportar SE.
- Produces: nada nuevo en qflib — cierra el módulo.

**Teoría:** simulación LMM completa bajo la medida terminal (7.2) con la calibración de 7.3 (vols de caplet + correlación de Rebonato), vía descomposición de Cholesky de la matriz de correlación para generar los brownianos correlacionados a partir de `normal_sampler`. Una **Bermudan swaption**: el derecho a entrar en un swap subyacente en cualquiera de varias fechas de ejercicio (no sólo una, como la europea) — el mismo problema de parada óptima de 3.4/4.4, ahora sobre un vector de estado de alta dimensión (todas las tasas forward vivas en cada fecha de decisión), donde una retícula (3.4) es inviable pero **Longstaff-Schwartz sí escala**: la regresión de 4.4 no le importa la dimensión del estado, sólo necesita elegir una base de funciones de él (aquí, funciones de la tasa swap subyacente en cada fecha de ejercicio, que resume el estado relevante para la decisión aunque no todo el vector de forwards). Cota de sanity: el valor de la Bermudan debe ser **al menos** el de la europea con la fecha de ejercicio más favorable de entre las disponibles (más derechos, nunca menos), replicando el argumento de 3.4 §1 en este contexto de mayor dimensión.

**Demos:** (1) simular $n=5\times10^4$ trayectorias de las 10 tasas forward bajo LMM (medida terminal, correlación de Rebonato calibrada en 7.3, vols calibradas a caps); (2) valuar la swaption **europea** (una sola fecha de ejercicio, la más favorable del conjunto Bermudan) por MC directo, y contra la vol de swaption sintética vía Black-76, como control de que la simulación calibrada es razonable; (3) valuar la **Bermudan** (4 fechas de ejercicio posibles) vía Longstaff-Schwartz, usando como variables de regresión la tasa swap subyacente en cada fecha de decisión y su cuadrado; (4) verificar la cota Bermudan $\geq$ europea, y graficar en qué fechas la política de LSM decide ejercer más frecuentemente sobre las trayectorias simuladas.

**Validación:** swaption europea (MC) vs Black-76 con la vol sintética, dentro de 3 SE; Bermudan (LSM) $\geq$ europea $-$ 3 SE combinadas (la desigualdad estructural, con margen de ruido MC porque ambas son estimadores); control duro análogo al de 4.4: con **una sola** fecha de ejercicio Bermudan (que coincide con la europea), LSM debe reproducir el precio europeo de MC directo sobre las mismas trayectorias, `atol=1e-8` (identidad, no test estadístico); comparación cualitativa final contra un motor de referencia de un factor (`ql.Gaussian1dSwaptionEngine` con un Hull-White calibrado a la misma superficie de caps de 6.2, sabiendo que HW de un factor y LMM multi-factor no tienen por qué coincidir exactamente — se reporta la diferencia y se discute su origen, no se fija una tolerancia dura sobre ella).

**Referencias:** Longstaff, F.A. & Schwartz, E.S. (2001) (reusado de 4.4); Andersen, L. (2000). *A Simple Approach to the Pricing of Bermudan Swaptions in the Multifactor LIBOR Market Model*. Journal of Computational Finance, 3(2), 5-32.; Brigo & Mercurio (2006) cap. 6, §6.7.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 7.4 (simulación LMM y Bermudan swaption)`

---

## Cierre de fase

- [ ] Re-ejecutar **todos** los notebooks del repo desde cero a un directorio temporal y confirmar 36/36 OK (23 de M0-M4 + 13 de esta fase)
- [ ] `conda run -n qfcurriculum pytest tests/ -v` verde (78 tests actuales + los nuevos de `curves` y `market`)
- [ ] Confirmar que todas las filas de M0–M7 están en ✅ en `README.md` y en los READMEs de módulo
- [ ] Actualizar la memoria del proyecto (estado de fases, lecciones nuevas si aparecen)
- [ ] Push final y continuar con el plan de Fase 4 (M8-M9: volatilidad y FX)
