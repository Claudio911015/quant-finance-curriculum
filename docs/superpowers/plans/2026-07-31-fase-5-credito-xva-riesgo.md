# Fase 5 — Crédito, XVA y riesgo de mercado (M10-M12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completar M10 (crédito: Merton/intensidad, CDS, cópulas), M11 (XVAs: exposiciones, CVA/DVA/wrong-way risk, FVA/ColVA/MVA/KVA panorama, netting/CSA) y M12 (riesgo de mercado: VaR/ES, backtesting, sensibilidades/stress testing) — 10 notebooks nuevos, cerrando el currículum completo de 54 notebooks (M0-M12).

**Architecture:** Mismo patrón que Fases 1-4: cada notebook se prototipa primero en un script standalone (verificación numérica contra una referencia independiente) antes de escribirse, se construye con `nbformat`, y se ejecuta limpio con `nbconvert --execute --inplace`. `qflib` crece sólo donde un notebook posterior necesita reusar algo ya derivado desde cero.

**Tech Stack:** Python 3.12 (conda env `qfcurriculum`), NumPy, SciPy (`optimize`, `stats`, `integrate`), QuantLib-Python, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-quant-curriculum-design.md` (secciones M10, M11, M12).

## Global Constraints

- Prosa en español; terminología técnica, código, docstrings y comentarios en inglés.
- Seeds fijos en toda simulación: `np.random.default_rng(42)`.
- Nombres de notebook: ya scaffoldeados en Fase 0 (`10.1-merton-intensity-models.ipynb`, etc. — usar exactamente esos nombres, ver `notebooks/10-credito/README.md`, `notebooks/11-xva/README.md`, `notebooks/12-riesgo-mercado/README.md`).
- Commits firmados con `claudiocp_2@hotmail.com` (config global), **sin** Co-Authored-By.
- Notebooks se commitean con outputs ejecutados.
- Regla de dependencia qflib: un notebook solo importa de `qflib` lo que un módulo anterior ya construyó desde cero.
- Antes de escribir cada notebook: prototipar la numérica crítica en un script standalone y verificarla contra una referencia independiente (patrón establecido en Fases 1-4 — atrapó bugs reales de convención de índices, drift bajo cambio de medida, y superficies con supuestos ocultos en casi todas las fases).
- Python del env: `conda run -n qfcurriculum python` / `conda run -n qfcurriculum pytest` / `conda run -n qfcurriculum jupyter nbconvert ...`.
- Al cerrar la fase: los **54/54** notebooks del repo completo (M0-M12, el currículum entero) deben re-ejecutarse desde cero sin error, y `pytest tests/ -v` debe estar verde.

---

### Task 1: Notebook 10.1 — `10.1-merton-intensity-models.ipynb`

**Files:**
- Create: `notebooks/10-credito/10.1-merton-intensity-models.ipynb`
- Modify: `README.md`, `notebooks/10-credito/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_price` (M3, Merton = call sobre los activos de la firma), `qflib.market.nelson_siegel_df` (curva libre de riesgo para descuento).
- Produces: nada nuevo en qflib.

**Teoría:** **Merton estructural** (1974): el valor de la firma $V_t$ sigue GBM; la deuda (bono cupón cero, cara $D$, vencimiento $T$) se repaga sólo si $V_T\ge D$ — el equity es entonces una call sobre $V_T$ con strike $D$ (Black-Scholes exacto), y la deuda es $V_0$ menos esa call (o, equivalentemente, un bono libre de riesgo menos una put sobre $V_T$: el valor de la opción de default). La probabilidad de default risk-neutral es $N(-d_2)$ de la misma fórmula. **Modelos de intensidad** (reducidos, Jarrow-Turnbull): el default es el primer salto de un proceso de Poisson con intensidad (hazard rate) $\lambda_t$ — no hay un mecanismo económico subyacente (a diferencia de Merton), pero calibra directamente a spreads de mercado; con $\lambda$ constante, $P(\tau>t)=e^{-\lambda t}$ y el bono con riesgo de crédito (recovery cero) vale $P(0,t)e^{-\lambda t}$.

**Demos:** (1) Merton: dados $V_0,\sigma_V,D,T,r$, calcular equity (call), deuda, y probabilidad de default $N(-d_2)$; graficar cómo la probabilidad de default varía con el apalancamiento $D/V_0$ y con $\sigma_V$; (2) verificar la identidad contable $E_0+\text{Deuda}_0=V_0$ (el valor de la firma se reparte exactamente entre equity y deuda, sin residuo); (3) intensidad constante: verificar $P(\tau>t)=e^{-\lambda t}$ por simulación directa de tiempos exponenciales (inversión de CDF, reusa 0.3) y comparar la fracción sobreviviente a la fórmula cerrada; (4) intensidad determinista dependiente del tiempo $\lambda(t)$: $P(\tau>t)=e^{-\int_0^t\lambda(s)ds}$, verificado por integración numérica vs simulación (muestreo por "thinning" o integración de la CDF inversa acumulada).

**Validación:** identidad contable Merton ($E_0+D_0=V_0$) a `atol=1e-10`; supervivencia simulada (intensidad constante, $n=200{,}000$ trayectorias) vs $e^{-\lambda t}$ dentro de 3 SE (proporción binomial, usar $SE=\sqrt{p(1-p)/n}$); supervivencia con $\lambda(t)$ determinista, simulación vs integración numérica de $e^{-\int\lambda}$ dentro de 3 SE.

**Referencias:** Merton, R. (1974). *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*. Journal of Finance, 29(2), 449-470. Jarrow, R., Turnbull, S. (1995). *Pricing Derivatives on Financial Securities Subject to Credit Risk*. Journal of Finance, 50(1), 53-85.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 10.1 (Merton estructural y modelos de intensidad)`

---

### Task 2: Notebook 10.2 — `10.2-cds-pricing-bootstrap.ipynb`

**Files:**
- Create: `notebooks/10-credito/10.2-cds-pricing-bootstrap.ipynb`
- Modify: `README.md`, `notebooks/10-credito/README.md`

**Interfaces:**
- Consumes: `qflib.curves.DiscountCurve` (M5, curva libre de riesgo), `qflib.market.nelson_siegel_df`.
- Produces: nada nuevo en qflib (la curva de intensidad/hazard es específica del notebook, análoga a por qué HW/SABR no promovieron su calibración a qflib).

**Teoría:** un **CDS** paga protección: a cambio de un spread periódico $s$ (premium leg) sobre el nocional, el vendedor de protección paga $(1-R)$ (pérdida dado default, $R$=recovery) si ocurre default antes del vencimiento. Bajo intensidad determinista por tramos $\lambda_i$ (constante en cada intervalo entre nodos de cotización, igual que el bootstrap de vol de 7.3), el spread par que hace el CDS valer cero en $t=0$ se resuelve **secuencialmente**: premium leg $=s\sum_i\Delta_i P(0,T_i)P(\tau>T_i)$ (más el "accrued on default", aproximado con la convención estándar de pago a mitad de periodo), protection leg $=(1-R)\int_0^T P(0,t)\,dP(\tau\le t)$ — con hazard por tramos, ambas integrales son sumas cerradas, y el spread de mercado en cada tenor pin-pointea el siguiente $\lambda_i$, exactamente análogo al bootstrapping de curvas de 5.2.

**Demos:** (1) precio de un CDS a spread de mercado dado (premium leg vs protection leg) con curva de hazard ya conocida, verificar NPV=0 en el spread par; (2) bootstrap de la curva de hazard por tramos desde una serie de spreads de mercado sintéticos (CDS 1Y,3Y,5Y,7Y,10Y, spreads crecientes con el tenor — curva de crédito "normal") — recursión secuencial, reprecio exacto de los spreads de entrada; (3) survival probability implícita de la curva de hazard bootstrapeada, graficada junto al spread de crédito; (4) sensibilidad CS01 (análogo a DV01 de 5.5): shock paralelo de +1bp en todos los spreads de entrada, ver el cambio en el valor de un CDS existente a spread fijo (no a mercado) — signo consistente (spread sube -> protección vale más -> CDS existente comprado a spread viejo más bajo gana valor).

**Validación:** reprecio exacto de los spreads de CDS de entrada tras el bootstrap, `atol=1e-8` en spread; NPV=0 del CDS a su propio spread par, `atol=1e-10`; CS01 con signo consistente en todos los tenores (mismo criterio que el basis01 de 9.3 y el DV01 de 5.5).

**Referencias:** O'Kane, D., Turnbull, S. (2003). *Valuation of Credit Default Swaps*. Lehman Brothers Quantitative Credit Research. Brigo, D., Mercurio, F. (2006), cap. 21-22 (crédito).

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 10.2 (CDS: pricing y bootstrap de curva de crédito)`

---

### Task 3: Notebook 10.3 — `10.3-copulas-default-correlation.ipynb`

**Files:**
- Create: `notebooks/10-credito/10.3-copulas-default-correlation.ipynb`
- Modify: `README.md`, `notebooks/10-credito/README.md`

**Interfaces:**
- Consumes: la curva de hazard bootstrapeada de 10.2 (redefinida localmente para 2-3 nombres con distintos spreads), `qflib.mc.normal_sampler` (M4, para simular el factor gaussiano correlacionado de la cópula).
- Produces: nada nuevo en qflib.

**Teoría:** el riesgo de correlación de default entre varios nombres de crédito no está determinado por las curvas de hazard marginales por sí solas — se necesita una **cópula** para acoplar los tiempos de default marginales en una distribución conjunta. **Cópula gaussiana** (industria estándar pre-2008, todavía pedagógicamente central): transformar cada tiempo de default marginal $\tau_i$ a una variable uniforme vía su propia CDF de supervivencia, luego mapear esas uniformes a un vector gaussiano multivariado correlacionado (matriz de correlación $\rho$) — el mecanismo estándar de simulación de cópulas (genera $Z\sim N(0,\Sigma)$, $U_i=\Phi(Z_i)$, $\tau_i=S_i^{-1}(1-U_i)$ donde $S_i$ es la supervivencia marginal del nombre $i$). Producto de interés: la distribución del número de defaults en una cesta (**basket**) hasta un horizonte, y cómo la correlación cambia su cola.

**Demos:** (1) simular tiempos de default para 5 nombres con curvas de hazard marginales distintas (spreads de crédito distintos), bajo cópula gaussiana con $\rho$ variable, verificando que las marginales simuladas (ignorando la correlación, mirando cada nombre por separado) reproducen su propia $S_i(t)$ marginal — la cópula NO debe distorsionar las marginales, sólo la dependencia conjunta; (2) distribución del número de defaults en la cesta a un horizonte fijo para $\rho\in\{0,0.3,0.7\}$ — a mayor $\rho$, más masa en "0 defaults" y en "todos default" simultáneamente (más bimodal), menos en resultados intermedios; (3) precio de un CDS de $k$-ésimo-default (first-to-default, $k=1$) vía MC, verificado en el límite $\rho\to1$ (todos los nombres defaultean juntos, first-to-default = default del nombre con el spread más alto) y en el límite $\rho\to0$ (nombres independientes, fórmula cerrada de $P(\min_i\tau_i>t)=\prod_iS_i(t)$).

**Validación:** marginales simuladas vs $S_i(t)$ analítica para cada uno de los 5 nombres, dentro de 3 SE (proporción binomial); límite $\rho\to0$ del first-to-default recupera $\prod_i S_i(t)$ dentro de 3 SE; límite $\rho\to1$ recupera $\min_i$ determinista de las supervivencias individuales (verificar que el nombre con mayor hazard casi siempre defaultea primero, `> 95%` de las trayectorias, documentado como chequeo probabilístico, no exacto).

**Referencias:** Li, D. (2000). *On Default Correlation: A Copula Function Approach*. Journal of Fixed Income, 9(4), 43-54. Schönbucher, P., Schubert, D. (2001). *Copula-dependent default risk in intensity models*.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 10.3 (cópulas y correlación de default)`

---

### Task 4: Notebook 11.1 — `11.1-exposures-ee-epe-pfe.ipynb`

**Files:**
- Create: `notebooks/11-xva/11.1-exposures-ee-epe-pfe.ipynb`
- Modify: `README.md`, `notebooks/11-xva/README.md`

**Interfaces:**
- Consumes: la simulación de exposición de tasas de 6.4 (`r_t=f(0,t)+\alpha(t)+x_t`, redefinida localmente), `qflib.market.par_swap_rate`, `qflib.mc.ou_paths` (M1, para el factor $x_t$ de Hull-White).
- Produces: nada nuevo en qflib.

**Teoría:** **exposición** de un swap (u otro derivado) en una fecha futura $t$ es $\max(V_t,0)$ para el lado que puede perder si la contraparte incumple (a diferencia del valor $V_t$ que puede ser negativo). **EE(t)** (expected exposure) $=E[\max(V_t,0)]$; **EPE** (expected positive exposure) es el promedio temporal de EE sobre un horizonte; **PFE** (potential future exposure) es un cuantil alto (típicamente 95-99%) de la distribución de $V_t$ en cada fecha — reusa exactamente la simulación de tasas de 6.4 (Hull-White, `r_t=f(0,t)+\alpha(t)+x_t`) para simular la curva futura en cada fecha de observación y revaluar un swap (o portafolio de 2-3 swaps con distinto signo, para ilustrar el efecto de netting parcial).

**Demos:** (1) simular la trayectoria de tasas de 6.4 hasta el vencimiento de un swap, revaluar el swap en cada fecha de observación (usando la curva simulada de ESE instante, vía `par_swap_rate`-style descuento), obtener el perfil de $V_t$ por trayectoria; (2) EE(t) y PFE(t) al 95% sobre la malla de fechas, graficados (el perfil típico "en forma de joroba" de un swap: exposición crece al inicio por la incertidumbre acumulada de tasas, decae al final por el roll-down del nocional restante); (3) EPE como promedio temporal de EE; (4) portafolio de 2 swaps con signos opuestos (netting bajo el mismo CSA/contraparte) vs sin netting (dos contrapartes separadas) — el EE del portafolio neteado es menor o igual a la suma de los EE individuales (desigualdad de Jensen aplicada a $\max(\cdot,0)$ de una suma vs suma de $\max(\cdot,0)$s).

**Validación:** EE(t) $\ge0$ en toda fecha (trivial pero se verifica); EPE = promedio temporal de EE(t) por construcción, verificado a `atol` de integración numérica; EE del portafolio neteado $\le$ suma de EE individuales en cada fecha (la desigualdad estructural de netting, análoga al argumento de dominio de opciones americanas de 3.4 y Bermudan $\ge$ European de 7.4).

**Referencias:** Gregory, J. (2015). *The xVA Challenge*, cap. 3-4. Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*, cap. 4.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 11.1 (exposiciones: EE/EPE/PFE)`

---

### Task 5: Notebook 11.2 — `11.2-cva-dva-wrong-way-risk.ipynb`

**Files:**
- Create: `notebooks/11-xva/11.2-cva-dva-wrong-way-risk.ipynb`
- Modify: `README.md`, `notebooks/11-xva/README.md`

**Interfaces:**
- Consumes: la simulación de exposición de 11.1 (redefinida localmente), la curva de hazard/supervivencia de 10.2 (redefinida localmente, para la contraparte y para la entidad propia).
- Produces: nada nuevo en qflib.

**Teoría:** **CVA** (credit valuation adjustment) $=(1-R_C)\int_0^T EE(t)\,dP(\tau_C\le t)$ — el ajuste al valor del derivado por el riesgo de default de la CONTRAPARTE, integrando la exposición esperada positiva contra la densidad de default de la contraparte; **DVA** es el simétrico con la exposición NEGATIVA (desde la perspectiva de la contraparte, el riesgo de que la entidad propia defaultee) y la propia curva de hazard de la entidad. **Wrong-way risk**: cuando la exposición y la probabilidad de default de la contraparte están CORRELACIONADAS (p.ej., una contraparte cuyo negocio empeora precisamente cuando las tasas se mueven en la dirección que aumenta la exposición del swap) — CVA calculado ignorando esa correlación subestima (o sobreestima, "right-way risk") el riesgo real; se simula acoplando el factor de tasas $x_t$ de 11.1 con la intensidad de default de la contraparte vía una cópula gaussiana simplificada (reusa 10.3).

**Demos:** (1) CVA vía la fórmula de integración discreta sobre la malla de fechas de 11.1 (EE(t) de esa simulación, curva de supervivencia de la contraparte de 10.2), reportado en valor y en bps del nocional; (2) DVA simétrico, verificando que $\text{CVA}-\text{DVA}$ tiene el signo esperado según quién esté más "in the money" en expectativa; (3) introducir wrong-way risk: correlacionar el factor de tasas con la intensidad de default de la contraparte (a través de un factor gaussiano compartido, cópula simple) y mostrar que el CVA sube cuando la correlación implica que la contraparte tiende a defaultear justo cuando la exposición es alta (y baja, o incluso se vuelve right-way risk, con el signo de correlación opuesto); (4) caso límite: correlación cero recupera el CVA "independiente" de la Demo 1 exactamente.

**Validación:** CVA con correlación cero coincide con el CVA de la Demo 1 (mismo cálculo, dos caminos) dentro de 3 SE de MC; CVA bajo wrong-way risk (correlación positiva exposición-hazard) $>$ CVA independiente, y bajo right-way risk (correlación negativa) $<$ CVA independiente — chequeo direccional, no de magnitud exacta (documentado como tal, ya que la magnitud depende de la calibración específica del acoplamiento).

**Referencias:** Gregory, J. (2015), cap. 5-8 (CVA/DVA), cap. 17 (wrong-way risk). Brigo, D., Pallavicini, A., Papatheodorou, V. (2011). *Arbitrage-Free Valuation of Bilateral Counterparty Risk for Interest-Rate Products: Impact of Volatilities and Correlations*.

- [x] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 11.2 (CVA/DVA y wrong-way risk)`

---

### Task 6: Notebook 11.3 — `11.3-fva-colva-mva-kva.ipynb`

**Files:**
- Create: `notebooks/11-xva/11.3-fva-colva-mva-kva.ipynb`
- Modify: `README.md`, `notebooks/11-xva/README.md`

**Interfaces:**
- Consumes: la simulación de exposición de 11.1 (redefinida localmente).
- Produces: nada nuevo en qflib.

**Teoría:** panorama de los XVAs restantes, cada uno un ajuste al valor por un costo de financiamiento distinto — **FVA** (funding valuation adjustment): costo/beneficio de financiar la posición no colateralizada al spread de funding propio (no a la tasa libre de riesgo), estructuralmente análogo a CVA/DVA pero integrando el spread de funding en vez de la intensidad de default: $\text{FVA}\approx\int_0^T s_{\text{funding}}(t)\,EE^{\pm}(t)\,P(0,t)\,dt$; **ColVA**: ajuste quando el colateral remunera a una tasa distinta de la de descuento del derivado (mismo mecanismo, integrando el diferencial de tasa de colateral); **MVA/KVA** (panorama, sin implementación completa — citados y explicados conceptualmente): costo de financiar el margen inicial exigido por regulación (MVA) y el costo de capital regulatorio (KVA), ambos requieren simular el PERFIL FUTURO del propio requerimiento de margen/capital, un cálculo de "exposición sobre la exposición" que excede el alcance de un notebook introductorio.

**Demos:** (1) FVA sobre el mismo swap de 11.1, usando un spread de funding sintético, comparado en magnitud con el CVA de 11.2 (típicamente del mismo orden); (2) ColVA con un diferencial de tasa de colateral sintético (p.ej., OIS vs una tasa de colateral remunerada distinta); (3) verificar el caso límite: si el spread de funding y el diferencial de colateral son ambos cero, FVA y ColVA se anulan exactamente (identidad trivial pero necesaria); (4) explicar (sin implementar) por qué MVA/KVA requieren simular el perfil futuro de Initial Margin (que a su vez depende de un VaR de la exposición futura — anticipa M12) y de capital regulatorio, citando la referencia estándar.

**Validación:** FVA y ColVA se anulan exactamente (`atol=1e-10`) cuando sus respectivos spreads/diferenciales son cero — el único chequeo numérico duro de esta sección; el resto es exploratorio/panorama por diseño (documentado explícitamente, sin assert de pricing de MVA/KVA, mismo criterio que la sección de rBergomi en 8.5).

**Referencias:** Green, A. (2015), cap. 11-14 (FVA, ColVA, MVA, KVA). Albanese, C., Andersen, L., Iabichino, S. (2015). *FVA: a real option*. Risk.

- [ ] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 11.3 (FVA, ColVA, MVA/KVA)`

---

### Task 7: Notebook 11.4 — `11.4-netting-collateral-csa.ipynb`

**Files:**
- Create: `notebooks/11-xva/11.4-netting-collateral-csa.ipynb`
- Modify: `README.md`, `notebooks/11-xva/README.md`

**Interfaces:**
- Consumes: la simulación de exposición de 11.1 (redefinida localmente, ahora sobre un portafolio de 3-4 derivados).
- Produces: nada nuevo en qflib.

**Teoría:** cierra M11. Un **netting set** agrupa varios derivados bajo el mismo acuerdo maestro (ISDA) de forma que, en caso de default, se liquida el NETO (no cada trade por separado) — reduce la exposición agregada (ya demostrado parcialmente en 11.1 con 2 swaps). Un **CSA** (Credit Support Annex) especifica el colateral que se debe postear cuando la exposición neta excede un umbral (**threshold**) — con **variation margin** (VM) posteado periódicamente para cubrir el valor mark-to-market corriente, la exposición residual (la que de verdad importa para CVA) se reduce al *margin period of risk* (MPOR, el tiempo entre el último re-margining y el default efectivo, típicamente 10 días hábiles) en vez de a todo el horizonte del trade.

**Demos:** (1) portafolio de 3-4 derivados bajo el mismo netting set, exposición neta vs suma de exposiciones individuales sin netting (generaliza la Demo 4 de 11.1 a más de 2 trades); (2) simular un CSA con threshold $H>0$: la exposición "cubierta por colateral" en cada fecha es $\max(V_t-C_t,0)$ donde $C_t$ es el colateral posteado (igual al exceso de $V_t$ sobre $H$ en la última fecha de margining, sin actualizar hasta el MPOR) — mostrar cómo un threshold alto deja más exposición sin cubrir; (3) efecto del MPOR: exposición efectiva post-colateral con MPOR de 0 días (colateral instantáneo, exposición residual solo del salto discreto MPOR=0) vs 10 días (la volatilidad de tasas en esos 10 días queda descubierta) — CVA recalculado con exposición post-colateral es sustancialmente menor que sin CSA; (4) caso límite: threshold=0 y MPOR=0 (colateral perfecto e instantáneo) reduce la exposición a (casi) cero en todo momento.

**Validación:** exposición neteada del portafolio de 3-4 trades $\le$ suma de exposiciones individuales en cada fecha (mismo argumento estructural que 11.1, generalizado); exposición post-CSA con threshold=0 y MPOR=0 es menor a un `atol` pequeño y explícitamente documentado (no exactamente cero por la discretización temporal de la simulación, se cuantifica el residual y se explica su origen); CVA post-CSA $<$ CVA sin CSA (chequeo direccional).

**Referencias:** Gregory, J. (2015), cap. 9-10 (netting y colateral), cap. 6 (CSA). ISDA (2013). *Standard Credit Support Annex*.

- [ ] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 11.4 (netting, colateral y CSA)`

---

### Task 8: Notebook 12.1 — `12.1-var-es.ipynb`

**Files:**
- Create: `notebooks/12-riesgo-mercado/12.1-var-es.ipynb`
- Modify: `README.md`, `notebooks/12-riesgo-mercado/README.md`

**Interfaces:**
- Consumes: `qflib.mc.gbm_paths`/`ou_paths` (M1, para el portafolio sintético), `qflib.black.bs_price`/`bs_delta` (M3, para el portafolio de opciones).
- Produces: nada nuevo en qflib.

**Teoría:** **VaR** (Value at Risk) al nivel $\alpha$ sobre un horizonte $h$: la pérdida tal que $P(\text{pérdida}>VaR_\alpha)=1-\alpha$ — el cuantil $(1-\alpha)$ de la distribución de pérdidas. **ES** (Expected Shortfall, o CVaR) $=E[\text{pérdida}\mid\text{pérdida}>VaR_\alpha]$ — coherente (subaditivo) a diferencia de VaR, que no lo es en general. Tres métodos: **paramétrico** (asume retornos normales, $VaR=-\mu h+\sigma\sqrt{h}\,z_\alpha$, fórmula cerrada para ES normal también cerrada vía la densidad normal truncada); **histórico** (percentil empírico de una serie de retornos históricos simulados, sin asumir distribución); **Monte Carlo** (simular el portafolio bajo su(s) modelo(s) de riesgo, aquí un portafolio con opciones vía `bs_price`, cuya distribución de P&L no es normal por la convexidad/gamma — el caso donde el método paramétrico falla y MC es necesario).

**Demos:** (1) portafolio lineal simple (solo delta, sin opciones): VaR paramétrico y VaR histórico (retornos simulados de una GBM, con volatilidad conocida) coinciden dentro de un margen esperado (documentado); (2) portafolio con opciones (gamma no nulo): la distribución de P&L es asimétrica — VaR paramétrico (que asume normalidad) se desvía sistemáticamente del VaR de Monte Carlo (que captura la asimetría real); cuantificar la desviación y explicarla por el signo de la convexidad de la posición (larga gamma: pérdidas menos extremas de lo que predice el modelo normal en la cola; corta gamma: al revés); (3) ES vs VaR en ambos portafolios — ES siempre $\ge$ VaR al mismo nivel (por construcción, es el promedio de la cola que empieza en VaR); (4) verificar subaditividad de ES en un ejemplo de dos posiciones combinadas vs separadas (ES del portafolio combinado $\le$ suma de ES individuales) y, como contraste pedagógico, un ejemplo (de la literatura, con distribuciones fuertemente asimétricas construidas a propósito) donde VaR **falla** la subaditividad.

**Validación:** ES $\ge$ VaR al mismo nivel de confianza, en todos los portafolios probados (identidad estructural); VaR paramétrico vs histórico en el portafolio lineal, dentro de una tolerancia documentada (ambos son estimadores del mismo cuantil teórico, con ruido de muestreo finito); subaditividad de ES verificada en el ejemplo de dos posiciones (`ES_combinado <= ES_1+ES_2`); el contraejemplo de no-subaditividad de VaR se construye y se verifica explícitamente que SÍ falla (validación negativa, mismo patrón que el control de arbitraje de calendario de 8.1).

**Referencias:** Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed. Artzner, P., Delbaen, F., Eber, J.M., Heath, D. (1999). *Coherent Measures of Risk*. Mathematical Finance, 9(3), 203-228.

- [ ] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 12.1 (VaR/ES: parametrico, historico, Monte Carlo)`

---

### Task 9: Notebook 12.2 — `12.2-backtesting-kupiec-christoffersen.ipynb`

**Files:**
- Create: `notebooks/12-riesgo-mercado/12.2-backtesting-kupiec-christoffersen.ipynb`
- Modify: `README.md`, `notebooks/12-riesgo-mercado/README.md`

**Interfaces:**
- Consumes: el VaR paramétrico/histórico de 12.1 (redefinido localmente sobre una serie de P&L simulada más larga, para tener suficientes violaciones que testear).
- Produces: nada nuevo en qflib.

**Teoría:** un modelo de VaR se **backtestea** contando cuántas veces la pérdida realizada excede el VaR pronosticado ("violaciones" o "exceptions") sobre una ventana histórica, y comparando esa tasa contra la esperada bajo el nivel de confianza declarado. **Test de Kupiec** (proporción de fallas, POF): bajo $H_0$ (el modelo es correcto), el número de violaciones sigue Binomial$(n,1-\alpha)$ — test de razón de verosimilitud asintóticamente $\chi^2(1)$. **Test de Christoffersen**: extiende Kupiec chequeando además la **independencia** de las violaciones (si el modelo es correcto, las violaciones no deberían agruparse en el tiempo — clusters de violaciones indican que el modelo subestima la persistencia de la volatilidad, p.ej. en crisis) — LR de independencia basado en una cadena de Markov de 2 estados (violación/no-violación) más LR de cobertura combinados en un test conjunto, también $\chi^2(2)$.

**Demos:** (1) generar una serie larga de P&L simulado bajo un modelo CONOCIDO (GBM con volatilidad constante) y un VaR calculado con esa MISMA volatilidad (modelo correcto por construcción) — Kupiec debe NO rechazar $H_0$ en la gran mayoría de repeticiones (documentar la tasa de rechazo empírica vs el nivel nominal del test, mismo patrón de "coverage" que `test_mc_estimate_ci_coverage` de qflib); (2) modelo de VaR deliberadamente MAL calibrado (volatilidad subestimada) — Kupiec debe rechazar con alta frecuencia; (3) violaciones agrupadas a propósito (simular un régimen de volatilidad que cambia, con el VaR calculado bajo volatilidad promedio constante — las violaciones se concentran en el periodo de alta volatilidad) — Christoffersen debe rechazar por falla de independencia AUNQUE la tasa total de violaciones sea correcta en promedio (el caso donde Kupiec sólo, sin Christoffersen, no detecta el problema); (4) mismo experimento con violaciones NO agrupadas (incluso con la tasa correcta) — Christoffersen no rechaza.

**Validación:** modelo correcto (Demo 1): tasa de rechazo de Kupiec al nivel nominal del test (5%) dentro de una banda estadística sobre repeticiones (mismo criterio de `test_mc_estimate_ci_coverage`); modelo mal calibrado (Demo 2): tasa de rechazo sustancialmente mayor al nivel nominal, documentado; Demo 3 (violaciones agrupadas, tasa correcta): Christoffersen SÍ rechaza (verificado explícitamente, es el punto pedagógico central) mientras Kupiec por sí solo puede no rechazar; Demo 4 (violaciones no agrupadas): Christoffersen no rechaza.

**Referencias:** Kupiec, P. (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models*. Journal of Derivatives, 3(2), 73-84. Christoffersen, P. (1998). *Evaluating Interval Forecasts*. International Economic Review, 39(4), 841-862.

- [ ] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 12.2 (backtesting: Kupiec, Christoffersen)`

---

### Task 10: Notebook 12.3 — `12.3-portfolio-sensitivities-stress-testing.ipynb`

**Files:**
- Create: `notebooks/12-riesgo-mercado/12.3-portfolio-sensitivities-stress-testing.ipynb`
- Modify: `README.md`, `notebooks/12-riesgo-mercado/README.md`

**Interfaces:**
- Consumes: `qflib.black.bs_delta`/`bs_gamma`/`bs_vega` (M3), la DV01/KRD de 5.5 (redefinida localmente sobre una curva sintética).
- Produces: nada nuevo en qflib (cierra el currículum — no hay notebooks posteriores que reusen esto).

**Teoría:** cierre del currículum: agregación de **sensibilidades de portafolio** (Greeks de un book de opciones + DV01/KRD de un book de swaps, sumadas por bucket de riesgo — mismo principio de agregación lineal de Griegas de 3.2, ahora a nivel portafolio) y su uso en **escenarios de estrés**: en vez de asumir una distribución de retornos (como VaR paramétrico/histórico de 12.1), un escenario de estrés aplica un shock ESPECÍFICO y potencialmente NO observado en la historia reciente (p.ej. "crisis 2008": equity -40%, vol +100%, tasas -200bp) y reprecia el portafolio completo bajo ese shock — el complemento necesario de VaR/ES porque los escenarios de cola extrema real rara vez están bien representados en una ventana histórica corta.

**Demos:** (1) agregación de Greeks de un book de 5-6 opciones (distintos strikes/vencimientos) en un vector de sensibilidades netas (delta, gamma, vega totales) — verificar que la aproximación de Taylor de 2do orden (delta-gamma) del P&L ante un shock pequeño coincide con el reprecio completo (revaluar cada opción bajo el shock) dentro de un error $O(\Delta S^3)$ documentado (mismo patrón de "afirmar el orden del error" de la Fase 1); (2) DV01/KRD de un book de 3-4 swaps (buckets por tenor, reusa 5.5) agregados en un perfil de sensibilidad por bucket, y su uso para un escenario de "empinamiento de la curva" (shock no paralelo: tasas cortas bajan, largas suben) — el P&L vía KRD debe coincidir con el reprecio completo del book bajo ese shock no paralelo (donde un DV01 paralelo simple fallaría); (3) escenario de estrés histórico completo ("2008-like": shock conjunto en equity, vol, y tasas) aplicado al portafolio COMBINADO (opciones + swaps), reprecio completo vs aproximación de sensibilidades — cuantificar dónde la aproximación de 1er/2do orden empieza a fallar quantitativamente para shocks grandes (el punto pedagógico: sensibilidades lineales sirven para shocks pequeños, estrés real exige reprecio completo).

**Validación:** P&L delta-gamma vs reprecio completo (opciones) para shocks pequeños, error relativo $<1\%$ y decreciente como $\Delta S^3$ conforme el shock se achica (assert sobre el orden del error, mismo criterio que Fase 1); P&L vía KRD vs reprecio completo (swaps) bajo shock no paralelo, dentro de una tolerancia documentada (el error residual de KRD con buckets discretos vs una curva continua, cuantificado explícitamente); el estrés histórico grande muestra una divergencia sensibilidades-vs-reprecio mayor que los casos de shock pequeño, documentada y explicada (no se "arregla" con una tolerancia más ancha — se reporta como el punto pedagógico central de la Demo 3).

**Referencias:** Jorion, P. (2006), cap. 15-17 (stress testing). Hull, J. *Options, Futures, and Other Derivatives*, cap. sobre Greeks del portafolio y escenarios.

- [ ] Escribir notebook / ejecutar / READMEs / commit `feat: notebook 12.3 (sensibilidades de portafolio y stress testing)`

---

## Cierre de fase (y del currículum completo)

- [ ] Re-ejecutar **los 54 notebooks del repo completo** (M0-M12, el currículum entero) desde cero a un directorio temporal, confirmar 54/54 OK.
- [ ] `conda run -n qfcurriculum pytest tests/ -v` verde.
- [ ] Confirmar filas M10-M12 en ✅ en `README.md` y en los `README.md` de `notebooks/10-credito/`, `notebooks/11-xva/`, `notebooks/12-riesgo-mercado/`.
- [ ] Actualizar la memoria del proyecto (`~/.claude/projects/-home-claudio/memory/quant-finance-curriculum.md` y su entrada en `MEMORY.md`) marcando el currículum como **completo** (54/54), con las lecciones nuevas de esta fase.
- [ ] Push final. Esta es la última fase del roadmap (`docs/superpowers/specs/2026-07-29-quant-curriculum-design.md`) — no hay Fase 6; al cerrar, reportar al usuario que el currículum de 54 notebooks está completo.
