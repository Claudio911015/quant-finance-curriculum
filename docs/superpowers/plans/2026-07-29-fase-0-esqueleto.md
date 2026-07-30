# Fase 0 — Esqueleto del repo: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar el repo publicable: estructura completa, entorno reproducible, `qflib` mínimo con tests, README con el currículum completo, y el notebook 0.1 como plantilla de referencia para los 48 restantes.

**Architecture:** Repo de notebooks organizados por módulo (`notebooks/NN-tema/`) + paquete `qflib/` mínimo (mercado sintético y plotting) testeado con pytest. Los notebooks son autocontenidos; `qflib` solo aporta lo transversal.

**Tech Stack:** Python 3.12 (conda env `qfcurriculum`), NumPy, SciPy, pandas, matplotlib, Jupyter, QuantLib (pip), pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-quant-curriculum-design.md`

## Global Constraints

- Prosa en español; terminología técnica, código, docstrings y comentarios en inglés.
- Seeds fijos en toda simulación: `np.random.default_rng(42)`.
- Nombres de notebook: `MM.N-slug-en-ingles.ipynb` (ej. `00.1-probability-spaces.ipynb`).
- Commits firmados con `claudiocp_2@hotmail.com` (config global), **sin** Co-Authored-By.
- Notebooks se commitean con outputs ejecutados.
- Regla de dependencia qflib: un notebook solo importa de `qflib` lo que un módulo anterior ya construyó desde cero.
- Python del env: `conda run -n qfcurriculum python` / `conda run -n qfcurriculum pytest`.

---

### Task 1: Estructura del repo, entorno y README

**Files:**
- Create: `.gitignore`, `environment.yml`, `README.md`
- Create: `notebooks/00-probabilidad/` … `notebooks/12-riesgo-mercado/` (13 carpetas, cada una con `README.md` corto)

**Interfaces:**
- Produces: env conda `qfcurriculum` funcional con `import QuantLib` OK; árbol de carpetas que el resto de tareas asume.

- [ ] **Step 1: Crear `.gitignore`**

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 2: Crear `environment.yml`**

```yaml
name: qfcurriculum
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - scipy
  - pandas
  - matplotlib
  - jupyter
  - pytest
  - pip
  - pip:
      - QuantLib
```

- [ ] **Step 3: Crear el env y verificar QuantLib**

Run: `conda env create -f environment.yml && conda run -n qfcurriculum python -c "import QuantLib as ql; print(ql.__version__)"`
Expected: imprime la versión de QuantLib (>=1.30) sin errores.

- [ ] **Step 4: Crear carpetas de módulos con READMEs**

Crear las 13 carpetas `notebooks/00-probabilidad` … `notebooks/12-riesgo-mercado` (nombres exactos del spec §Estructura del repo). Cada una con un `README.md` de 3-6 líneas: título del módulo, índice de sus notebooks (del spec §Currículum) y prerequisitos (módulos anteriores). Ejemplo para `notebooks/00-probabilidad/README.md`:

```markdown
# M0 — Fundamentos de probabilidad

| Notebook | Tema |
|---|---|
| 00.1-probability-spaces.ipynb | Espacios de probabilidad, v.a., distribuciones, momentos, convergencia |
| 00.2-conditional-expectation-martingales.ipynb | Esperanza condicional y martingalas en tiempo discreto |
| 00.3-random-number-generation.ipynb | Generación de aleatorios, transformada inversa, Box-Muller, Cholesky |

**Prerequisitos:** ninguno (módulo inicial).
```

Los slugs de los 49 notebooks se derivan del spec: número `MM.N` + slug corto en inglés del tema.

- [ ] **Step 5: Crear `README.md` principal**

Contenido: título, párrafo de propósito (2-3 líneas del spec §Propósito), sección "Cómo correr" (`conda env create -f environment.yml`, `jupyter lab`), y sección "Currículum" con una tabla por módulo copiando los 49 notebooks del spec §Currículum, con columna de estado (`✅` / `—`, todos `—` salvo lo que exista). Links relativos a cada notebook (aunque aún no existan — se irán completando).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: estructura del repo, entorno conda y currículum en README"
```

---

### Task 2: `qflib/plotting.py`

**Files:**
- Create: `qflib/__init__.py`, `qflib/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Produces: `apply_style() -> None` (fija rcParams del repo); `plot_paths(t: np.ndarray, paths: np.ndarray, ax=None, **kwargs) -> matplotlib.axes.Axes` (paths shape `(n_paths, len(t))`).

- [ ] **Step 1: Test que falla**

```python
# tests/test_plotting.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qflib.plotting import apply_style, plot_paths


def test_apply_style_sets_rcparams():
    apply_style()
    assert plt.rcParams["figure.figsize"] == [9.0, 5.0]
    assert plt.rcParams["axes.grid"] is True


def test_plot_paths_returns_axes():
    t = np.linspace(0.0, 1.0, 50)
    paths = np.random.default_rng(42).standard_normal((10, 50)).cumsum(axis=1)
    ax = plot_paths(t, paths)
    assert len(ax.lines) == 10
```

- [ ] **Step 2: Verificar que falla**

Run: `conda run -n qfcurriculum pytest tests/test_plotting.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'qflib'`.

- [ ] **Step 3: Implementación mínima**

```python
# qflib/__init__.py
```

```python
# qflib/plotting.py
"""Shared matplotlib style and plotting helpers for the curriculum."""
import matplotlib.pyplot as plt
import numpy as np

_STYLE = {
    "figure.figsize": (9.0, 5.0),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "lines.linewidth": 1.4,
}


def apply_style() -> None:
    """Apply the repo-wide matplotlib style."""
    plt.rcParams.update(_STYLE)


def plot_paths(t: np.ndarray, paths: np.ndarray, ax=None, **kwargs):
    """Plot simulated paths (one line per row of `paths`) against time grid `t`."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, np.asarray(paths).T, alpha=kwargs.pop("alpha", 0.6), **kwargs)
    ax.set_xlabel("t")
    return ax
```

- [ ] **Step 4: Verificar que pasa**

Run: `conda run -n qfcurriculum pytest tests/test_plotting.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add qflib/ tests/test_plotting.py && git commit -m "feat: qflib.plotting con estilo del repo"
```

---

### Task 3: `qflib/market.py` — mercado sintético mínimo

**Files:**
- Create: `qflib/market.py`
- Test: `tests/test_market.py`

**Interfaces:**
- Produces:
  - `nelson_siegel_zero(t, beta0=0.045, beta1=-0.01, beta2=0.01, tau=2.0) -> np.ndarray` — zero rate continuo compuesto en `t` (años, escalar o array; en `t=0` devuelve el límite `beta0 + beta1`).
  - `nelson_siegel_df(t, **params) -> np.ndarray` — discount factors `exp(-z(t)*t)`, `df(0)=1`.
  - `svi_total_variance(k, a=0.03, b=0.12, rho=-0.4, m=0.0, sigma=0.25) -> np.ndarray` — total variance SVI raw en log-moneyness `k`.

- [ ] **Step 1: Tests que fallan**

```python
# tests/test_market.py
import numpy as np
import pytest
from qflib.market import nelson_siegel_zero, nelson_siegel_df, svi_total_variance


def test_ns_zero_short_end_limit():
    assert nelson_siegel_zero(0.0) == pytest.approx(0.045 - 0.01)


def test_ns_zero_long_end_limit():
    assert nelson_siegel_zero(1e6) == pytest.approx(0.045, abs=1e-4)


def test_ns_df_at_zero_is_one():
    assert nelson_siegel_df(0.0) == pytest.approx(1.0)


def test_ns_df_decreasing():
    t = np.linspace(0.0, 30.0, 121)
    df = nelson_siegel_df(t)
    assert np.all(np.diff(df) < 0)


def test_svi_positive_and_convex_wings():
    k = np.linspace(-2.0, 2.0, 201)
    w = svi_total_variance(k)
    assert np.all(w > 0)
    assert w[0] > w[100] and w[-1] > w[100]  # smile shape with rho=-0.4, m=0
```

- [ ] **Step 2: Verificar que fallan**

Run: `conda run -n qfcurriculum pytest tests/test_market.py -v`
Expected: FAIL con `ModuleNotFoundError` (no existe `qflib.market`).

- [ ] **Step 3: Implementación**

```python
# qflib/market.py
"""Synthetic market data generators (parametric curves and vol surfaces)."""
import numpy as np


def nelson_siegel_zero(t, beta0=0.045, beta1=-0.01, beta2=0.01, tau=2.0):
    """Nelson-Siegel continuously-compounded zero rate at maturity t (years)."""
    t = np.asarray(t, dtype=float)
    x = t / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.where(x > 0, (1 - np.exp(-x)) / x, 1.0)
    curvature = loading - np.exp(-x)  # -> 0 as t -> 0
    z = beta0 + beta1 * loading + beta2 * curvature
    return z if z.shape else float(z)


def nelson_siegel_df(t, **params):
    """Discount factor exp(-z(t) * t) from the Nelson-Siegel zero curve."""
    t = np.asarray(t, dtype=float)
    df = np.exp(-nelson_siegel_zero(t, **params) * t)
    return df if df.shape else float(df)


def svi_total_variance(k, a=0.03, b=0.12, rho=-0.4, m=0.0, sigma=0.25):
    """Raw SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(k, dtype=float)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    return w if w.shape else float(w)
```

Nota: en `t=0`, `loading=1` y `curvature = 1 - e^0 = 0`, así que devuelve exactamente `beta0 + beta1` (lo verifica `test_ns_zero_short_end_limit`).

- [ ] **Step 4: Verificar que pasan**

Run: `conda run -n qfcurriculum pytest tests/test_market.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add qflib/market.py tests/test_market.py && git commit -m "feat: mercado sintético mínimo (Nelson-Siegel, SVI)"
```

---

### Task 4: Notebook 0.1 — plantilla de referencia

**Files:**
- Create: `notebooks/00-probabilidad/00.1-probability-spaces.ipynb`

**Interfaces:**
- Produces: el notebook plantilla cuya estructura de secciones replican los 48 restantes.

Estructura de celdas (secuencia exacta; prosa en español, código en inglés):

1. **Markdown — título y motivación:** `# 0.1 — Espacios de probabilidad, variables aleatorias y convergencia`. Motivación (≤10 líneas): todo el pricing moderno es un cálculo de esperanzas bajo una medida; este notebook fija el lenguaje (espacio de probabilidad, v.a., convergencia) que usan los 12 módulos siguientes.
2. **Markdown — teoría 1:** espacio `(Ω, 𝓕, ℙ)`, σ-álgebra como información, v.a. como función medible, ley/distribución, CDF/PDF. Definiciones formales en LaTeX.
3. **Markdown — teoría 2:** esperanza como integral de Lebesgue, momentos, varianza, desigualdades de Markov y Chebyshev (con demostración corta de Chebyshev).
4. **Markdown — teoría 3:** modos de convergencia (a.s., en probabilidad, en Lp, en distribución) con el diagrama de implicaciones; enunciados de LLN fuerte y CLT (sin demostración, referencia a Williams).
5. **Code — setup:** imports (`numpy`, `scipy.stats`, `matplotlib`), `from qflib.plotting import apply_style`, `apply_style()`, `rng = np.random.default_rng(42)`.
6. **Code + Markdown — demo LLN:** medias acumuladas de 5 000 exponenciales(λ=2) en 20 réplicas, gráfica convergiendo a 1/λ; comentario interpretando.
7. **Code + Markdown — demo CLT:** histograma de `sqrt(n)*(X̄−μ)/σ` para n∈{2, 10, 50, 500} con muestras exponenciales vs densidad N(0,1) superpuesta (grid 2×2).
8. **Code + Markdown — demo Chebyshev:** frecuencia empírica de `|X̄−μ| ≥ kσ_X̄` vs cota `1/k²`, tabla para k∈{1,2,3,4}.
9. **Markdown + Code — validación:** aquí QuantLib no aplica; validación estadística con `scipy.stats.kstest` de las medias estandarizadas (n=500) contra `norm`: p-value > 0.05 impreso y verificado con un `assert`.
10. **Markdown — referencias:** Shreve *Stochastic Calculus for Finance I* cap. 1-2; Williams *Probability with Martingales*; Jacod & Protter *Probability Essentials*.

- [ ] **Step 1: Crear el notebook con la estructura anterior** — construirlo con `nbformat` (no editar JSON a mano):

```bash
conda run -n qfcurriculum python - <<'EOF'
import nbformat as nbf
nb = nbf.v4.new_notebook()
# ... celdas según la estructura 1-10 de arriba ...
nbf.write(nb, "notebooks/00-probabilidad/00.1-probability-spaces.ipynb")
EOF
```

(El implementador escribe el contenido real de cada celda siguiendo la estructura y los temas listados arriba; las fórmulas y demos indicadas son el contenido, no opcionales.)

- [ ] **Step 2: Ejecutarlo de inicio a fin y guardar outputs**

Run: `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace notebooks/00-probabilidad/00.1-probability-spaces.ipynb`
Expected: sin errores; el `assert` del K-S test pasa; el notebook queda con outputs.

- [ ] **Step 3: Verificación visual rápida** — abrir y confirmar que las 3 gráficas se ven (LLN, CLT 2×2, Chebyshev) y el LaTeX renderiza.

- [ ] **Step 4: Actualizar estado en READMEs** — marcar 0.1 como `✅` en `README.md` principal y en `notebooks/00-probabilidad/README.md`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/00-probabilidad/ README.md && git commit -m "feat: notebook 0.1 (plantilla de referencia del currículum)"
```

---

### Task 5: Publicar en GitHub

**Files:** ninguno nuevo.

- [ ] **Step 1: Suite completa verde**

Run: `conda run -n qfcurriculum pytest tests/ -v`
Expected: todos los tests pasan.

- [ ] **Step 2: Crear repo público y push**

```bash
gh repo create Claudio911015/quant-finance-curriculum --public --source . --push
```

Expected: repo visible en GitHub con README renderizando el currículum y el notebook 0.1 con gráficas.

---

## Plantilla de tarea por notebook (Fases 1–5)

Cada notebook de las fases siguientes es una tarea con estos steps fijos (los planes de fase solo añaden el contenido específico: secciones de teoría, demos y validación):

1. Escribir el notebook siguiendo el formato estándar del spec (motivación → teoría → implementación → ejemplos → validación → referencias), estructura de celdas detallada en el plan de fase.
2. `conda run -n qfcurriculum jupyter nbconvert --to notebook --execute --inplace <notebook>` sin errores; asserts de validación (tolerancias vs QuantLib donde aplique) dentro del propio notebook.
3. Si el notebook promueve código a `qflib/` (según la regla de dependencia): test pytest + implementación + `pytest tests/ -v` verde.
4. Actualizar estado en README principal y README del módulo.
5. Commit (un notebook = un commit).
