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
