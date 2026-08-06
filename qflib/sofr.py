"""Composición de SOFR sobre un periodo de devengo bajo las convenciones RFR
estándar (lookback, lockout, observation shift, rate cutoff).

Ver notebooks/13-sofr/13.2-accrual-conventions.ipynb para la derivación y el
entregable que compara el resultado contra el SOFR Average publicado por el
NY Fed.
"""
import bisect
from datetime import date


def _shift_business_days(dates: list[date], d: date, n: int) -> date:
    """Fecha `n` días hábiles antes (n<0) o después (n>0) de `d`, según `dates`."""
    i = bisect.bisect_left(dates, d)
    j = i + n
    if j < 0 or j >= len(dates):
        raise ValueError(
            "shift fuera del rango de `dates`; amplía la ventana de fixings"
        )
    return dates[j]


def _calendar_weights(dates: list[date], idx: list[int], window_end: date) -> list[int]:
    weights = []
    for k, i in enumerate(idx):
        nxt = dates[idx[k + 1]] if k + 1 < len(idx) else window_end
        weights.append((nxt - dates[i]).days)
    return weights


def compound_sofr(
    dates: list[date],
    rates: list[float],
    period_start: date,
    period_end: date,
    lookback: int = 0,
    lockout: int = 0,
    observation_shift: bool = False,
    rate_cutoff: int = 0,
) -> float:
    """Tasa SOFR compuesta anualizada (Act/360) sobre [period_start, period_end).

    dates: fixings de días hábiles, ascendente, con margen suficiente antes de
        period_start si lookback/observation_shift > 0.
    rates: tasas SOFR (decimal, ej. 0.0530), alineadas 1:1 con `dates`.
    lookback: días hábiles de rezago en la TASA (no en el peso). 0 = in-arrears puro.
    lockout: días hábiles al final del periodo que congelan la tasa en su valor
        `lockout` días hábiles antes del fin.
    observation_shift: si True, tasa Y peso vienen de la ventana desplazada
        `lookback` días hábiles hacia atrás, en vez de solo la tasa.
    rate_cutoff: mismo mecanismo que `lockout` (freeze de los últimos días
        hábiles), nombre distinto por convención de mercado.
    """
    if observation_shift:
        shifted_start = _shift_business_days(dates, period_start, -lookback)
        shifted_end = _shift_business_days(dates, period_end, -lookback)
        idx = [i for i, d in enumerate(dates) if shifted_start <= d < shifted_end]
        weights = _calendar_weights(dates, idx, shifted_end)
        rate_idx = idx
    else:
        idx = [i for i, d in enumerate(dates) if period_start <= d < period_end]
        weights = _calendar_weights(dates, idx, period_end)
        rate_idx = [i - lookback for i in idx]
        # Validar que los índices estén dentro del rango
        for ri in rate_idx:
            if ri < 0 or ri >= len(dates):
                raise ValueError(
                    "shift fuera del rango de `dates`; amplía la ventana de fixings"
                )

    freeze = max(lockout, rate_cutoff)
    freeze_at = max(0, len(rate_idx) - freeze)
    accrual = 1.0
    for k, ri in enumerate(rate_idx):
        use_idx = rate_idx[freeze_at] if freeze and k >= freeze_at else ri
        accrual *= 1.0 + rates[use_idx] * weights[k] / 360.0

    total_days = sum(weights)
    return (accrual - 1.0) * 360.0 / total_days
