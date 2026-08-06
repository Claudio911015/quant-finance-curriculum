from datetime import date

import pytest

from tools.fetch_sofr_data import parse_sofr_json, parse_sofrai_json

SOFR_FIXTURE = {
    "refRates": [
        {"effectiveDate": "2026-08-04", "type": "SOFR", "percentRate": 3.66,
         "percentPercentile1": 3.60, "percentPercentile25": 3.64,
         "percentPercentile75": 3.71, "percentPercentile99": 3.74,
         "volumeInBillions": 3036, "revisionIndicator": ""},
        {"effectiveDate": "2026-08-03", "type": "SOFR", "percentRate": 3.65,
         "percentPercentile1": 3.61, "percentPercentile25": 3.63,
         "percentPercentile75": 3.70, "percentPercentile99": 3.73,
         "volumeInBillions": 3055, "revisionIndicator": ""},
    ]
}

SOFRAI_FIXTURE = {
    "refRates": [
        {"effectiveDate": "2026-08-05", "type": "SOFRAI", "average30day": 3.62213,
         "average90day": 3.62708, "average180day": 3.66293, "index": 1.25363504,
         "revisionIndicator": ""},
        {"effectiveDate": "2026-08-04", "type": "SOFRAI", "average30day": 3.62146,
         "average90day": 3.62652, "average180day": 3.66288, "index": 1.25350760,
         "revisionIndicator": ""},
    ]
}


def test_parse_sofr_json_convierte_a_decimal_y_ordena_ascendente():
    rows = parse_sofr_json(SOFR_FIXTURE)
    assert rows == [
        (date(2026, 8, 3), 0.0365),
        (date(2026, 8, 4), 0.0366),
    ]


def test_parse_sofrai_json_convierte_a_decimal_y_ordena_ascendente():
    rows = parse_sofrai_json(SOFRAI_FIXTURE)
    assert rows[0]["date"] == date(2026, 8, 4)
    assert rows[0]["average_90d"] == pytest.approx(0.0362652)
    assert rows[1]["date"] == date(2026, 8, 5)
    assert rows[1]["index"] == 1.25363504
