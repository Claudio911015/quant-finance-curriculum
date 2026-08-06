#!/usr/bin/env python3
"""Descarga la serie SOFR y SOFR Averages/Index del NY Fed y guarda un
snapshot fechado en data/. Ver notebooks/13-sofr/13.2-accrual-conventions.ipynb.

Uso:  python tools/fetch_sofr_data.py [--n-days 400] [--out-dir data]
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import requests

SOFR_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/{n}.json"
SOFRAI_URL = "https://markets.newyorkfed.org/api/rates/secured/sofrai/last/{n}.json"


def fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_sofr_json(payload: dict) -> list[tuple[date, float]]:
    """[(fecha, tasa_decimal)] ascendente por fecha."""
    rows = [
        (date.fromisoformat(r["effectiveDate"]), r["percentRate"] / 100.0)
        for r in payload["refRates"]
    ]
    return sorted(rows)


def parse_sofrai_json(payload: dict) -> list[dict]:
    """[{date, average_30d, average_90d, average_180d, index}] ascendente por fecha."""
    rows = [
        {
            "date": date.fromisoformat(r["effectiveDate"]),
            "average_30d": r["average30day"] / 100.0,
            "average_90d": r["average90day"] / 100.0,
            "average_180d": r["average180day"] / 100.0,
            "index": r["index"],
        }
        for r in payload["refRates"]
    ]
    return sorted(rows, key=lambda x: x["date"])


def write_sofr_csv(rows: list[tuple[date, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sofr"])
        for d, r in rows:
            writer.writerow([d.isoformat(), r])


def write_sofrai_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "average_30d", "average_90d", "average_180d", "index"])
        for row in rows:
            writer.writerow([
                row["date"].isoformat(), row["average_30d"],
                row["average_90d"], row["average_180d"], row["index"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=400,
                         help="días hábiles a descargar (default 400, ~19 meses)")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    today = date.today().isoformat()

    sofr_rows = parse_sofr_json(fetch_json(SOFR_URL.format(n=args.n_days)))
    sofr_path = args.out_dir / f"sofr_{today}.csv"
    write_sofr_csv(sofr_rows, sofr_path)

    sofrai_rows = parse_sofrai_json(fetch_json(SOFRAI_URL.format(n=args.n_days)))
    sofrai_path = args.out_dir / f"sofrai_{today}.csv"
    write_sofrai_csv(sofrai_rows, sofrai_path)

    print(f"Escrito: {sofr_path.name} ({len(sofr_rows)} filas), "
          f"{sofrai_path.name} ({len(sofrai_rows)} filas)")


if __name__ == "__main__":
    main()
