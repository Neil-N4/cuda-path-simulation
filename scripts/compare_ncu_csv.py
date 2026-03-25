#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def extract_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            # Nsight CSV raw format: ...,"Metric Name",...,"Metric Value",...
            metric = row[4].strip()
            value = row[7].strip()
            if not metric or metric == "Metric Name":
                continue
            try:
                out[metric] = float(value)
            except ValueError:
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Nsight Compute metric CSVs.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()

    b = extract_metrics(args.before)
    a = extract_metrics(args.after)
    common = sorted(set(b.keys()) & set(a.keys()))
    if not common:
        raise SystemExit("No overlapping metrics found.")

    print("metric,before,after,delta,delta_pct")
    for k in common:
        bv = b[k]
        av = a[k]
        d = av - bv
        dp = (d / bv * 100.0) if abs(bv) > 1e-12 else 0.0
        print(f"{k},{bv:.6f},{av:.6f},{d:.6f},{dp:.2f}%")


if __name__ == "__main__":
    main()
