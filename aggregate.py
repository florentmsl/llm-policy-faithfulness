"""Aggregate manual labels from 04_results/**/summary.csv into per-RQ and per-model stats.

Reads experiment metadata from experiments/*.yml (experiment_id -> rq, game) and joins
against the manual `pass` labels in each summary.csv. Prints two tables:

1. Pass rate per RQ across all models.
2. Pass count per (model, RQ) cell.

Run: `python aggregate.py`
"""

import csv
from collections import defaultdict
from pathlib import Path

import yaml


def _load_experiment_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for yml_path in Path("experiments").glob("*.yml"):
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        game = str(data["defaults"]["game"]).strip()
        for row in data.get("experiments", []):
            metadata[str(row["id"]).strip()] = {
                "rq": str(row["rq"]).strip().lower(),
                "game": game,
            }
    return metadata


def _load_pass_rows(meta: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in Path("04_results").glob("*/*/summary.csv"):
        model_key = summary.parent.name
        with summary.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                exp_id = row["experiment_id"].strip()
                if exp_id not in meta:
                    continue
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "model": model_key,
                        "rq": meta[exp_id]["rq"],
                        "game": meta[exp_id]["game"],
                        "pass": row["pass"].strip().lower(),
                    }
                )
    return rows


def _print_per_rq(rows: list[dict[str, str]]) -> None:
    by_rq: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_rq[r["rq"]].append(r["pass"] == "true")
    print("=== Pass rate per RQ (all models) ===")
    print(f"{'rq':<6} {'n':>4} {'pass':>5} {'fail':>5} {'rate':>6}")
    for rq in sorted(by_rq):
        passes = sum(by_rq[rq])
        n = len(by_rq[rq])
        rate = passes / n if n else 0.0
        print(f"{rq:<6} {n:>4} {passes:>5} {n - passes:>5} {rate:>6.0%}")
    print()


def _print_model_rq_matrix(rows: list[dict[str, str]]) -> None:
    by_cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    models: set[str] = set()
    rqs: set[str] = set()
    for r in rows:
        by_cell[(r["model"], r["rq"])].append(r["pass"] == "true")
        models.add(r["model"])
        rqs.add(r["rq"])
    rqs_sorted = sorted(rqs)
    print("=== Model x RQ pass/total ===")
    header = f"{'model':<48} " + " ".join(f"{rq:>8}" for rq in rqs_sorted)
    print(header)
    for model in sorted(models):
        cells = []
        for rq in rqs_sorted:
            results = by_cell.get((model, rq), [])
            if not results:
                cells.append("    -   ")
            else:
                cells.append(f"{sum(results):>3}/{len(results):<4}")
        print(f"{model:<48} " + " ".join(cells))
    print()


def _print_failures(rows: list[dict[str, str]]) -> None:
    failures = [r for r in rows if r["pass"] != "true"]
    if not failures:
        return
    print("=== Failures (LLM unfaithful to policy) ===")
    for r in sorted(failures, key=lambda r: (r["model"], r["rq"], r["experiment_id"])):
        print(f"  [{r['model']}] {r['rq']} {r['experiment_id']}")
    print()


def main() -> None:
    meta = _load_experiment_metadata()
    rows = _load_pass_rows(meta)
    if not rows:
        print("No labeled rows found under 04_results/**/summary.csv.")
        return
    print(f"Loaded {len(rows)} labeled rows across {len({r['model'] for r in rows})} model(s).\n")
    _print_per_rq(rows)
    _print_model_rq_matrix(rows)
    _print_failures(rows)


if __name__ == "__main__":
    main()
