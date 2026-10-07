"""Export descriptive ECDFs using only verified cold samples."""
import argparse
import csv
import json
from pathlib import Path

from protocol import VARIANTS, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    rows = []
    evidence = {}
    for directory in args.directories:
        rows.extend(json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines())
        evidence.update(json.loads((directory / "evidence.json").read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    colors = {"none": "#5865b3", "cpu": "#208060", "gpu": "#c56821"}
    selected = []
    for variant in VARIANTS:
        seen = set()
        cohort = []
        for row in rows:
            if row["variant"] == variant and eligible(row, evidence) and row["boot_id"] not in seen:
                seen.add(row["boot_id"])
                cohort.append(row)
        if not cohort:
            continue
        selected.extend(cohort)
        execution, mode = variant.split("-")
        style = "--" if execution == "eager" else "-"
        cold = sorted(r["client_ttft_ms"] / 1000 for r in cohort)
        warm = sorted(r["warm_probe"]["server_ttft_ms"] for r in cohort
                      if r.get("warm_probe", {}).get("same_worker")
                      and r["warm_probe"]["token"]["token_id"] == r["token"]["token_id"])
        for axis, values in zip(axes, (cold, warm)):
            if values:
                axis.step([values[0] * .98, *values],
                          [0, *[(i + 1) / len(values) for i in range(len(values))]], where="post",
                          color=colors[mode], linestyle=style,
                          label=f"{execution} / {mode} (n={len(values)})")
    axes[0].set_title("Cold request to first token (caller)")
    axes[0].set_xlabel("Seconds, log scale")
    axes[1].set_title("Warm first-token computation (worker)")
    axes[1].set_xlabel("Milliseconds, log scale")
    for axis in axes:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axis.xaxis.set_minor_formatter(FuncFormatter(lambda value, position: ""))
        axis.set_ylabel("Fraction of eligible samples")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Eager versus compiled startup — Qwen2.5-0.5B, A10, FP16\n"
                 "Creation runs, warm reuse, and output mismatches excluded", fontsize=12)
    fig.savefig(args.output_dir / "latency-ecdf.png", dpi=180)
    fig.savefig(args.output_dir / "latency-ecdf.svg")
    plt.close(fig)
    with (args.output_dir / "eligible-samples.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "boot_id", "cold_client_ttft_ms", "cold_server_ttft_ms",
                         "warm_server_ttft_ms"])
        for row in selected:
            writer.writerow([row["variant"], row["boot_id"], row["client_ttft_ms"],
                             row["server_ttft_ms"], row.get("warm_probe", {}).get("server_ttft_ms")])


if __name__ == "__main__":
    main()
