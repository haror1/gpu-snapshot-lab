"""Descriptive distributions; snapshot capture stages are never counted as restore work."""
import argparse
import json
import math
import statistics
from pathlib import Path

from protocol import VARIANTS, eligible


def describe(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "median": statistics.median(ordered),
            "p25": ordered[math.ceil(.25 * len(ordered)) - 1],
            "p75": ordered[math.ceil(.75 * len(ordered)) - 1],
            "min": ordered[0], "max": ordered[-1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1] if len(ordered) >= 100 else None}


def analyze(directories):
    rows = []
    evidence = {}
    for directory in directories:
        rows.extend(json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines())
        evidence.update(json.loads((directory / "evidence.json").read_text()))
    result = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        seen = set()
        selected = []
        for row in subset:
            if eligible(row, evidence) and row["boot_id"] not in seen:
                selected.append(row)
                seen.add(row["boot_id"])
        warm = [r["warm_probe"] for r in selected if r.get("warm_probe", {}).get("same_worker")
                and r["warm_probe"]["token"]["token_id"] == r["token"]["token_id"]]
        creation = [r for r in subset if evidence.get(r.get("boot_id"), {}).get("kind") == "creation"]
        environment_groups = {}
        for row in selected:
            env = row["environment"]
            key = f"{env['gpu']} | driver {env['driver']}"
            environment_groups.setdefault(key, []).append(row)
        stage_rows = selected if variant.endswith("-none") else creation
        def stages(items, key):
            names = {name for r in items for name in r.get(key, {})}
            return {name: describe([r[key][name] for r in items if name in r.get(key, {})])
                    for name in sorted(names)}
        result[variant] = {
            "attempts": len(subset), "errors": sum(r["status"] == "error" for r in subset),
            "eligible": len(selected), "creation_runs": len(creation),
            "warm_reuse": sum(r.get("fresh_worker") is False for r in subset),
            "output_mismatches": sum(r.get("correct_token") is False for r in subset),
            "client_ttft_ms": describe([r["client_ttft_ms"] for r in selected]),
            "server_ttft_ms": describe([r["server_ttft_ms"] for r in selected]),
            "outside_instrumented_stages_ms": describe([
                r["client_ttft_ms"] - r["server_ttft_ms"] - sum(r[
                    "capture_stages_ms" if r["mode"] == "none" else "post_restore_stages_ms"
                ].values()) for r in selected]),
            "warm_client_ttft_ms": describe([r["client_ttft_ms"] for r in warm]),
            "warm_server_ttft_ms": describe([r["server_ttft_ms"] for r in warm]),
            "initialization_stages_ms": stages(stage_rows, "capture_stages_ms"),
            "initialization_provenance": "current_worker" if variant.endswith("-none") else "snapshot_creation_runs",
            "post_restore_stages_ms": stages(selected, "post_restore_stages_ms"),
            "errors_detail": [r.get("error") for r in subset if r["status"] == "error"],
            "by_gpu_driver": {key: {
                "client_ttft_ms": describe([r["client_ttft_ms"] for r in items]),
                "warm_server_ttft_ms": describe([r["warm_probe"]["server_ttft_ms"] for r in items
                    if r.get("warm_probe", {}).get("same_worker")
                    and r["warm_probe"]["token"]["token_id"] == r["token"]["token_id"]]),
            } for key, items in sorted(environment_groups.items())},
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(analyze(args.directories), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
