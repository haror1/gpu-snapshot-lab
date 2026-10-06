"""Only aggregate fresh workers with independently verified restore evidence."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(rows, evidence):
    result = {}
    seen = set()
    for mode in ("none", "cpu", "gpu"):
        subset = [r for r in rows if r["mode"] == mode]
        values = []
        for row in subset:
            boot = row.get("boot_id")
            eligible = (row["status"] == "ok" and row.get("fresh_worker") is True
                        and boot not in seen and boot is not None)
            if mode != "none":
                proof = evidence.get(boot, {})
                eligible = eligible and proof.get("kind") == "restore" and bool(proof.get("source"))
            if eligible:
                values.append(row["client_ttft_ms"])
                seen.add(boot)
        values.sort()
        result[mode] = {
            "attempts": len(subset), "errors": sum(r["status"] == "error" for r in subset),
            "eligible": len(values), "excluded": len(subset) - len(values),
            "median_client_ttft_ms": statistics.median(values) if values else None,
            # Avoid implying a useful tail estimate from the smoke/pilot run.
            "p95_client_ttft_ms": values[math.ceil(.95 * len(values)) - 1] if len(values) >= 100 else None,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
    print(json.dumps(summarize(rows, evidence), indent=2))

