"""Serial, bounded pilot. Keeps errors and warm-worker contamination."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

CLASSES = {"none": "Baseline", "cpu": "CpuSnapshot", "gpu": "GpuSnapshot"}
PROMPT = "Explain why the sky is blue in one sentence."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3, help="Attempts per mode, including failures")
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--modes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100 or args.idle_seconds < 10:
        parser.error("samples must be 1..100; idle-seconds must be >=10")
    import modal

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clients = {mode: modal.Cls.from_name("gpu-snapshot-lab-cold-start", CLASSES[mode])()
               for mode in args.modes}
    seen_boots = set()
    rng = random.Random(args.seed)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    # Exclusive creation avoids overwriting or mixing experiments.
    with args.output.open("x") as handle:
        for sample in range(args.samples):
            order = list(clients)
            rng.shuffle(order)
            for mode in order:
                row = {
                    "schema_version": 1, "mode": mode, "sample": sample,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "prompt": PROMPT, "seed": args.seed, "idle_seconds": args.idle_seconds,
                    "client_python": platform.python_version(),
                    "client_modal": importlib.metadata.version("modal"),
                    "source_sha256": source_hashes,
                    "snapshot_verification": "not_applicable" if mode == "none" else "unverified",
                }
                started = time.perf_counter_ns()
                try:
                    events = clients[mode].first_token.remote_gen(PROMPT)
                    token = next(events)
                    row["client_ttft_ms"] = (time.perf_counter_ns() - started) / 1e6
                    if token["event"] != "token":
                        raise ValueError("Expected first token event")
                    row["token"] = token
                    metadata = next(events)
                    if metadata["event"] != "metadata":
                        raise ValueError("Expected metadata event")
                    row.update({k: v for k, v in metadata.items() if k != "event"})
                    list(events)  # Drain the RPC before the idle interval.
                    row["fresh_worker"] = row["request_index"] == 1 and row["boot_id"] not in seen_boots
                    seen_boots.add(row["boot_id"])
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(mode, sample, row["status"], row.get("client_ttft_ms"), flush=True)
                time.sleep(args.idle_seconds)


if __name__ == "__main__":
    main()

