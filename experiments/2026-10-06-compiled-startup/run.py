"""Bounded serial experiment, live snapshot evidence, and warm inference probes."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import random
import time
import zipfile
from pathlib import Path

from protocol import VARIANTS, classify_logs, eligible

APP_NAME = "gpu-snapshot-lab-compiled-startup"
PROMPT = "Explain why the sky is blue in one sentence."
EXPECTED_TOKEN = 576


def invoke(function):
    start = time.perf_counter_ns()
    stream = function.remote_gen(PROMPT)
    token = next(stream)
    ttft_ms = (time.perf_counter_ns() - start) / 1e6
    if token.get("event") != "token":
        raise ValueError("First stream event must contain the token")
    metadata = next(stream)
    if metadata.get("event") != "metadata":
        raise ValueError("Second stream event must contain metadata")
    list(stream)
    return {"token": token, "client_ttft_ms": ttft_ms,
            **{k: v for k, v in metadata.items() if k != "event"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--target", type=int, default=30, help="Verified eligible samples per variant")
    parser.add_argument("--max-attempts", type=int, default=40, help="Per variant; includes errors/exclusions")
    parser.add_argument("--max-errors", type=int, default=3, help="Stop a variant after this many failures")
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.target <= args.max_attempts <= 100 or args.max_errors < 1:
        parser.error("Require 1 <= target <= max-attempts <= 100 and max-errors >= 1")
    import modal

    args.output_dir.mkdir(parents=True, exist_ok=False)
    functions = {variant: modal.Cls.from_name(APP_NAME, VARIANTS[variant])().first_token
                 for variant in args.variants}
    rows = []
    counts = {variant: 0 for variant in functions}
    attempts = dict(counts)
    errors = dict(counts)
    seen = set()
    logs = []
    log_keys = set()
    rng = random.Random(args.seed)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob("*.py")}
    manifest = {**vars(args), "output_dir": str(args.output_dir),
                "source_sha256": hashes, "client_modal": importlib.metadata.version("modal"),
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.output_dir / "sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in hashes:
            archive.write(Path(__file__).parent / filename, filename)
    deadline = time.monotonic() + args.max_wall_seconds

    def refresh_logs():
        # App-level logs include platform snapshot events as well as worker UUIDs.
        for entry in modal.App.lookup(APP_NAME).logs.tail(1000):
            item = {"message": entry.message, "timestamp": entry.timestamp.isoformat(),
                    "source": entry.source, "context_ids": entry.context_ids}
            key = json.dumps(item, sort_keys=True)
            if key not in log_keys:
                log_keys.add(key)
                logs.append(item)
        logs.sort(key=lambda x: x["timestamp"])
        (args.output_dir / "logs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in logs))
        evidence = classify_logs(logs)
        (args.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        for variant in counts:
            counts[variant] = sum(r["variant"] == variant and eligible(r, evidence) for r in rows)
        return evidence

    with (args.output_dir / "requests.jsonl").open("x") as output:
        while time.monotonic() < deadline:
            active = [v for v in functions if counts[v] < args.target
                      and attempts[v] < args.max_attempts and errors[v] < args.max_errors]
            if not active:
                break
            rng.shuffle(active)
            for variant in active:
                if time.monotonic() >= deadline:
                    break
                fn = functions[variant]
                row = {"variant": variant, "execution": variant.split("-")[0],
                       "mode": variant.split("-")[1], "attempt": attempts[variant],
                       "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "prompt": PROMPT, "source_sha256": hashes}
                attempts[variant] += 1
                start = time.monotonic()
                try:
                    observations = []
                    row["cooldown_observations"] = observations
                    cool_deadline = min(start + args.cooldown_timeout, deadline)
                    while True:
                        stats = fn.get_current_stats()
                        observations.append({"elapsed_s": time.monotonic() - start,
                                             "runners": stats.num_total_runners,
                                             "running_inputs": stats.num_running_inputs})
                        if stats.num_total_runners == 0 and stats.num_running_inputs == 0:
                            break
                        if time.monotonic() >= cool_deadline:
                            raise TimeoutError("Worker did not scale to zero within cooldown budget")
                        time.sleep(5)
                    row["cooldown_observations"] = observations
                    row["cooldown_seconds"] = time.monotonic() - start
                    row.update(invoke(fn))
                    row["fresh_worker"] = row["request_index"] == 1 and row["boot_id"] not in seen
                    seen.add(row["boot_id"])
                    row["correct_token"] = row["token"]["token_id"] == EXPECTED_TOKEN
                    # Immediate same-prompt probe, outside the cold TTFT interval.
                    row["warm_probe"] = invoke(fn)
                    row["warm_probe"]["same_worker"] = row["warm_probe"]["boot_id"] == row["boot_id"]
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["elapsed_seconds"] = time.monotonic() - start
                    errors[variant] += 1
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                output.flush()
                try:
                    evidence = refresh_logs()
                    verification = evidence.get(row.get("boot_id"), {}).get("kind", "unverified")
                except Exception as exc:
                    verification = "log_fetch_failed"
                    print(f"Log verification failed: {type(exc).__name__}: {exc}", flush=True)
                print(json.dumps({"variant": variant, "attempt": row["attempt"],
                                  "status": row["status"], "ttft_ms": row.get("client_ttft_ms"),
                                  "verification": verification, "eligible": counts[variant],
                                  "error": row.get("error")}), flush=True)
        # Final fetch can recover evidence that was not yet visible after a request.
        refresh_logs()
    completion = {"eligible": counts, "attempts": attempts, "errors": errors,
                  "target_met": all(c >= args.target for c in counts.values())}
    (args.output_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
