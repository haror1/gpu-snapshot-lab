# Experiment 0: cold-start anatomy

**State: deployed on Modal; baseline and snapshot smoke validation complete.** See the [lab entry](../../notes/2026-10-06-cold-start.md). Smoke samples validate the harness; they do not establish comparative performance.

Hypothesis: snapshotting reduces repeated initialization, but restoration costs can offset the gain for a small, uncompiled model. This baseline uses eager attention, FP16, one A10, and exactly one greedy output token. It is not a production serving-engine benchmark.

The [snapshot-boundary investigation](../../notes/2026-10-06-snapshot-boundaries.md) outlines the next hypotheses and a possible compilation experiment. Current warmup performs eager forwards, not Torch compilation or explicit CUDA graph capture. Host allocation/cache state is unobserved; a worker cold start is not a physical-machine cold boot.

## Run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
modal setup
modal deploy experiments/2026-10-06-cold-start/app.py
python experiments/2026-10-06-cold-start/run.py --samples 3 --output experiments/2026-10-06-cold-start/results/smoke.jsonl
```

Deployment builds an image containing the model: downloading is outside the measured path. The first build resolves `main` and records the exact model commit in each result. For reproduction, deploy with `MODEL_REVISION=<recorded-commit> modal deploy ...`. Save deployment logs and image/deployment identifiers alongside the results; dependencies are pinned in the remote image and actual package versions are recorded. Freeze your local environment with `uv pip freeze > experiments/2026-10-06-cold-start/results/client-requirements.txt`.

These commands use your Modal account and consume compute credits. The default smoke run makes nine serial attempts, with one container maximum per mode and a five-second idle scale-down window. A 15-second wait encourages scale-down; it does **not** prove it happened. The runner checks a post-restore UUID and per-worker request counter. Failures count toward the bounded attempt total and remain in the output. No automatic retries or scheduled jobs are configured. Image builds and failed initialization can also cost money.

After checking the smoke run, use a new output filename and `--samples 30` for a pilot. The harness randomizes mode order within each round using a recorded seed. Do not run another caller against the deployed app during measurement.

## What each number means

| Field | Interpretation |
| --- | --- |
| `client_ttft_ms` | Caller starts the RPC until its first token event arrives; includes network, routing, startup, and streaming overhead. Not HTTP endpoint TTFT. |
| `capture_stages_ms` | Imports, tokenizer/CPU weights, and (except CPU-only snapshot mode) CUDA/GPU/warmup timings from initialization at capture time. These values survive restoration and **must not be added to restore latency**. For baseline workers they describe current initialization. |
| `post_restore_stages_ms` | GPU initialization, transfer, and warmup performed after CPU snapshot restoration; empty for baseline/GPU snapshot modes. |
| `request_stages_ms` | Tokenization and transfer, prefill and greedy selection, token decoding. GPU stage boundaries synchronize CUDA. |
| `server_ttft_ms` | Method entry to decoded token availability; excludes startup and stream delivery. |
| `capture_id` / `boot_id` | Capture lineage versus a UUID generated after restoration. A repeated capture ID is supporting evidence, not proof of restore. |

Warmup performs two forwards on the fixed prompt for every configuration. The CPU snapshot contains CPU weights; the GPU snapshot additionally contains the initialized GPU model. No compiled graph or persistent KV cache is included. Warmup tensors/outputs are not retained.

Do not label `client_ttft_ms - server_ttft_ms` as container boot or snapshot restore time. It combines several unobservable components, plus initialization and untimed metadata collection. Metadata collection happens before serving in every worker and adds overhead. Capture stages also omit interpreter/platform initialization. Monotonic clock durations are measured separately; timestamps from different machines are never subtracted.

## Verify before comparing snapshots

Snapshots require deployed apps. Check Modal's Containers tab and logs for snapshot creation/restoration. Initial invocations can create snapshots for different underlying worker types; exclude those from restoration comparisons. Export the evidence promptly and map verified runs to their `boot_id` using worker logs below. Keep raw JSONL immutable.

Create `results/evidence.json`, keyed by the boot UUID, containing entries of this form (replace the placeholders):

```json
{
  "POST-RESTORE-BOOT-UUID": {
    "kind": "restore",
    "source": "saved-container-log.txt: snapshot restore line and matching worker UUID"
  }
}
```

Use `kind: "creation"` for creation runs. Missing evidence excludes snapshot runs from summaries.

```sh
python experiments/2026-10-06-cold-start/summarize.py experiments/2026-10-06-cold-start/results/smoke.jsonl --evidence experiments/2026-10-06-cold-start/results/evidence.json
python -m unittest discover -s tests
```

The summary reports excluded attempts and errors, and uses nearest-rank p95 only with at least 100 eligible samples. Inspect stage distributions and token IDs before publishing a speedup; equivalent model revisions, prompts, and output tokens are required. No inferential significance is implied by this summary.

References: [Modal memory snapshots](https://modal.com/docs/guide/memory-snapshots), [scaling configuration](https://modal.com/docs/guide/scale), [streaming RPCs](https://modal.com/docs/guide/streaming-endpoints).
