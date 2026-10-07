# Eager versus compiled startup

**Status: protocol implemented; measurements in progress.**

Hypothesis: preserving compiled runtime state changes the balance between GPU snapshot restore cost and avoided initialization. Reproduce eager startup first, then compare a compiled variant. The initial smoke entry remains unchanged.

## Controls

Same Qwen2.5-0.5B-Instruct commit `7ae557604adf67be50417f59c2c2f167def9a775`, requested A10 GPU class, FP16, eager attention, prompt, and two warmup forwards in all six configurations. `torch.compile(mode="default", dynamic=False)` is the only execution change. This experiment does not explicitly capture CUDA graphs. Compile threads are limited to one in all variants. Model download is outside timing; weights live in the immutable image.

The requested GPU class does not pin the physical GPU/driver pair. Collection has observed both NVIDIA A10 with driver 580.95.05 and NVIDIA A10G with driver 610.57.04. Analysis reports separate GPU/driver groups alongside pooled distributions. These groups may have small sample sizes; a pooled comparison cannot establish a causal compilation effect independent of platform placement.

Compilation wraps the GPU-ready model. The first forward triggers lazy compilation; the second tests an already-warmed path. CPU snapshot mode performs this after restore, GPU snapshot mode before capture. Each cold request gets an immediate same-worker warm probe. Token ID 576 from the original eager run is the correctness check; matching one token is a limited check, not full-model numerical validation.

## Run

From the repo root with the existing virtual environment:

```sh
.venv/bin/modal deploy experiments/2026-10-06-compiled-startup/app.py
.venv/bin/python experiments/2026-10-06-compiled-startup/run.py --variants eager-none eager-cpu eager-gpu --target 30 --max-attempts 40 --output-dir experiments/2026-10-06-compiled-startup/results/eager
.venv/bin/python experiments/2026-10-06-compiled-startup/run.py --variants compiled-none compiled-cpu compiled-gpu --target 1 --max-attempts 5 --output-dir experiments/2026-10-06-compiled-startup/results/compiled-smoke
```

Only expand the compiled cohort after smoke validation. Reuse of an output directory is rejected. The runner stops a variant after three errors, forty attempts, or reaching its target; the entire process has a two-hour default wall budget checked between calls. A call already in flight can exceed that boundary. Startup and method execution each have a 300-second remote timeout.

Before each timed RPC, poll runner counts for zero, with a 120-second cooldown budget. Confirm freshness again using the post-restore boot UUID and request index. Scale-to-zero polling is outside TTFT. The runner collects platform logs separately after requests and classifies creation-inclusive boots separately from restores. Missing evidence never counts as a restore. Host cache state and physical-host provisioning remain unobserved.

Snapshot log verification occurs outside the timed interval. Raw requests stay immutable; evidence may be refreshed as platform logs become visible. Log entries include timestamps and container IDs. A manual spot check of evidence remains part of publication review.

Each result directory contains a `sources.zip` archive of the collector/instrumentation scripts matching its manifest hashes. Use those versions to reproduce a historical cohort when subsequent reporting changes have updated the live scripts.

```sh
.venv/bin/python experiments/2026-10-06-compiled-startup/analyze.py experiments/2026-10-06-compiled-startup/results/eager --output experiments/2026-10-06-compiled-startup/results/eager-summary.json
```

Report median, interquartile range, min/max, attempted/eligible sample counts, and exclusions. With fewer than 100 eligible samples, p95 is omitted. Snapshot creation-stage distributions use creation runs only; repeated copies of the same captured timings are not restore measurements. These are descriptive statistics, with uncontrolled platform placement and host cache state. No significance claim is implied.

`outside_instrumented_stages_ms` subtracts current-worker initialization stages and method TTFT from caller TTFT. It includes untimed initialization/metadata collection, snapshot restoration, scheduling, network/RPC, and stream delivery as applicable. It is not a measurement of any single platform stage. The `worker_ready_hook` log is a boot-identity marker at entry to the post-restore hook; it precedes completion of CPU-mode GPU setup and is not a serving-readiness timestamp.

For plots, install `requirements-analysis.txt` in this directory and run `plot.py <result-directories> --output-dir <plot-directory>`. ECDFs use eligible cold samples; warm probes must match worker identity and token output.

References: [Torch compilation](https://pytorch.org/docs/stable/generated/torch.compile.html), [snapshot compatibility](https://modal.com/docs/guide/memory-snapshots).
