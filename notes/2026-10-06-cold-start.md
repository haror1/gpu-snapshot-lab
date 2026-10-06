# 2026-10-06 — Cold-start anatomy

**Status: baseline and snapshot smoke validation complete. Repeated benchmark measurements pending.**

## Hypothesis

Snapshots skip repeated initialization; for a small eager-mode model, their restore cost may offset that benefit.

## Method

Compare no snapshots, CPU snapshots, and GPU snapshots with one A10 and Qwen2.5-0.5B-Instruct. Cache model files in the image. Use the same prompt, FP16 eager attention, two warmup forwards, and one greedy token. Record caller TTFT and instrumented worker stages. Reject warm workers and verify restore evidence independently. [Protocol](../experiments/2026-10-06-cold-start/README.md).

## Numbers

One successful fresh baseline worker, following one failed attempt. A10, Python 3.12.10, PyTorch 2.6.0+cu124, CUDA 12.4, driver 580.95.05. Model commit: `7ae557604adf67be50417f59c2c2f167def9a775`.

| Measurement | Single successful sample (ms) |
| --- | ---: |
| Caller request to first token | 10,859.98 |
| Torch/Transformers imports | 3,432.32 |
| Tokenizer loading | 399.46 |
| CPU weight loading | 2,328.30 |
| CUDA initialization | 199.14 |
| GPU weight transfer | 126.39 |
| Two warmup forwards | 373.45 |
| Method entry to decoded token | 24.20 |

These stages are not a complete decomposition of caller latency. Platform startup, network/RPC overhead, metadata collection, and other untimed work remain unassigned. No tail latency estimate is available.

Raw data: [successful baseline](../experiments/2026-10-06-cold-start/results/baseline-smoke-fixed-2026-10-06.jsonl), [failed first attempt](../experiments/2026-10-06-cold-start/results/baseline-smoke-2026-10-06.jsonl), [deployment](../experiments/2026-10-06-cold-start/results/deployment-2026-10-06.json).

The first attempt delivered token ID 576 after 12,226.66 ms, then failed to deserialize metadata on the caller. Cause: `torch.__version__` is a `TorchVersion` object, which brings a PyTorch dependency into deserialization. Fix: serialize it as a builtin string. A regression check verifies that conversion. Keep this attempt excluded from complete timing summaries.

### Snapshot smoke checks

Six serial attempts, three per mode, all successfully returned token ID 576. Modal logs identify three creation runs and two restores of existing snapshots. One additional request reused the CPU restore worker (request index 2) despite the idle interval; exclude it from cold-start results.

| Mode | Attempts | Verified fresh restores | Creation runs | Warm reuse | Single eligible caller TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU snapshot | 3 | 1 | 1 | 1 | 4,653.52 |
| GPU snapshot | 3 | 1 | 2 | 0 | 6,087.89 |

Creation-inclusive GPU requests took 29,236.22 and 22,581.05 ms; the CPU creation request took 23,050.81 ms. They include initialization, snapshot creation, and restoration and are not restoration-only samples. The warm CPU request took 162.56 ms and is not a cold start. These small samples do not establish a ranking or speedup.

[Raw snapshot results](../experiments/2026-10-06-cold-start/results/snapshot-smoke-2026-10-06.jsonl), [saved Modal logs](../experiments/2026-10-06-cold-start/results/modal-logs-2026-10-06.txt), and [verification mapping](../experiments/2026-10-06-cold-start/results/evidence.json). Snapshot raw rows retain `unverified`; the separate mapping supplies verified provenance without modifying the original measurements.

## Conclusion

All three configurations produce the same first token and record worker stage timings. Imports are the largest instrumented baseline stage in this one sample. The warm-worker filter caught one contaminated attempt, and logs confirmed a later GPU request created a second snapshot. Repeated measurements are required before choosing an optimization. No snapshot speedup claim yet.

## Next question

Can we collect 30 eligible cold samples per configuration while excluding snapshot creation and warm reuse, and does the baseline stage ranking persist?

The next investigation is [snapshot boundaries and restore costs](2026-10-06-snapshot-boundaries.md): compiled runtime artifacts, weight-transfer costs, and serving-engine snapshot contents. First reproduce the eager baseline; those mechanisms are not measured by these smoke samples.
