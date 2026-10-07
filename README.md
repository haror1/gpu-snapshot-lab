# gpu-snapshot-lab

**How fast can a GPU inference worker go from zero to first token—and how much of that time can we explain?**

A public lab notebook investigating GPU inference startup latency. Honest measurements, experiment by experiment, failures included.

Each experiment follows **hypothesis → method → numbers → conclusion → next question**.

- `experiments/` — scripts, environment specifications, raw results
- `notes/` — dated lab-notebook entries

## Experiment ladder

0. **Cold-start anatomy.** Measure request-to-first-token latency and instrument imports, CUDA initialization, CPU weight loading, GPU transfer, and warmup. Compare ordinary startup, CPU snapshots, and GPU snapshots on Modal. [Experiment protocol](experiments/2026-10-06-cold-start/README.md). **Status: 30 eligible eager samples per configuration collected.**
1. **Snapshot/restore survey.** Investigate direct CUDA checkpointing, CRIU, and process-based approaches where the environment permits them. Document the runtime and driver requirements of each approach, and distinguish missing privileges from technology limitations. **Status: planned.**
2. **Attack the biggest slice.** Use Experiment 0 to select one optimization. Candidate follow-ups include compiled runtime restoration or serving-engine snapshot contents. Publish before/after results. **Status: planned.**

[Snapshot boundaries and restore costs](notes/2026-10-06-snapshot-boundaries.md) sets out the next hypotheses: which initialization work is worth capturing, and which state is cheaper to recreate?

## Ground rules

- Every performance claim ships with measurements.
- Reproduce before optimizing.
- One deep experiment beats ten shallow ones.
- Unobservable time stays unassigned; failed runs stay in the dataset.
- A snapshot setting is not proof that a particular worker restored a snapshot.

## Repeated eager measurements

| Startup path | Median caller time to first token (ms) | Eligible samples |
| --- | ---: | ---: |
| Ordinary startup | 9,180.70 | 30 |
| CPU snapshot restore | 3,453.48 | 30 |
| GPU snapshot restore | 3,262.36 | 30 |

Qwen2.5-0.5B-Instruct, FP16, requested A10 class, image-cached weights. Snapshot creation and unverified restores are excluded; outliers remain, including a 60,059.13 ms GPU restore. Platform placement and host caches are uncontrolled. [Distributions, method, and exclusions](notes/2026-10-06-eager-vs-compiled.md). Compiled validation is in progress.

The [initial smoke entry](notes/2026-10-06-cold-start.md) preserves the earlier harness validation and its failures.
