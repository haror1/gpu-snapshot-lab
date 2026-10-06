# gpu-snapshot-lab

**How fast can a GPU inference worker go from zero to first token—and how much of that time can we explain?**

A public lab notebook investigating GPU inference startup latency. Honest measurements, experiment by experiment, failures included.

Each experiment follows **hypothesis → method → numbers → conclusion → next question**.

- `experiments/` — scripts, environment specifications, raw results
- `notes/` — dated lab-notebook entries

## Experiment ladder

0. **Cold-start anatomy.** Measure request-to-first-token latency and instrument imports, CUDA initialization, CPU weight loading, GPU transfer, and warmup. Compare ordinary startup, CPU snapshots, and GPU snapshots on Modal. [Experiment protocol](experiments/2026-10-06-cold-start/README.md). **Status: smoke validation complete; repeated benchmark measurements pending.**
1. **Snapshot/restore survey.** Investigate direct CUDA checkpointing, CRIU, and process-based approaches where the environment permits them. Document the runtime and driver requirements of each approach, and distinguish missing privileges from technology limitations. **Status: planned.**
2. **Attack the biggest slice.** Use Experiment 0 to select one optimization. Candidate follow-ups include compiled runtime restoration or serving-engine snapshot contents. Publish before/after results. **Status: planned.**

[Snapshot boundaries and restore costs](notes/2026-10-06-snapshot-boundaries.md) sets out the next hypotheses: which initialization work is worth capturing, and which state is cheaper to recreate?

## Ground rules

- Every performance claim ships with measurements.
- Reproduce before optimizing.
- One deep experiment beats ten shallow ones.
- Unobservable time stays unassigned; failed runs stay in the dataset.
- A snapshot setting is not proof that a particular worker restored a snapshot.

## First measurements

| Startup path | Caller time to first token (ms) | Eligible samples |
| --- | ---: | ---: |
| Ordinary startup | 10,859.98 | 1 |
| CPU snapshot restore | 4,653.52 | 1 |
| GPU snapshot restore | 6,087.89 | 1 |

These smoke runs validate the harness. One metadata failure, three snapshot-creation runs, and one warm-worker reuse were excluded. Repeated measurements are needed to establish distributions and compare configurations. [Method, raw data, and exclusions](notes/2026-10-06-cold-start.md).

## Related writing

Companion writing at [himanshu-arora.com](https://himanshu-arora.com): DFlash vs DFlash 2, DSpark, Inside SGLang: DFlash Implementation, and RDMA for LLM Inference: When the Network Enters the Token Loop.
