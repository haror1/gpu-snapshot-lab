# 2026-10-06 — Snapshot boundaries and restore costs

**Status: hypotheses for follow-up experiments.**

## Hypothesis

A snapshot helps when the initialization it avoids costs more than restoring its state. Capturing more state may skip more work, but can also increase the bytes that must be moved. The useful boundary may differ between model weights, compiled runtime artifacts, and an empty KV cache.

The experiment should find that boundary, rather than assume the most complete snapshot is the fastest.

## Method

First collect 30 eligible cold samples per configuration for the eager baseline, with a bounded attempt budget. Keep the model revision, GPU, prompt, precision, and warmup constant. Separate snapshot creation, verified restoration, warm reuse, and failure. Increase the idle interval and continue checking worker identity: elapsed time alone did not prevent reuse in the smoke run.

Choose one follow-up from the measured bottleneck:

| Question | Controlled change | Measurements |
| --- | --- | --- |
| Is compilation worth capturing? | Compare eager and compiled execution across the existing three startup modes. Use fixed input shapes. | Compile-wrapper setup, first forward that triggers compilation, subsequent warmup, caller TTFT, and warm inference latency. Test explicit CUDA graph capture separately. |
| Which weights belong in the snapshot? | Compare loading weights before capture against loading them after restore, with the same storage source and GPU-ready result. | CPU weight loading, GPU transfer, verified-restore TTFT, and snapshot size where exposed. |
| Is an empty KV cache cheaper to recreate? | In one serving engine, retain versus discard unused KV state before capture; recreate discarded state after restore. | Allocated VRAM, available snapshot size, cache recreation time, TTFT, and output correctness. |

Use these controls when interpreting results:

- **Cache state:** a fresh worker is not an empty host cache. Keep downloads outside timing, record image identity, and label host cache state unknown. Lazy image delivery means image size alone does not establish bytes read.
- **Snapshot boundary:** CPU mode still initializes and warms the GPU after restore; GPU mode captures that work. Capture-time measurements cannot be counted as work performed by each restored worker.
- **Runtime requirements:** direct CRIU and gVisor checkpointing are separate mechanisms. CUDA checkpointing coordinates process locking, device state saved to host memory, host checkpointing, device restoration, and unlocking. Managed-platform TTFT does not isolate these phases. A direct API experiment must record driver support and permissions.

Background references for these mechanisms: [CPU checkpointing and page loading](https://modal.com/blog/mem-snapshots), [CUDA checkpointing and compiled state](https://modal.com/blog/gpu-mem-snapshots), [image delivery and fleet allocation](https://modal.com/blog/truly-serverless-gpus), and [snapshot compatibility and serving-engine state](https://modal.com/docs/guide/memory-snapshots).

## Numbers

The [smoke experiment](2026-10-06-cold-start.md) produced one eligible sample per configuration: 10,859.98 ms for ordinary startup, 4,653.52 ms for a CPU snapshot restore, and 6,087.89 ms for a GPU snapshot restore. One warm reuse and three creation-inclusive requests were excluded from the cold-restore comparison.

Those observations motivate repeated measurements; they do not establish a ranking. No compiled-model, weight-placement, or KV-cache experiment has run yet.

## Conclusion

The harness can distinguish some misleadingly fast or slow requests from eligible cold samples. The next task is to establish distributions before changing the workload. A later optimization must improve measured startup while preserving output correctness and accounting for any change in warm inference performance.

Worker cold starts also differ from provisioning physical GPUs. A future burst trace can compare scale-to-zero against a warm application pool, reporting billed resource time and cost alongside latency.

## Next question

After repeated measurements, does the largest avoidable cost come from rebuilding runtime artifacts or moving state into place?
