"""Controlled eager/compiled startup comparison; deploy before invoking."""
import json
import os
import time
import uuid
from pathlib import Path

import modal

APP_NAME = "gpu-snapshot-lab-compiled-startup"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "Explain why the sky is blue in one sentence."
app = modal.App(APP_NAME)


def download_model():
    from huggingface_hub import snapshot_download

    source = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    import shutil

    shutil.copytree(source, "/model", dirs_exist_ok=True)
    Path("/model/revision.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": Path(source).name,
    }))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "huggingface-hub==0.30.2")
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_COMPILE_THREADS": "1"})
    .run_function(download_model, env={"HF_HUB_OFFLINE": "0"})
)


class Worker:
    """Shared implementation; hooks are attached separately to each variant."""
    mode = "none"
    execution = "eager"

    def measure(self, name, operation, gpu=False):
        if gpu:
            self.torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = operation()
        if gpu:
            self.torch.cuda.synchronize()
        self.active_stages[name] = (time.perf_counter_ns() - start) / 1e6
        return result

    def capture(self):
        self.capture_stages = {}
        self.active_stages = self.capture_stages
        self.capture_id = str(uuid.uuid4())
        start = time.perf_counter_ns()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.active_stages["imports_ms"] = (time.perf_counter_ns() - start) / 1e6
        self.tokenizer = self.measure("tokenizer_load_ms", lambda: AutoTokenizer.from_pretrained(
            "/model", local_files_only=True))
        self.model = self.measure("cpu_weight_load_ms", lambda: AutoModelForCausalLM.from_pretrained(
            "/model", local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="eager").eval())
        if self.mode != "cpu":
            self.prepare_gpu()

    def prepare_gpu(self):
        # Do not synchronize before init: that would initialize CUDA outside the timer.
        self.measure("cuda_init_ms", lambda: (self.torch.cuda.init(), self.torch.cuda.synchronize()))
        self.measure("gpu_weight_transfer_ms", lambda: self.model.to("cuda"), gpu=True)
        if self.execution == "compiled":
            # Default mode isolates compilation without opting into CUDA graphs.
            self.model = self.measure("compile_wrapper_ms", lambda: self.torch.compile(
                self.model, mode="default", dynamic=False))
        inputs = self.tokenizer(PROMPT, return_tensors="pt").to("cuda")

        def forward():
            with self.torch.inference_mode():
                self.model(**inputs, use_cache=False)

        self.measure("first_forward_ms", forward, gpu=True)
        self.measure("second_forward_ms", forward, gpu=True)
        self.startup_vram_bytes = int(self.torch.cuda.memory_allocated())

    def resume(self):
        # This UUID must be generated AFTER restore; captured UUIDs are reused.
        self.boot_id = str(uuid.uuid4())
        self.requests = 0
        print(json.dumps({"event": "worker_ready_hook", "boot_id": self.boot_id,
                          "capture_id": self.capture_id, "mode": self.mode,
                          "execution": self.execution}), flush=True)
        self.post_restore_stages = {}
        self.active_stages = self.post_restore_stages
        if self.mode == "cpu":
            self.prepare_gpu()
        import platform
        import subprocess

        self.environment = {
            **json.loads(Path("/model/revision.json").read_text()),
            "python": platform.python_version(), "torch": str(self.torch.__version__),
            "cuda": self.torch.version.cuda,
            "gpu": self.torch.cuda.get_device_name(),
            "driver": subprocess.check_output([
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ], text=True).strip(),
            "packages": subprocess.check_output([
                "python", "-m", "pip", "freeze"
            ], text=True).splitlines(),
        }

    def token(self, prompt):
        self.requests += 1
        stages = {}
        self.active_stages = stages
        started = time.perf_counter_ns()
        inputs = self.measure("tokenize_and_transfer_ms", lambda:
                              self.tokenizer(prompt, return_tensors="pt").to("cuda"), gpu=True)

        def forward():
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                return int(output.logits[0, -1].argmax().item())

        token_id = self.measure("prefill_and_select_ms", forward, gpu=True)
        text = self.measure("decode_ms", lambda: self.tokenizer.decode([token_id]))
        server_ttft_ms = (time.perf_counter_ns() - started) / 1e6
        yield {"event": "token", "token_id": token_id, "text": text}
        yield {
            "event": "metadata", "mode": self.mode, "execution": self.execution,
            "boot_id": self.boot_id, "capture_id": self.capture_id,
            "request_index": self.requests,
            "capture_stages_ms": self.capture_stages,
            "post_restore_stages_ms": self.post_restore_stages,
            "request_stages_ms": stages, "server_ttft_ms": server_ttft_ms,
            "environment": self.environment,
            "startup_vram_bytes": self.startup_vram_bytes,
        }


def make_worker(name, mode, execution="eager"):
    # Explicitly attach decorated hooks to each concrete class.
    def capture(self):
        Worker.capture(self)

    def resume(self):
        Worker.resume(self)

    def first_token(self, prompt):
        yield from Worker.token(self, prompt)

    hooks = {"mode": mode, "execution": execution,
             "first_token": modal.method(is_generator=True)(first_token)}
    if mode == "none":
        def initialize(self):
            Worker.capture(self)
            Worker.resume(self)
        hooks["initialize"] = modal.enter()(initialize)
    else:
        hooks["initialize"] = modal.enter(snap=True)(capture)
        hooks["after_restore"] = modal.enter(snap=False)(resume)
    cls = type(name, (Worker,), hooks)
    return app.cls(
        image=image, gpu="A10", cpu=2, memory=8192,
        min_containers=0, max_containers=1, buffer_containers=0,
        scaledown_window=5, timeout=300, startup_timeout=300, retries=0,
        enable_memory_snapshot=mode != "none",
        experimental_options={"enable_gpu_snapshot": True} if mode == "gpu" else {},
    )(cls)


Baseline = make_worker("Baseline", "none")
CpuSnapshot = make_worker("CpuSnapshot", "cpu")
GpuSnapshot = make_worker("GpuSnapshot", "gpu")
CompiledBaseline = make_worker("CompiledBaseline", "none", "compiled")
CompiledCpuSnapshot = make_worker("CompiledCpuSnapshot", "cpu", "compiled")
CompiledGpuSnapshot = make_worker("CompiledGpuSnapshot", "gpu", "compiled")
