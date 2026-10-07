"""Classification shared by the live collector and offline analysis."""
import json

VARIANTS = {
    "eager-none": "Baseline", "eager-cpu": "CpuSnapshot", "eager-gpu": "GpuSnapshot",
    "compiled-none": "CompiledBaseline", "compiled-cpu": "CompiledCpuSnapshot",
    "compiled-gpu": "CompiledGpuSnapshot",
}


def classify_logs(entries):
    containers = {}
    evidence = {}
    for index, entry in enumerate(entries):
        context = entry.get("context_ids", [])
        container = next((x for x in context if x.startswith("ta-")), None)
        if container is None:
            continue
        state = containers.setdefault(container, {"creation": False, "restore": False, "lines": []})
        message = entry["message"]
        if ("Creating GPU memory snapshot" in message or "Creating CPU memory snapshot" in message
                or "Snapshot created. Restoring Function from memory snapshot." in message):
            state["creation"] = True
            state["lines"].append(index + 1)
        if "Restoring Function from memory snapshot." in message:
            state["restore"] = True
            state["lines"].append(index + 1)
        for line in message.splitlines():
            try:
                payload = json.loads(line)
            except (ValueError, TypeError):
                continue
            if payload.get("event") != "worker_ready_hook":
                continue
            kind = "creation" if state["creation"] else "restore" if state["restore"] else "unverified"
            evidence[payload["boot_id"]] = {
                "kind": kind, "container_id": container,
                "capture_id": payload.get("capture_id"),
                "source": {"log_lines": state["lines"] + [index + 1]},
            }
    # A restore must reference a separately observed capture lineage, protecting
    # against a truncated log window hiding creation on the same request.
    captures = {proof["capture_id"] for proof in evidence.values()
                if proof["kind"] == "creation" and proof["capture_id"]}
    for proof in evidence.values():
        if proof["kind"] == "restore" and proof["capture_id"] not in captures:
            proof["kind"] = "unverified"
    return evidence


def eligible(row, evidence):
    if row.get("status") != "ok" or not row.get("fresh_worker") or not row.get("correct_token"):
        return False
    if row["mode"] == "none":
        return True
    return evidence.get(row.get("boot_id"), {}).get("kind") == "restore"
