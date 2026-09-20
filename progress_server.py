#!/usr/bin/env python3
"""Serve the experiment notebook plus a read-only live training API."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


WEB_ROOT = Path(__file__).resolve().parent
EXP_ROOT = WEB_ROOT.parent / "experiments" / "pid_omni"
RUN_ROOT = EXP_ROOT / "runs" / "geocrd_v2_full"
RD_ROOT = EXP_ROOT / "runs"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def tail_jsonl(path: Path, limit: int = 300):
    if not path.exists():
        return []
    rows = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(rows)


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def process_state():
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if ("geocrd_v2_full" in cmd or "geocrd_ablation_e1_" in cmd) and (
            "train_geocrd_v2_classification.py" in cmd
            or "evaluate_geocrd_v2_classification.py" in cmd
            or "evaluate_geocrd_v2_evidence_controls.py" in cmd
        ):
            matches.append({"pid": int(entry.name), "command": cmd.strip()})
    return sorted(matches, key=lambda item: item["pid"])


def gpu_state():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=4, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4:
            gpus.append({
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization": int(fields[3]),
            })
    return gpus


def validation_summaries(prefix: str):
    results = []
    for summary_path in sorted(RUN_ROOT.glob(f"{prefix}*/summary.json")):
        summary = read_json(summary_path, {})
        results.append({
            "name": summary_path.parent.name.replace(prefix, ""),
            "samples": summary.get("paired_samples"),
            "coalitions": {
                key: {
                    "distortion": value.get("mean_geo_distortion"),
                    "nll": value.get("mean_normalized_nll"),
                    "accuracy": {
                        scale: metrics.get("accuracy")
                        for scale, metrics in value.get("per_resolution", {}).items()
                    },
                }
                for key, value in summary.get("coalitions", {}).items()
            },
            "gains": summary.get("conditional_gains", {}),
        })
    return results


def evidence_control_progress():
    conditions = [
        ("clean", "Clean"),
        ("vision_blur_10", "Blur-10"),
        ("vision_lowres_28", "28px"),
    ]
    tasks = [
        ("audio_shuffled", "VA"),
        ("audio_shuffled", "VAT"),
        ("text_shuffled", "VT"),
        ("text_shuffled", "VAT"),
        ("both_shuffled", "VAT"),
    ]
    expected = 7550
    rows = []
    completed_tasks = 0
    processed = 0
    active = None
    for key, label in conditions:
        root = RUN_ROOT / f"evidence_controls_val_{key}"
        task_rows = []
        for control, coalition in tasks:
            path = root / control / f"{coalition}.jsonl"
            count = 0
            try:
                with path.open("rb") as handle:
                    count = sum(1 for _ in handle)
            except OSError:
                pass
            capped = min(count, expected)
            done = capped >= expected
            processed += capped
            completed_tasks += int(done)
            if active is None and not done and (count > 0 or root.exists()):
                active = {"condition": label, "control": control, "coalition": coalition, "samples": count}
            task_rows.append({"control": control, "coalition": coalition, "samples": count, "expected": expected, "done": done})
        rows.append({"key": key, "label": label, "tasks": task_rows, "completed": sum(t["done"] for t in task_rows)})
    total = len(conditions) * len(tasks) * expected
    return {
        "conditions": rows,
        "completed_tasks": completed_tasks,
        "total_tasks": len(conditions) * len(tasks),
        "processed": processed,
        "total": total,
        "percent": 100 * processed / total if total else 0,
        "active": active,
        "complete": completed_tasks == len(conditions) * len(tasks),
    }


def ablation_progress(processes):
    variants = (
        "native_head", "deterministic_router", "no_conditioning",
        "no_rate", "nll_only", "full",
    )
    command_text = " ".join(item["command"] for item in processes)
    history = read_json(RD_ROOT / "geocrd_ablation_convergence.json", {}) or {}
    rows = []
    total_first_steps = 0
    completed_first_steps = 0
    current = None
    current_phase = None
    current_condition = None
    for variant in variants:
        root = RD_ROOT / f"geocrd_ablation_e1_{variant}"
        config = read_json(root / "run_config.json", {}) or {}
        train_samples = int(config.get("train_samples", 62790) or 62790)
        batch_size = int(config.get("batch_size", 4) or 4)
        per_epoch = math.ceil(train_samples / batch_size)
        logs = tail_jsonl(root / "train.jsonl", 300)
        step = int(logs[-1].get("global_step", 0)) if logs else 0
        variant_commands = [item["command"] for item in processes if f"geocrd_ablation_e1_{variant}" in item["command"]]
        running = bool(variant_commands)
        phase = None
        condition = None
        if running:
            current = variant
            joined = " ".join(variant_commands)
            phase = "验证" if "evaluate_geocrd_v2_classification.py" in joined else "训练"
            for candidate in ("vision:lowres:28", "vision:blur:10", "clean"):
                if f"--condition {candidate}" in joined:
                    condition = candidate
                    break
            current_phase, current_condition = phase, condition
        record = history.get(variant, {})
        status = record.get("status")
        converged = status in {"converged", "max_epochs"}
        first_done = step >= per_epoch
        total_first_steps += per_epoch
        completed_first_steps += min(step, per_epoch)
        rows.append({
            "variant": variant,
            "step": step,
            "expected": per_epoch,
            "epoch_equivalent": step / per_epoch,
            "target_epoch": config.get("epochs", 1),
            "first_done": first_done,
            "running": running,
            "phase": phase,
            "condition": condition,
            "converged": converged,
            "best_epoch": record.get("best_epoch"),
            "best_score": record.get("best_score"),
            "stale": record.get("stale", 0),
            "trainable_parameters": config.get("trainable_parameters"),
        })
    return {
        "variants": rows,
        "current": current,
        "current_phase": current_phase,
        "current_condition": current_condition,
        "first_checkpoint_percent": 100 * completed_first_steps / total_first_steps if total_first_steps else 0,
        "first_completed": sum(row["first_done"] for row in rows),
        "converged": sum(row["converged"] for row in rows),
        "total_variants": len(rows),
        "eta_seconds": None,
    }


def build_progress():
    config = read_json(RUN_ROOT / "run_config.json", {}) or {}
    rows = tail_jsonl(RUN_ROOT / "train.jsonl", 300)
    latest = rows[-1] if rows else {}
    train_samples = int(config.get("train_samples", 0) or 0)
    batch_size = int(config.get("batch_size", 1) or 1)
    batches_per_epoch = math.ceil(train_samples / batch_size) if train_samples else 0
    target_epochs = 6
    total_batches = batches_per_epoch * target_epochs
    global_step = int(latest.get("global_step", 0) or 0)
    seconds_per_batch = mean(rows, "seconds")
    remaining_batches = max(0, total_batches - global_step)
    train_eta_seconds = remaining_batches * seconds_per_batch if seconds_per_batch else None
    processes = process_state()
    command_text = " ".join(item["command"] for item in processes)

    epoch3_done = (RUN_ROOT / "checkpoint_epoch3.pt").exists()
    epoch6_done = (RUN_ROOT / "checkpoint_epoch6.pt").exists()
    eval3 = [
        row for row in validation_summaries("evaluation_val_")
        if not row["name"].startswith("e6_")
    ]
    eval6 = validation_summaries("evaluation_val_e6_")
    expected_evals = 6
    rd_done = (RD_ROOT / "geocrd_v2_rd_summary.json").exists()

    if len(eval6) >= expected_evals:
        stage, detail = "全部完成", "epoch 6最终验证已完成"
    elif "evaluate_geocrd_v2_classification.py" in command_text and epoch6_done:
        stage, detail = "最终验证", f"epoch 6验证：{len(eval6)}/{expected_evals}组完成"
    elif "train_geocrd_v2_classification.py" in command_text and epoch3_done:
        stage, detail = "续训至epoch 6", f"epoch {int(latest.get('epoch', 0)) + 1}/6"
    elif len(eval3) >= expected_evals:
        stage, detail = "等待续训", "epoch 3验证完成，等待epoch 4–6"
    elif "evaluate_geocrd_v2_classification.py" in command_text and epoch3_done:
        stage, detail = "epoch 3完整验证", f"{len(eval3)}/{expected_evals}组完成"
    elif "train_geocrd_v2_classification.py" in command_text:
        stage, detail = "全量训练至epoch 3", f"epoch {int(latest.get('epoch', 0)) + 1}/3"
    elif rd_done:
        stage, detail = "等待全量训练", "RD预算筛选已完成"
    else:
        stage, detail = "RD预算筛选", "正在选择Rate预算"

    causal = evidence_control_progress()
    ablations = ablation_progress(processes)
    if causal["complete"]:
        stage, detail = "因果对照完成", "三条件×五阶段全量错配验证已完成"
    elif any("evaluate_geocrd_v2_evidence_controls.py" in item["command"] for item in processes):
        current = causal.get("active") or {}
        stage = "全量因果对照"
        detail = f"{current.get('condition', '—')} · {current.get('control', '—')} · {current.get('coalition', '—')} · {current.get('samples', 0)}/7550"

    if ablations["current"]:
        row = next(item for item in ablations["variants"] if item["variant"] == ablations["current"])
        stage = "GeoCRD架构归因消融"
        phase = ablations.get("current_phase") or "运行"
        condition = ablations.get("current_condition")
        suffix = f" · {condition}" if condition else ""
        detail = f"{ablations['current']} · {phase}{suffix} · 累计{row['epoch_equivalent']:.2f} epochs · {ablations['converged']}/6已收敛"

    phases = [
        {"name": "RD预算筛选", "state": "done" if rd_done else "active"},
        {"name": "epoch 1–3全量训练", "state": "done" if epoch3_done else ("active" if global_step else "pending")},
        {"name": "epoch 3完整验证", "state": "done" if len(eval3) >= expected_evals else ("active" if epoch3_done else "pending")},
        {"name": "epoch 4–6续训", "state": "done" if epoch6_done else ("active" if epoch3_done and "train_geocrd" in command_text else "pending")},
        {"name": "epoch 6最终验证", "state": "done" if len(eval6) >= expected_evals else ("active" if epoch6_done else "pending")},
    ]

    return {
        "updated_at": time.time(),
        "running": bool(processes),
        "stage": stage,
        "stage_detail": detail,
        "phases": phases,
        "training": {
            "epoch": int(latest.get("epoch", 0)) + 1 if latest else 0,
            "target_epochs": target_epochs,
            "batch": int(latest.get("batch", -1)) + 1 if latest else 0,
            "batches_per_epoch": batches_per_epoch,
            "global_step": global_step,
            "total_batches": total_batches,
            "percent": 100 * global_step / total_batches if total_batches else 0,
            "distortion": latest.get("distortion"),
            "normalized_nll": latest.get("normalized_nll"),
            "normalized_energy": latest.get("normalized_energy"),
            "rate": latest.get("total_rate"),
            "ema_rate": latest.get("ema_rate"),
            "rate_budget": config.get("rate_budget"),
            "beta": latest.get("beta"),
            "window_distortion": mean(rows, "distortion"),
            "window_nll": mean(rows, "normalized_nll"),
            "window_rate": mean(rows, "total_rate"),
            "seconds_per_batch": seconds_per_batch,
            "train_eta_seconds": train_eta_seconds,
            "coalition": latest.get("coalition"),
            "corruption": latest.get("corruption"),
        },
        "validation": {"epoch3": eval3, "epoch6": eval6, "expected": expected_evals},
        "evidence_controls": causal,
        "ablations": ablations,
        "gpus": gpu_state(),
        "processes": processes,
    }


class ProgressHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path == "/api/training-progress":
            payload = json.dumps(build_progress(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        if urlparse(self.path).path != "/api/training-progress":
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ProgressHandler)
    print(f"Live experiment log: http://{args.host}:{args.port}/progress.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
