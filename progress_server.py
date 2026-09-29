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
UTILITY_RUNS = {
    "utility · init 42": RD_ROOT / "geocrd_hybrid_utility_seed42",
    "utility · init 43": RD_ROOT / "geocrd_hybrid_utility_init43",
}
FORMAL_UTILITY_RUN = RD_ROOT / "geocrd_hybrid_utility_full_seed42"
FORMAL_HISTORY = RD_ROOT / "geocrd_hybrid_utility_full_seed42_convergence.json"
TEMPORAL_FIXED_RUN = RD_ROOT / "geocrd_residual_no_rate_audio_specificity_seed42"
TEMPORAL_FIXED_EXPECTED = 240
QWEN_ATTRIBUTION_ROOT = WEB_ROOT.parent / "experiments" / "scale_modality_probe" / "runs" / "qwen_audio_attribution"
QWEN_ATTRIBUTION_EXPECTED = 3000
RATEACTIVE_RUN = RD_ROOT / "geocrd_residual_rate003_full_utility5_seed42"
VISUAL_ANCHOR_RUN = RD_ROOT / "geocrd_visual_anchor_audio_seed42"
VISUAL_ANCHOR_DIFFERENCE_RUN = RD_ROOT / "geocrd_visual_anchor_audio_difference_seed42"
VISUAL_ANCHOR_DECOMPOSED_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_seed42"
VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_source_cross_r0_seed42"
VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED43_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_source_cross_r0_seed43"
VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED44_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_source_cross_r0_seed44"
VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED45_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_source_cross_r0_seed45"
VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED46_RUN = RD_ROOT / "geocrd_visual_anchor_audio_decomposed_source_cross_r0_seed46"
VISUAL_ANCHOR_FIVE_SEED_ANALYSIS = EXP_ROOT / "analysis" / "source_cross_r0_five_seed_hierarchical_bootstrap.json"
VISUAL_ANCHOR_HEAD_RUN = RD_ROOT / "geocrd_visual_anchor_head_seed42"
ENCODER_ATTRIBUTION_ROOT = QWEN_ATTRIBUTION_ROOT.parent / "encoder_attribution"
RATEACTIVE_CONDITIONS = (
    ("clean", "Clean", "clean"),
    ("vision_lowres_24.1062", "Low-resolution", "vision:lowres:24.1062"),
    ("vision_blur_4.65623", "Blur", "vision:blur:4.65623"),
    ("vision_dark_0.0676037", "Dark", "vision:dark:0.0676037"),
    ("vision_occlusion_0.35", "Occlusion", "vision:occlusion:0.35"),
)
UTILITY_CONDITIONS = (
    "clean",
    "vision_lowres_24.1062",
    "vision_blur_4.65623",
    "vision_dark_0.0676037",
    "vision_occlusion_0.35",
)


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


def mean_task_metric(rows, key):
    values = [
        float(row["task_metrics"][key])
        for row in rows
        if (row.get("task_metrics") or {}).get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def line_count(path: Path):
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def process_state():
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
            executable = (entry / "exe").resolve().name
        except (OSError, UnicodeDecodeError):
            continue
        if not executable.startswith("python"):
            continue
        if "extract_qwen_audio_attribution.py" in cmd:
            matches.append({"pid": int(entry.name), "command": cmd.strip()})
            continue
        if (
            "geocrd_v2_full" in cmd
            or "geocrd_ablation_e1_" in cmd
            or "geocrd_hybrid_utility_" in cmd
            or "geocrd_rateactive_" in cmd
            or RATEACTIVE_RUN.name in cmd
            or "temporal_fixed240_" in cmd
            or "geocrd_query_residual" in cmd
            or "geocrd_visual_anchor_audio" in cmd
            or "geocrd_visual_anchor_head" in cmd
        ) and (
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
    variants = ("full", "deterministic_router", "no_conditioning", "no_rate")
    roots = {
        "full": RD_ROOT / "geocrd_v2_full",
        "deterministic_router": RD_ROOT / "geocrd_ablation_e1_deterministic_router",
        "no_conditioning": RD_ROOT / "geocrd_ablation_e1_no_conditioning",
        "no_rate": RD_ROOT / "geocrd_ablation_e1_no_rate",
    }
    command_text = " ".join(item["command"] for item in processes)
    history = read_json(RD_ROOT / "geocrd_mainline_convergence.json", {}) or {}
    rows = []
    total_first_steps = 0
    completed_first_steps = 0
    current = None
    current_phase = None
    current_condition = None
    for variant in variants:
        root = roots[variant]
        config = read_json(root / "run_config.json", {}) or {}
        train_samples = int(config.get("train_samples", 62790) or 62790)
        batch_size = int(config.get("batch_size", 4) or 4)
        per_epoch = math.ceil(train_samples / batch_size)
        logs = tail_jsonl(root / "train.jsonl", 300)
        step = int(logs[-1].get("global_step", 0)) if logs else 0
        variant_commands = [item["command"] for item in processes if root.name in item["command"]]
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


def utility_progress(processes):
    rows = []
    current = None
    current_phase = None
    current_condition = None
    for label, root in UTILITY_RUNS.items():
        config = read_json(root / "run_config.json", {}) or {}
        logs = tail_jsonl(root / "train.jsonl", 300)
        step = int(logs[-1].get("global_step", 0)) if logs else 0
        expected = int(config.get("max_steps", 1500) or 1500)
        commands = [item["command"] for item in processes if root.name in item["command"]]
        phase = None
        condition = None
        if commands:
            joined = " ".join(commands)
            phase = "验证" if "evaluate_geocrd_v2_classification.py" in joined else "训练"
            if phase == "验证":
                for candidate in UTILITY_CONDITIONS:
                    value = candidate.replace("vision_", "vision:", 1).replace("_", ":", 1)
                    if f"--condition {value}" in joined:
                        condition = candidate
                        break
            current, current_phase, current_condition = label, phase, condition
        results = []
        for condition_name in UTILITY_CONDITIONS:
            summary = read_json(root / f"screen512_{condition_name}" / "summary.json", {}) or {}
            gain = (summary.get("conditional_gains", {}).get("A_given_V", {}) or {}).get("mean")
            if gain is not None:
                results.append({"condition": condition_name, "gain": float(gain)})
        degraded = [item["gain"] for item in results if item["condition"] != "clean"]
        active_records = 0
        active_expected = 1024
        if condition and not (root / f"screen512_{condition}" / "summary.json").exists():
            active_dir = root / f"screen512_{condition}"
            active_records = sum(line_count(active_dir / f"{coalition}.jsonl") for coalition in ("V", "VA"))
        rows.append({
            "label": label,
            "run": root.name,
            "step": step,
            "expected": expected,
            "percent": 100 * min(step, expected) / expected if expected else 0,
            "trained": (root / "latest.pt").exists() and step >= expected,
            "running": bool(commands),
            "phase": phase,
            "condition": condition,
            "evaluated": len(results),
            "expected_evaluations": len(UTILITY_CONDITIONS),
            "active_records": active_records,
            "active_expected": active_expected,
            "active_percent": 100 * min(active_records, active_expected) / active_expected if condition else None,
            "degraded_mean_gain": sum(degraded) / len(degraded) if degraded else None,
            "results": results,
            "utility_weight": config.get("utility_weight"),
            "utility_margin": config.get("utility_margin"),
            "init_seed": config.get("init_seed"),
            "data_seed": config.get("data_seed"),
            "corruption_seed": config.get("corruption_seed"),
        })
    return {
        "runs": rows,
        "current": current,
        "current_phase": current_phase,
        "current_condition": current_condition,
        "complete": all(row["evaluated"] == len(UTILITY_CONDITIONS) for row in rows),
        "active_or_started": any(row["step"] or row["evaluated"] for row in rows),
    }



def rateactive_progress(processes):
    root = RATEACTIVE_RUN
    config = read_json(root / "run_config.json", {}) or {}
    logs = tail_jsonl(root / "train.jsonl", 300)
    latest = logs[-1] if logs else {}
    expected_steps = int(config.get("max_steps", 0) or 0)
    if expected_steps <= 0:
        train_samples = int(config.get("train_samples", 0) or 0)
        global_batch = int(config.get("global_batch_size", config.get("batch_size", 1)) or 1)
        epochs = int(config.get("epochs", 1) or 1)
        expected_steps = ((train_samples + global_batch - 1) // global_batch) * epochs
    expected_steps = max(1, expected_steps)
    step = int(latest.get("global_step", 0) or 0)
    commands = [item["command"] for item in processes if root.name in item["command"]]
    command_text = " ".join(commands)
    current_condition = None
    for key, _label, cli in RATEACTIVE_CONDITIONS:
        if f"--condition {cli}" in command_text:
            current_condition = key
            break
    conditions = []
    evaluated = 0
    for key, label, cli in RATEACTIVE_CONDITIONS:
        output = root / f"screen1024_{key}"
        summary = read_json(output / "summary.json", {}) or {}
        done = bool(summary)
        evaluated += int(done)
        processed = sum(line_count(output / f"{coalition}.jsonl") for coalition in ("V", "VA", "VT", "VAT"))
        gains = summary.get("conditional_gains", {}) if done else {}
        conditions.append({
            "key": key, "label": label, "cli": cli,
            "processed": processed, "expected": 4096,
            "percent": 100 * min(processed, 4096) / 4096,
            "done": done,
            "a_given_v": (gains.get("A_given_V") or {}).get("mean"),
            "a_given_vt": (gains.get("A_given_VT") or {}).get("mean"),
        })
    training_done = (root / "latest.pt").exists() and step >= expected_steps
    running = bool(commands)
    return {
        "run": root.name,
        "training_step": step, "training_expected": expected_steps,
        "training_percent": 100 * min(step, expected_steps) / expected_steps,
        "training_done": training_done, "running": running,
        "phase": "评估" if "evaluate_geocrd_v2_classification.py" in command_text else ("训练" if running else None),
        "current_condition": current_condition,
        "conditions": conditions, "evaluated": evaluated,
        "expected_evaluations": len(RATEACTIVE_CONDITIONS),
        "complete": training_done and evaluated == len(RATEACTIVE_CONDITIONS),
        "overall_percent": 100 * ((1 if training_done else min(step, expected_steps) / expected_steps) + sum(item["percent"] / 100 for item in conditions)) / (1 + len(RATEACTIVE_CONDITIONS)),
        "rate_budget": config.get("rate_budget"), "ema_rate": latest.get("ema_rate"),
        "beta": latest.get("beta"), "distortion": latest.get("distortion"),
        "utility_loss": latest.get("utility_loss"),
    }


def temporal_fixed_progress(processes):
    strata = []
    processed = 0
    total = 4 * 2 * TEMPORAL_FIXED_EXPECTED
    for index in range(4):
        paired_path = TEMPORAL_FIXED_RUN / f"temporal_fixed240_s{index}" / "VA.jsonl"
        wrong_path = (
            TEMPORAL_FIXED_RUN / f"temporal_fixed_wrong240_s{index}"
            / "audio_shuffled" / "VA.jsonl"
        )
        paired = min(line_count(paired_path), TEMPORAL_FIXED_EXPECTED)
        wrong = min(line_count(wrong_path), TEMPORAL_FIXED_EXPECTED)
        processed += paired + wrong
        strata.append({
            "stratum": index,
            "paired": paired,
            "wrong": wrong,
            "expected": TEMPORAL_FIXED_EXPECTED,
            "done": paired >= TEMPORAL_FIXED_EXPECTED and wrong >= TEMPORAL_FIXED_EXPECTED,
        })
    commands = [
        item["command"] for item in processes if "temporal_fixed240_" in item["command"]
    ]
    started = processed > 0 or bool(commands)
    return {
        "strata": strata,
        "processed": processed,
        "total": total,
        "percent": 100 * processed / total if total else 0,
        "running": bool(commands),
        "started": started,
        "complete": processed >= total,
    }


def qwen_attribution_progress(processes):
    shards = []
    processed = failures = 0
    for index in range(4):
        status = read_json(QWEN_ATTRIBUTION_ROOT / f"shard{index}.pt.status.json", {}) or {}
        completed = int(status.get("completed", 0) or 0)
        failed = int(status.get("failures", 0) or 0)
        processed += completed
        failures += failed
        shards.append({
            "shard": index, "completed": completed, "expected": 750,
            "failures": failed, "done": completed >= 750,
        })
    commands = [
        item["command"] for item in processes
        if "extract_qwen_audio_attribution.py" in item["command"]
    ]
    summary = read_json(ENCODER_ATTRIBUTION_ROOT / "summary.json", {}) or {}
    probe_complete = bool(summary.get("representations"))
    common = ((summary.get("protocol") or {}).get("population") or {}).get("population_size")
    started = processed > 0 or bool(commands)
    return {
        "shards": shards,
        "processed": processed,
        "total": QWEN_ATTRIBUTION_EXPECTED,
        "percent": 100 * processed / QWEN_ATTRIBUTION_EXPECTED,
        "failures": failures,
        "running": bool(commands),
        "started": started,
        "complete": probe_complete,
        "common_population": common,
        "summary": summary,
    }


def visual_anchor_progress(processes):
    replication_roots = (
        VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED46_RUN,
        VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED45_RUN,
        VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED44_RUN,
        VISUAL_ANCHOR_SOURCE_CROSS_R0_SEED43_RUN,
        VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN,
    )
    active_replication = next(
        (candidate for candidate in replication_roots if any(
            candidate.name in item["command"] for item in processes
        )),
        None,
    )
    source_cross_r0_active = any(
        VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN.name in item["command"] for item in processes
    )
    decomposed_active = any(
        VISUAL_ANCHOR_DECOMPOSED_RUN.name in item["command"] for item in processes
    )
    difference_active = any(
        VISUAL_ANCHOR_DIFFERENCE_RUN.name in item["command"] for item in processes
    )
    root = (
        active_replication
        if active_replication is not None
        else (
        next(
            (candidate for candidate in replication_roots if (candidate / "run_config.json").exists()),
            None,
        )
        or VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN
        )
        if any((candidate / "run_config.json").exists() for candidate in replication_roots) or active_replication is not None
        else (
            VISUAL_ANCHOR_DECOMPOSED_RUN
            if decomposed_active or (VISUAL_ANCHOR_DECOMPOSED_RUN / "run_config.json").exists()
            else (
            VISUAL_ANCHOR_DIFFERENCE_RUN
            if difference_active or (VISUAL_ANCHOR_DIFFERENCE_RUN / "run_config.json").exists()
            else VISUAL_ANCHOR_RUN
            )
        )
    )
    config = read_json(root / "run_config.json", {}) or {}
    logs = tail_jsonl(root / "train.jsonl", 300)
    latest = logs[-1] if logs else {}
    step = int(latest.get("global_step", 0) or 0)
    expected = int(config.get("max_steps", 0) or 0)
    head_config = read_json(VISUAL_ANCHOR_HEAD_RUN / "run_config.json", {}) or {}
    head_logs = tail_jsonl(VISUAL_ANCHOR_HEAD_RUN / "train.jsonl", 300)
    head_latest = head_logs[-1] if head_logs else {}
    head_step = int(head_latest.get("global_step", 0) or 0)
    head_expected = int(head_config.get("max_steps", 0) or 0)
    commands = [item["command"] for item in processes if root.name in item["command"] or VISUAL_ANCHOR_HEAD_RUN.name in item["command"]]
    command_text = " ".join(commands)
    eval_dirs = {
        "paired": root / "eval_paired_1024",
        "cyclic": root / "eval_cyclic_1024",
        "cross_r0": root / "eval_cross_r0_1024",
    }
    counts = {
        "head_v": line_count(VISUAL_ANCHOR_HEAD_RUN / "eval_v_1024" / "V.jsonl"),
        "paired_v": line_count(eval_dirs["paired"] / "V.jsonl"),
        "paired_va": line_count(eval_dirs["paired"] / "VA.jsonl"),
        "cyclic": line_count(eval_dirs["cyclic"] / "VA.jsonl"),
        "cross_r0": line_count(eval_dirs["cross_r0"] / "VA.jsonl"),
    }
    mechanism_dirs = {
        "paired": root / "mechanism_repr_paired_256",
        "cyclic": root / "mechanism_repr_cyclic_256",
        "cross_r0": root / "mechanism_repr_cross_r0_256",
    }
    mechanism_counts = {key: line_count(path / "VA.jsonl") for key, path in mechanism_dirs.items()}
    mechanism_processed = sum(min(value, 256) for value in mechanism_counts.values())
    mechanism_analysis = read_json(root / "audio_content_mechanism_256.json", {}) or {}
    analysis = read_json(root / "bridge_analysis_1024.json", {}) or {}
    multiseed_analysis = read_json(VISUAL_ANCHOR_FIVE_SEED_ANALYSIS, {}) or {}
    multiseed_gates = multiseed_analysis.get("gates", {})
    comparisons = analysis.get("comparisons", {})
    def result(name):
        row = (comparisons.get(name) or {}).get("geo_distortion_delta") or {}
        return {"mean": row.get("mean"), "ci": row.get("bootstrap_95ci")}
    def multiseed_result(name):
        row = multiseed_gates.get(name) or {}
        return {
            "mean": row.get("mean_of_seed_means"),
            "ci": row.get("hierarchical_bootstrap_95ci"),
            "positive_seeds": row.get("positive_seeds"),
        }
    if "mechanism_repr" in command_text:
        phase = "音频内容特异性机制诊断"
    elif mechanism_analysis:
        phase = (
            "机制诊断完成 · 内容特异性提升"
            if root == VISUAL_ANCHOR_DIFFERENCE_RUN
            else "机制诊断完成 · 融合层信息塌缩"
        )
    elif multiseed_gates:
        phase = "五随机种子稳定性完成"
    elif analysis:
        phase = "三道门槛完成"
    elif root.name in command_text and "evaluate_geocrd_v2_classification.py" in command_text:
        phase = "固定验证"
    elif root.name in command_text and "train_geocrd_v2_classification.py" in command_text:
        phase = (
            "同来源跨区域反事实训练"
            if root in replication_roots
            else (
                "视觉校准 + 音频创新双路径训练"
                if root == VISUAL_ANCHOR_DECOMPOSED_RUN
                else (
                    "内容差分音频修正训练"
                    if root == VISUAL_ANCHOR_DIFFERENCE_RUN
                    else "音频内容修正训练"
                )
            )
        )
    elif VISUAL_ANCHOR_HEAD_RUN.name in command_text and "evaluate_geocrd_v2_classification.py" in command_text:
        phase = "原始视觉锚点头固定验证"
    elif VISUAL_ANCHOR_HEAD_RUN.name in command_text:
        phase = "原始视觉锚点头训练"
    elif step:
        phase = "等待固定验证"
    elif head_step >= head_expected and head_expected:
        phase = "视觉锚点头完成 · 等待音频修正"
    elif head_step:
        phase = "视觉锚点头等待续训"
    else:
        phase = "等待启动"
    head_eval_active = phase == "原始视觉锚点头固定验证"
    if head_eval_active:
        validation_processed = min(counts["head_v"], 1024)
        validation_total = 1024
    else:
        validation_processed = sum(min(counts[key], 1024) for key in ("paired_v", "paired_va", "cyclic", "cross_r0"))
        validation_total = 4096
    display_step = step if step else head_step
    display_expected = expected if step or expected else head_expected
    display_latest = latest if step else head_latest
    display_logs = logs if step else head_logs
    train_fraction = min(display_step / display_expected, 1.0) if display_expected else 0.0
    validation_fraction = validation_processed / validation_total
    return {
        "run": root.name,
        "started": bool(step or head_step or commands or analysis),
        "running": bool(commands),
        "phase": phase,
        "step": display_step,
        "expected": display_expected,
        "head_step": head_step,
        "head_expected": head_expected,
        "train_percent": 100 * train_fraction,
        "counts": counts,
        "validation_processed": validation_processed,
        "validation_total": validation_total,
        "mechanism_counts": mechanism_counts,
        "mechanism_processed": mechanism_processed,
        "mechanism_total": 768,
        "mechanism": mechanism_analysis,
        "overall_percent": 100 * (train_fraction + validation_fraction) / 2,
        "latest": display_latest,
        "seconds_per_step": mean(display_logs, "seconds"),
        "seed_count": int((multiseed_analysis.get("protocol") or {}).get("seeds", 1)),
        "utility": multiseed_result("paired_utility_over_vision") if multiseed_gates else result("paired_utility_over_vision"),
        "global_specificity": multiseed_result("paired_specificity_over_global") if multiseed_gates else {"mean": None, "ci": None},
        "cyclic_specificity": multiseed_result("paired_specificity_over_cyclic") if multiseed_gates else result("paired_specificity_over_cyclic"),
        "cross_r0_specificity": multiseed_result("paired_specificity_over_cross_r0") if multiseed_gates else result("paired_specificity_over_cross_r0"),
        "complete": bool(multiseed_gates or analysis),
    }

def build_progress():
    processes = process_state()
    active_utility_root = next(
        (root for root in (*UTILITY_RUNS.values(), FORMAL_UTILITY_RUN, RATEACTIVE_RUN)
         if any(root.name in item["command"] for item in processes)),
        None,
    )
    latest_utility_root = max(
        (root for root in (*UTILITY_RUNS.values(), FORMAL_UTILITY_RUN, RATEACTIVE_RUN) if (root / "train.jsonl").exists()),
        key=lambda root: (root / "train.jsonl").stat().st_mtime,
        default=None,
    )
    display_root = active_utility_root or latest_utility_root or RUN_ROOT
    config = read_json(display_root / "run_config.json", {}) or {}
    rows = tail_jsonl(display_root / "train.jsonl", 300)
    latest = rows[-1] if rows else {}
    train_samples = int(config.get("train_samples", 0) or 0)
    batch_size = int(config.get("batch_size", 1) or 1)
    batches_per_epoch = math.ceil(train_samples / batch_size) if train_samples else 0
    target_epochs = int(config.get("epochs", 6) or 6)
    max_steps = int(config.get("max_steps", 0) or 0)
    total_batches = max_steps or batches_per_epoch * target_epochs
    if max_steps:
        batches_per_epoch = max_steps
    global_step = int(latest.get("global_step", 0) or 0)
    seconds_per_batch = mean(rows, "seconds")
    remaining_batches = max(0, total_batches - global_step)
    train_eta_seconds = remaining_batches * seconds_per_batch if seconds_per_batch else None
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
    utility = utility_progress(processes)
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
        detail = f"{ablations['current']} · {phase}{suffix} · 累计{row['epoch_equivalent']:.2f} epochs · {ablations['converged']}/{ablations['total_variants']}已收敛"

    if utility["active_or_started"]:
        run42, run43 = utility["runs"]
        phases = [
            {"name": "效用目标实现", "state": "done"},
            {"name": "init 42训练", "state": "done" if run42["trained"] else ("active" if run42["running"] else "pending")},
            {"name": "init 42五条件", "state": "done" if run42["evaluated"] == 5 else ("active" if run42["trained"] else "pending")},
            {"name": "init 43复现", "state": "done" if run43["trained"] else ("active" if run43["running"] else "pending")},
            {"name": "跨种子结论", "state": "done" if utility["complete"] else "pending"},
        ]
        if utility["current"]:
            row = next(item for item in utility["runs"] if item["label"] == utility["current"])
            stage = "条件效用跨种子复现"
            suffix = f" · {utility['current_condition']}" if utility["current_condition"] else ""
            partial = f" · 当前条件{row['active_percent']:.1f}%" if row.get("active_percent") is not None else ""
            detail = f"{utility['current']} · {utility['current_phase']}{suffix} · {row['step']}/{row['expected']}步 · {row['evaluated']}/5条件{partial}"
        elif utility["complete"]:
            stage, detail = "条件效用复现完成", "init 42/43训练与五条件评估均已完成"
        else:
            stage, detail = "条件效用验证等待", f"init 42完成{run42['evaluated']}/5条件；init 43完成{run43['evaluated']}/5条件"
    else:
        phases = [
            {"name": "RD预算筛选", "state": "done" if rd_done else "active"},
            {"name": "epoch 1–3全量训练", "state": "done" if epoch3_done else ("active" if global_step else "pending")},
            {"name": "epoch 3完整验证", "state": "done" if len(eval3) >= expected_evals else ("active" if epoch3_done else "pending")},
            {"name": "epoch 4–6续训", "state": "done" if epoch6_done else ("active" if epoch3_done and "train_geocrd" in command_text else "pending")},
            {"name": "epoch 6最终验证", "state": "done" if len(eval6) >= expected_evals else ("active" if epoch6_done else "pending")},
        ]

    formal_history = read_json(FORMAL_HISTORY, {}) or {}
    formal_commands = [item["command"] for item in processes if FORMAL_UTILITY_RUN.name in item["command"]]
    formal_checkpoint_epochs = sorted(
        int(path.stem.replace("checkpoint_epoch", ""))
        for path in FORMAL_UTILITY_RUN.glob("checkpoint_epoch*.pt")
        if path.stem.replace("checkpoint_epoch", "").isdigit()
    )
    if formal_commands:
        formal_text = " ".join(formal_commands)
        formal_phase = "固定验证集评估" if "evaluate_geocrd_v2_classification.py" in formal_text else "完整epoch训练"
        current_epoch = int(latest.get("epoch", 0)) + 1 if latest else 1
        stage = "新目标正式收敛训练"
        detail = f"epoch {current_epoch} · {formal_phase} · {global_step:,}个累计batch · 已完成{len(formal_history.get('epochs', []))}轮收敛评估"
        phases = [
            {"name": "短程跨种子验证", "state": "done"},
            {"name": "完整epoch训练", "state": "active" if formal_phase == "完整epoch训练" else "done"},
            {"name": "固定验证集评估", "state": "active" if formal_phase == "固定验证集评估" else "pending"},
            {"name": "patience收敛判断", "state": "pending"},
            {"name": "最佳检查点确定", "state": "pending"},
        ]
    elif formal_history.get("status") in {"converged", "max_epochs"}:
        stage = "新目标正式训练完成"
        detail = f"{formal_history['status']} · best epoch {formal_history.get('best_epoch')} · score {formal_history.get('best_score')}"

    rateactive = rateactive_progress(processes)
    if rateactive["training_step"] or rateactive["evaluated"]:
        phases = [
            {"name": "正式全量训练", "state": "done" if rateactive["training_done"] else ("active" if rateactive["phase"] == "训练" else "pending")},
            {"name": "全量五条件评估", "state": "done" if rateactive["complete"] else ("active" if rateactive["phase"] == "评估" else "pending")},
            {"name": "正式配置结论", "state": "done" if rateactive["complete"] else "pending"},
            {"name": "预算/结构决策", "state": "pending"},
            {"name": "正式全量重训", "state": "pending"},
        ]
        if rateactive["running"]:
            stage = "正式全量训练与评估"
            if rateactive["phase"] == "训练":
                detail = (f"训练 {rateactive['training_step']:,}/{rateactive['training_expected']:,} "
                          f"({rateactive['training_percent']:.1f}%) · EMA Rate {rateactive['ema_rate']}")
            else:
                condition = next((x["label"] for x in rateactive["conditions"] if x["key"] == rateactive["current_condition"]), "—")
                detail = f"评估 · {condition} · 已完成{rateactive['evaluated']}/5条件"
        elif rateactive["complete"]:
            stage = "正式全量评估完成"
            detail = "五条件已完成，等待Rate有效性与模态增益联合判断"
        elif rateactive["training_done"]:
            stage = "正式全量训练等待评估"
            detail = f"{rateactive['training_expected']:,}步训练完成 · 已评估{rateactive['evaluated']}/5条件"

    temporal = temporal_fixed_progress(processes)
    display_epoch = int(latest.get("epoch", 0)) + 1 if latest else 0
    display_target_epochs = target_epochs
    display_batch = int(latest.get("batch", -1)) + 1 if latest else 0
    display_batches = batches_per_epoch
    display_percent = 100 * global_step / total_batches if total_batches else 0
    display_eta = train_eta_seconds
    if temporal["started"]:
        stage = "修复后Qwen音频时间窗口审计"
        detail = " · ".join(
            f"S{x['stratum']} 正确{x['paired']}/{x['expected']} 错配{x['wrong']}/{x['expected']}"
            for x in temporal["strata"]
        )
        phases = [
            {"name": f"时间分层 S{x['stratum']}", "state": "done" if x["done"] else ("active" if x["paired"] or x["wrong"] or temporal["running"] else "pending")}
            for x in temporal["strata"]
        ] + [{"name": "编码器归因决策", "state": "done" if temporal["complete"] else "pending"}]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = temporal["processed"], temporal["total"]
        display_percent = temporal["percent"]
        display_eta = None


    attribution = qwen_attribution_progress(processes)
    if attribution["started"]:
        stage = "同协议编码器归因 · Qwen表征提取"
        detail = " · ".join(
            f"GPU{x['shard']} {x['completed']}/{x['expected']}"
            for x in attribution["shards"]
        ) + f" · 失败{attribution['failures']}"
        phases = [
            {"name": "Qwen三层表征提取", "state": "done" if attribution["complete"] else "active"},
            {"name": "统一256维映射", "state": "pending"},
            {"name": "共享任务头训练", "state": "pending"},
            {"name": "正确/错配证据检验", "state": "pending"},
            {"name": "编码器归因结论", "state": "pending"},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = attribution["processed"], attribution["total"]
        display_percent, display_eta = attribution["percent"], None
        if attribution["complete"]:
            common = attribution.get("common_population") or 0
            stage = "同协议编码器归因完成"
            detail = f"共同有效样本{common} · Qwen基座有可用配对证据 · 主要失效点转向GeoCRD融合/瓶颈路径"
            phases = [
                {"name": "Qwen三层表征提取", "state": "done"},
                {"name": "统一256维映射", "state": "done"},
                {"name": "共享任务头训练", "state": "done"},
                {"name": "正确/错配证据检验", "state": "done"},
                {"name": "编码器归因结论", "state": "done"},
            ]
            display_batch = display_batches = common
            display_percent = 100.0

    query_commands = [
        item["command"] for item in processes
        if "geocrd_query_residual" in item["command"]
    ]
    query_runs = sorted(
        RD_ROOT.glob("geocrd_query_residual*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if query_commands or query_runs:
        query_root = query_runs[0]
        query_config = read_json(query_root / "run_config.json", {}) or {}
        query_rows = tail_jsonl(query_root / "train.jsonl", 300)
        query_latest = query_rows[-1] if query_rows else {}
        query_step = int(query_latest.get("global_step", 0) or 0)
        query_total = int(query_config.get("max_steps", 0) or 0)
        stage = "最终Query残差筛选" if query_commands else "最终Query残差筛选已暂停"
        detail = (
            f"{query_root.name} · {query_step}/{query_total or '完整epoch'}步 · "
            f"batch {query_config.get('batch_size', '—')}/卡 · "
            "保留Qwen原始token，仅向最终地理状态注入条件创新"
        )
        phases = [
            {"name": "音频时长单位修复", "state": "done"},
            {"name": "最终Query加性残差", "state": "done"},
            {"name": "4卡吞吐筛选", "state": "active" if query_commands else "done"},
            {"name": "正确/错配模态门槛", "state": "pending"},
            {"name": "正式全量训练决策", "state": "pending"},
        ]
        query_eval_commands = [command for command in query_commands if "evaluate_geocrd_v2_classification.py" in command]
        if query_eval_commands:
            eval_root = query_root / "evaluation_val_selection_blur"
            eval_counts = {name: line_count(eval_root / f"{name}.jsonl") for name in ("V", "VA", "VT", "VAT")}
            query_step = sum(min(value, 1024) for value in eval_counts.values())
            query_total = 4096
            stage = "最终Query残差固定验证"
            detail = "Blur-selection · " + " · ".join(f"{name} {value}/1024" for name, value in eval_counts.items())
            phases[2] = {"name": "4卡吞吐筛选", "state": "done"}
            phases[3] = {"name": "配对模态增益门槛", "state": "active"}
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = query_step, query_total
        display_percent = 100 * query_step / query_total if query_total else 0
        display_eta = None
        latest, rows, config = query_latest, query_rows, query_config
        global_step, total_batches = query_step, query_total
        seconds_per_batch = mean(query_rows, "seconds")
    visual_anchor = visual_anchor_progress(processes)
    if visual_anchor["started"]:
        stage = "视觉锚点保持的音频内容修正"
        detail = (
            f"{visual_anchor['phase']} · 训练 {visual_anchor['step']}/{visual_anchor['expected'] or '—'} · "
            f"验证 {visual_anchor['validation_processed']}/{visual_anchor['validation_total']}"
        )
        phases = [
            {"name": "视觉头冻结", "state": "done"},
            {"name": "8窗音频内容修正", "state": "done" if visual_anchor["step"] >= visual_anchor["expected"] and visual_anchor["expected"] else ("active" if visual_anchor["running"] else "pending")},
            {"name": "正确音频效用", "state": "done" if visual_anchor["complete"] else ("active" if visual_anchor["counts"]["paired_va"] else "pending")},
            {"name": "循环错配特异性", "state": "done" if visual_anchor["complete"] else ("active" if visual_anchor["counts"]["cyclic"] else "pending")},
            {"name": "跨r0错配特异性", "state": "done" if visual_anchor["complete"] else ("active" if visual_anchor["counts"]["cross_r0"] else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch = visual_anchor["step"] if visual_anchor["validation_processed"] == 0 else visual_anchor["validation_processed"]
        display_batches = visual_anchor["expected"] if visual_anchor["validation_processed"] == 0 else visual_anchor["validation_total"]
        display_percent = visual_anchor["train_percent"] if visual_anchor["validation_processed"] == 0 else 100 * visual_anchor["validation_processed"] / visual_anchor["validation_total"]
        display_eta = (visual_anchor["expected"] - visual_anchor["step"]) * visual_anchor["seconds_per_step"] if visual_anchor.get("seconds_per_step") and visual_anchor["expected"] else None
        latest = visual_anchor["latest"] or latest
        rows = tail_jsonl(VISUAL_ANCHOR_RUN / "train.jsonl", 300)
        config = read_json(VISUAL_ANCHOR_RUN / "run_config.json", {}) or {}
        global_step, total_batches = visual_anchor["step"], visual_anchor["expected"]
        seconds_per_batch = visual_anchor.get("seconds_per_step")

    return {
        "updated_at": time.time(),
        "running": bool(processes),
        "stage": stage,
        "stage_detail": detail,
        "phases": phases,
        "training": {
            "epoch": display_epoch,
            "target_epochs": display_target_epochs,
            "batch": display_batch,
            "batches_per_epoch": display_batches,
            "global_step": global_step,
            "total_batches": total_batches,
            "percent": display_percent,
            "distortion": latest.get("distortion"),
            "normalized_nll": latest.get("normalized_nll", (latest.get("task_metrics") or {}).get("normalized_nll")),
            "normalized_energy": latest.get("normalized_energy", (latest.get("task_metrics") or {}).get("normalized_energy")),
            "rate": latest.get("total_rate"),
            "ema_rate": latest.get("ema_rate"),
            "rate_budget": config.get("rate_budget"),
            "beta": latest.get("beta"),
            "window_distortion": mean(rows, "distortion"),
            "window_nll": mean(rows, "normalized_nll") or mean_task_metric(rows, "normalized_nll"),
            "window_rate": mean(rows, "total_rate"),
            "window_utility_loss": mean(rows, "utility_loss"),
            "seconds_per_batch": seconds_per_batch,
            "train_eta_seconds": display_eta,
            "coalition": latest.get("coalition"),
            "corruption": latest.get("corruption"),
        },
        "validation": {"epoch3": eval3, "epoch6": eval6, "expected": expected_evals},
        "evidence_controls": causal,
        "ablations": ablations,
        "utility": utility,
        "rateactive": rateactive,
        "temporal_fixed": temporal,
        "qwen_attribution": attribution,
        "visual_anchor": visual_anchor,
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
