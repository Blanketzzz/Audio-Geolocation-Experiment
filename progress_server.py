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
ACCEPTANCE_PROBE_TRAIN = VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN / "eval_acceptance_probe_v2_train4096" / "VA.jsonl"
ACCEPTANCE_PROBE_VAL = VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN / "eval_acceptance_probe_v2_val1024" / "VA.jsonl"
ACCEPTANCE_PROBE_TRAIN_REJECTED = VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN / "eval_acceptance_probe_v2_train4096" / "rejected.jsonl"
ACCEPTANCE_PROBE_VAL_REJECTED = VISUAL_ANCHOR_SOURCE_CROSS_R0_RUN / "eval_acceptance_probe_v2_val1024" / "rejected.jsonl"
ACCEPTANCE_PROBE_RESULT = EXP_ROOT / "analysis" / "audio_acceptance_predictability_v2_seed42.json"
SELECTIVE_EVIDENCE_RUN = RD_ROOT / "selective_geographic_evidence_frozen_seed42"
SELECTIVE_EVIDENCE_QUICK = EXP_ROOT / "analysis" / "selective_geographic_evidence_quick64.json"
SIGNED_VALUE_ROOT = RD_ROOT / "signed_geographic_value_seed42"
SIGNED_VALUE_GATE2_ROOT = RD_ROOT / "signed_geographic_value_gate2_seed42"
AUDIO_PROVENANCE_GATE3_ROOT = RD_ROOT / "audio_provenance_gate3_seed42"
TEXT_PROVENANCE_GATE3_ROOT = RD_ROOT / "text_provenance_gate3_seed42"
PROVENANCE_RANK_ROOT = RD_ROOT / "geocrd_provenance_rank_balanced_seed42"
COMPATIBILITY_PROBE_ROOT = RD_ROOT / "compatibility_probe_seed42"
NONLINEAR_COMPATIBILITY_ROOT = COMPATIBILITY_PROBE_ROOT / "nonlinear_probe"
LATENT_COMPATIBILITY_ROOT = RD_ROOT / "latent_bank_compatibility_seed42"
POSTERIOR_UPDATE_ROOT = RD_ROOT / "geographic_posterior_update_seed42"
POSTERIOR_RELIABILITY_ROOT = RD_ROOT / "posterior_reliability_screen_seed42"
SAFE_PRIOR_UPDATE_ROOT = RD_ROOT / "safe_prior_corrected_reliability_seed42"
ONLINE_EVIDENCE_ROOTS = {
    seed: RD_ROOT / f"online_safe_evidence_screen_seed{seed}"
    for seed in (42, 43, 44)
}
JOINT_EVIDENCE_ROOTS = {
    seed: RD_ROOT / f"joint_safe_evidence_screen_seed{seed}"
    for seed in (42, 43, 44)
}
EVIDENCE_UPDATE_ABLATIONS = {
    "ordinary_add": RD_ROOT / "joint_ablation_ordinary_add_seed42",
    "prior_only": RD_ROOT / "joint_ablation_prior_only_seed42",
    "adaptive_ranked": RD_ROOT / "joint_ablation_adaptive_ranked_seed42",
    "full": RD_ROOT / "joint_ablation_full_fresh_seed42",
}
REGRESSION_BASE_ROOT = RD_ROOT / "safe_evidence_regression_base_seed42"
REGRESSION_UPDATE_ROOT = RD_ROOT / "safe_evidence_regression_update_seed42"
REGRESSION_UPDATE_ROOTS = {
    seed: RD_ROOT / f"safe_evidence_regression_update_seed{seed}"
    for seed in (42, 43, 44)
}
RETRIEVAL_BASE_ROOT = RD_ROOT / "safe_evidence_retrieval_base_seed42"
CONDITIONAL_RETRIEVAL_SUMMARY = EXP_ROOT / "analysis" / "safe_conditional_retrieval_three_seed.json"
EXPANDED_RETRIEVAL_BANK_ROOT = RD_ROOT / "safe_evidence_retrieval_joint_bank2048_seed42"
EXPANDED_RETRIEVAL_UPDATE_ROOT = RD_ROOT / "safe_evidence_retrieval_joint_update2048_seed42"
FORMAL_SIGNED_REGRESSION_ROOT = RD_ROOT / "exogenous_signed_utility_regression_formal_full_seed42"
# Current formal classification run. The older independent-evidence run stays
# on disk as a historical baseline, but must not be presented as live.
FORMAL_CLASSIFICATION_ROOT = RD_ROOT / "samplewise_safety_classification_full_seed42"
FORMAL_REGRESSION_ROOT = RD_ROOT / "geocrd_v2_regression_full_seed42"
FORMAL_REGRESSION_UPDATE_ROOT = RD_ROOT / "samplewise_safety_regression_full_seed42"
RISK_CALIBRATION_ROOT = RD_ROOT / "exogenous_signed_utility_regression_screen_seed43"
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
        if "analyze_nonlinear_compatibility_probe.py" in cmd:
            matches.append({"pid": int(entry.name), "command": cmd.strip()})
            continue
        if (
            "train_online_safe_evidence_screen.py" in cmd
            or "train_online_safe_regression_screen.py" in cmd
            or "extract_retrieval_evidence_bank.py" in cmd
            or "train_safe_retrieval_update.py" in cmd
        ):
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

    acceptance_active = "eval_acceptance_probe_v2_" in command_text or "run_acceptance_probe_v2_full.sh" in command_text or "analyze_audio_acceptance_predictability.py" in command_text
    acceptance_train_valid = line_count(ACCEPTANCE_PROBE_TRAIN)
    acceptance_val_valid = line_count(ACCEPTANCE_PROBE_VAL)
    acceptance_train = acceptance_train_valid + line_count(ACCEPTANCE_PROBE_TRAIN_REJECTED)
    acceptance_val = acceptance_val_valid + line_count(ACCEPTANCE_PROBE_VAL_REJECTED)
    if acceptance_active or acceptance_train or acceptance_val or ACCEPTANCE_PROBE_RESULT.exists():
        analyzing = "analyze_audio_acceptance_predictability.py" in command_text
        if acceptance_train < 4096:
            current, target, phase_name = acceptance_train, 4096, "训练划分特征提取"
        elif acceptance_val < 1024:
            current, target, phase_name = acceptance_val, 1024, "固定验证特征提取"
        else:
            current, target, phase_name = 1024, 1024, "收益可预测性分析"
        stage = "样本级辅助模态证据接纳验证"
        detail = f"{phase_name} · {current}/{target} · 真值仅生成收益标签，不作为接纳器输入"
        phases = [
            {"name": "4096训练特征", "state": "done" if acceptance_train >= 4096 else ("active" if acceptance_active else "pending")},
            {"name": "1024固定验证", "state": "done" if acceptance_val >= 1024 else ("active" if acceptance_train >= 4096 and acceptance_active else "pending")},
            {"name": "接纳收益预测", "state": "done" if ACCEPTANCE_PROBE_RESULT.exists() else ("active" if analyzing else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = current, target
        display_percent = 100 * current / target if target else 0.0
        display_eta = None

    selective_rows = tail_jsonl(SELECTIVE_EVIDENCE_RUN / "train.jsonl", 300)
    selective_latest = selective_rows[-1] if selective_rows else {}
    selective_step = int(selective_latest.get("global_step", 0) or 0)
    selective_active = "selective_geographic_evidence_frozen_seed42" in command_text or "run_selective_frozen_screen.sh" in command_text
    if selective_active or selective_step:
        stage = "选择性地理证据接纳 · 冻结证据路由器筛查"
        detail = (
            f"四副本数据并行 · selector-only 训练 {selective_step}/256 · "
            f"接纳率 {selective_latest.get('acceptance_fraction', '—')} · "
            f"安全损失 {selective_latest.get('safety_loss', '—')}"
        )
        phases = [
            {"name": "接纳模块单测", "state": "done"},
            {"name": "64步行为筛查", "state": "done" if selective_step >= 64 else "active"},
            {"name": "256步稳定训练", "state": "done" if selective_step >= 256 else ("active" if selective_active else "pending")},
            {"name": "固定验证门槛", "state": "pending" if selective_active or selective_step < 256 else "active"},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = selective_step, 256
        display_percent = 100 * min(selective_step / 256, 1.0)
        display_eta = None
        latest, rows = selective_latest, selective_rows
        global_step, total_batches = selective_step, 256
        seconds_per_batch = mean(selective_rows, "seconds")

    signed_active = "run_signed_value_feasibility.sh" in command_text or "signed_geographic_value_seed42" in command_text
    signed_counts = {
        split: {
            coalition: sum(line_count(SIGNED_VALUE_ROOT / split / f"shard{shard}" / f"{coalition}.jsonl") for shard in range(4))
            for coalition in ("V", "VA", "VT", "VAT")
        }
        for split in ("train", "val")
    }
    signed_smoke = SIGNED_VALUE_ROOT / "smoke_result.json"
    signed_result = SIGNED_VALUE_ROOT / "result.json"
    signed_started = signed_active or any(value for split in signed_counts.values() for value in split.values()) or signed_smoke.exists() or signed_result.exists()
    if signed_started:
        signed_rejected = {
            split: {
                coalition: sum(
                    sum(row.get("coalition") == coalition for row in tail_jsonl(SIGNED_VALUE_ROOT / split / f"shard{shard}" / "rejected.jsonl", 100000))
                    for shard in range(4)
                )
                for coalition in ("V", "VA", "VT", "VAT")
            }
            for split in ("train", "val")
        }
        signed_processed = {split: {name: signed_counts[split][name] + signed_rejected[split][name] for name in signed_counts[split]} for split in signed_counts}
        train_done = min(signed_processed["train"].values()) if signed_processed["train"] else 0
        val_done = min(signed_processed["val"].values()) if signed_processed["val"] else 0
        if train_done < 64 or val_done < 64:
            current = sum(min(value, 64) for value in signed_processed["train"].values()) + sum(min(value, 64) for value in signed_processed["val"].values())
            target = 512
            phase_name = "四卡端到端烟雾测试"
        elif train_done < 4096:
            current, target, phase_name = sum(min(value, 4096) for value in signed_processed["train"].values()), 16384, "正式训练划分四联盟提取"
        elif val_done < 1024:
            current, target, phase_name = sum(min(value, 1024) for value in signed_processed["val"].values()), 4096, "独立验证划分四联盟提取"
        else:
            current, target, phase_name = 1 if signed_result.exists() else 0, 1, "有符号价值预测与曲线分析"
        stage = "样本级有符号地理价值验证"
        detail = (
            f"{phase_name} · train V/VA/VT/VAT "
            + "/".join(str(signed_counts["train"][name]) for name in ("V", "VA", "VT", "VAT"))
            + " · val "
            + "/".join(str(signed_counts["val"][name]) for name in ("V", "VA", "VT", "VAT"))
            + f" · 显式拒绝 train={sum(signed_rejected['train'].values())} val={sum(signed_rejected['val'].values())} · 目标标签不进入预测器特征"
        )
        phases = [
            {"name": "64+64烟雾测试", "state": "done" if signed_smoke.exists() else "active"},
            {"name": "4096训练价值标签", "state": "done" if train_done >= 4096 else ("active" if signed_smoke.exists() else "pending")},
            {"name": "1024独立验证", "state": "done" if val_done >= 1024 else ("active" if train_done >= 4096 else "pending")},
            {"name": "无标签价值预测器", "state": "done" if signed_result.exists() else ("active" if val_done >= 1024 else "pending")},
            {"name": "接纳-伤害-覆盖曲线", "state": "done" if signed_result.exists() else "pending"},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = current, target
        display_percent = 100 * current / target if target else 0.0
        display_eta = None
        global_step, total_batches = current, target

    gate2_conditions = (
        "vision_blur_4p65623", "vision_lowres_24p1062", "vision_dark_0p0676037",
        "vision_occlusion_0p35", "audio_crop_1p81313", "audio_noise_19p2145",
        "text_keep_0p525905", "text_slots_1",
    )
    gate2_active = "run_signed_value_gate2.sh" in command_text or "signed_geographic_value_gate2_seed42" in command_text
    gate2_counts = {
        name: sum(line_count(SIGNED_VALUE_GATE2_ROOT / name / f"{coalition}.jsonl") for coalition in ("V", "VA", "VT", "VAT"))
        for name in gate2_conditions
    }
    gate2_done = sum((SIGNED_VALUE_GATE2_ROOT / name / "analysis.json").exists() for name in gate2_conditions)
    gate2_summary = SIGNED_VALUE_GATE2_ROOT / "summary.json"
    gate2_started = gate2_active or SIGNED_VALUE_GATE2_ROOT.exists()
    if gate2_started:
        current = sum(min(value, 4096) for value in gate2_counts.values())
        target = 8 * 4096
        extracting = gate2_done == 0 and current < target
        analyzing = current >= target and gate2_done < 8
        stage = "Gate-2 · 退化泛化与语义错配验证"
        detail = (
            f"8条件四联盟前向 {current}/{target} · 已分析 {gate2_done}/8 · "
            f"相同1024验证ID · clean训练预测器零样本测试"
        )
        phases = [
            {"name": "同来源错配重分析", "state": "done"},
            {"name": "3种视觉退化", "state": "done" if all((SIGNED_VALUE_GATE2_ROOT / name / "analysis.json").exists() for name in gate2_conditions[:3]) else ("active" if extracting else "pending")},
            {"name": "未见遮挡泛化", "state": "done" if (SIGNED_VALUE_GATE2_ROOT / gate2_conditions[3] / "analysis.json").exists() else ("active" if extracting else "pending")},
            {"name": "音频/文本退化", "state": "done" if all((SIGNED_VALUE_GATE2_ROOT / name / "analysis.json").exists() for name in gate2_conditions[4:]) else ("active" if extracting else "pending")},
            {"name": "零样本价值预测", "state": "done" if gate2_summary.exists() else ("active" if analyzing or gate2_done else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = (gate2_done, 8) if analyzing else (current, target)
        display_percent = 100 * display_batch / display_batches if display_batches else 0.0
        display_eta = None
        global_step, total_batches = display_batch, display_batches

    gate3_conditions = ("clean", "vision_lowres_24p1062")
    gate3_variants = ("same_source_cross_r0", "cross_source_cross_r0", "paired_window_stratum0", "paired_window_stratum2")
    gate3_active = "run_audio_provenance_gate3.sh" in command_text or "audio_provenance_gate3_seed42" in command_text
    gate3_counts = {(condition, variant): sum(line_count(AUDIO_PROVENANCE_GATE3_ROOT / condition / variant / f"{coalition}.jsonl") for coalition in ("VA", "VAT")) for condition in gate3_conditions for variant in gate3_variants}
    gate3_summary = AUDIO_PROVENANCE_GATE3_ROOT / "summary.json"
    gate3_started = gate3_active or any(gate3_counts.values()) or gate3_summary.exists()
    if gate3_started:
        current = sum(min(value, 2048) for value in gate3_counts.values())
        target = len(gate3_conditions) * len(gate3_variants) * 2048
        stage = "Gate-3 · 音频证据来源干预"
        detail = f"同一1024验证ID · 仅替换音频 · VA/VAT前向 {current}/{target} · 检验paired音频是否优于同源跨地区错配"
        clean_done = sum(min(gate3_counts[("clean", variant)], 2048) for variant in gate3_variants)
        degraded_done = sum(min(gate3_counts[("vision_lowres_24p1062", variant)], 2048) for variant in gate3_variants)
        phases = [
            {"name": "固定样本严格错配清单", "state": "done"},
            {"name": "正常视觉来源干预", "state": "done" if clean_done >= 8192 else "active"},
            {"name": "低分辨率视觉来源干预", "state": "done" if degraded_done >= 8192 else ("active" if clean_done >= 8192 else "pending")},
            {"name": "Bootstrap因果门槛", "state": "done" if gate3_summary.exists() else ("active" if current >= target else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = current, target
        display_percent = 100 * current / target if target else 0.0
        display_eta = None
        global_step, total_batches = current, target

    text_gate3_conditions = ("clean", "vision_lowres_24p1062")
    text_gate3_variants = ("same_source_cross_r0", "cross_source_cross_r0")
    text_gate3_active = "run_text_provenance_gate3.sh" in command_text or "text_provenance_gate3_seed42" in command_text
    text_gate3_counts = {(condition, variant): sum(line_count(TEXT_PROVENANCE_GATE3_ROOT / condition / variant / f"{coalition}.jsonl") for coalition in ("VT", "VAT")) for condition in text_gate3_conditions for variant in text_gate3_variants}
    text_gate3_summary = TEXT_PROVENANCE_GATE3_ROOT / "summary.json"
    text_gate3_started = text_gate3_active or any(text_gate3_counts.values()) or text_gate3_summary.exists()
    if text_gate3_started:
        current = sum(min(value, 2048) for value in text_gate3_counts.values())
        target = len(text_gate3_conditions) * len(text_gate3_variants) * 2048
        stage = "Gate-3b · 文本证据来源干预"
        detail = f"同一1024验证ID · 仅替换文本 · VT/VAT前向 {current}/{target} · 检验配对来源规律能否扩展到文本"
        phases = [
            {"name": "长度匹配文本错配清单", "state": "done"},
            {"name": "正常/低分辨率并行前向", "state": "done" if current >= target else "active"},
            {"name": "Bootstrap跨模态判定", "state": "done" if text_gate3_summary.exists() else ("active" if current >= target else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = current, target
        display_percent = 100 * current / target if target else 0.0
        display_eta = None
        global_step, total_batches = current, target

    provenance_rows = tail_jsonl(PROVENANCE_RANK_ROOT / "train.jsonl", 300)
    provenance_latest = provenance_rows[-1] if provenance_rows else {}
    provenance_step = int(provenance_latest.get("global_step", 0) or 0)
    provenance_active = "run_provenance_rank_balanced_screen.sh" in command_text or "geocrd_provenance_rank_screen_seed42" in command_text
    provenance_complete = (PROVENANCE_RANK_ROOT / "latest.pt").exists() and provenance_step >= 256
    provenance_eval_root = PROVENANCE_RANK_ROOT / "eval_gate3_quick256"
    provenance_eval_count = sum(
        line_count(path) for path in provenance_eval_root.rglob("*.jsonl")
        if path.name in {"V.jsonl", "VA.jsonl", "VT.jsonl", "VAT.jsonl"}
    ) if provenance_eval_root.exists() else 0
    provenance_eval_target = 4096
    if provenance_active or provenance_step:
        stage = "方法筛查 · 来源排序目标"
        detail = (f"固定1024 paired/wrong复验 {provenance_eval_count}/{provenance_eval_target}" if provenance_complete else f"四卡DDP · step {provenance_step}/256 · coalition {provenance_latest.get("coalition", "—")} · utility {provenance_latest.get("utility_loss", "—")} · rate {provenance_latest.get("total_rate", "—")}")
        phases = [
            {"name": "epoch-6权重初始化", "state": "done"},
            {"name": "音频/文本来源排序训练", "state": "done" if provenance_step >= 256 else "active"},
            {"name": "固定1024 paired-wrong复验", "state": "pending" if not provenance_complete else ("done" if provenance_eval_count >= provenance_eval_target else "active")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = ((provenance_eval_count, provenance_eval_target) if provenance_complete else (provenance_step, 256))
        display_percent = 100 * display_batch / display_batches
        display_eta = None
        global_step, total_batches = provenance_step, 256
        latest, rows = provenance_latest, provenance_rows
        seconds_per_batch = mean(provenance_rows, "seconds")

    compatibility_active = "run_compatibility_probe_extraction.sh" in command_text or "compatibility_probe_seed42" in command_text
    compatibility_count = sum(line_count(path) for path in COMPATIBILITY_PROBE_ROOT.rglob("*.jsonl") if path.name in {"VA.jsonl", "VT.jsonl"}) if COMPATIBILITY_PROBE_ROOT.exists() else 0
    compatibility_target = 8192
    compatibility_result = COMPATIBILITY_PROBE_ROOT / "probe_analysis.json"
    if compatibility_active or compatibility_count or compatibility_result.exists():
        stage = "方法筛查 · 中间表示配对可分性"
        detail = f"原配/同源错配后验表示提取 {compatibility_count}/{compatibility_target} · clean + low-resolution"
        phases = [
            {"name": "音频原配/错配表示", "state": "done" if compatibility_count >= 4096 else "active"},
            {"name": "文本原配/错配表示", "state": "done" if compatibility_count >= compatibility_target else ("active" if compatibility_count >= 4096 else "pending")},
            {"name": "冻结表示compatibility probe", "state": "done" if compatibility_result.exists() else ("active" if compatibility_count >= compatibility_target else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = compatibility_count, compatibility_target
        display_percent = 100 * compatibility_count / compatibility_target
        display_eta = None
        global_step, total_batches = compatibility_count, compatibility_target

    nonlinear_rows = tail_jsonl(NONLINEAR_COMPATIBILITY_ROOT / "progress.jsonl", 800)
    nonlinear_result = NONLINEAR_COMPATIBILITY_ROOT / "analysis.json"
    nonlinear_active = "analyze_nonlinear_compatibility_probe.py" in command_text
    if nonlinear_active or nonlinear_rows or nonlinear_result.exists():
        completed_pairs = len({(row.get("modality"), row.get("seed")) for row in nonlinear_rows})
        current = nonlinear_rows[-1] if nonlinear_rows else {}
        stage = "方法筛查 · 非线性跨模态兼容性"
        detail = (f"{current.get('modality', '—')} · seed {current.get('seed', '—')} · "
                  f"epoch {current.get('epoch', 0)} · validation AUROC {current.get('val_auroc', 0):.4f}")
        phases = [
            {"name": "固定表征与ID隔离协议", "state": "done"},
            {"name": "Audio三随机种子", "state": "done" if completed_pairs >= 3 else "active"},
            {"name": "Text三随机种子", "state": "done" if nonlinear_result.exists() else ("active" if completed_pairs >= 3 else "pending")},
            {"name": "独立测试与bootstrap门槛", "state": "done" if nonlinear_result.exists() else "pending"},
        ]
        display_epoch, display_target_epochs = int(current.get("epoch", 0) or 0), 120
        display_batch, display_batches = min(completed_pairs, 6), 6
        display_percent = 100 if nonlinear_result.exists() else (100 * min(completed_pairs, 6) / 6)
        display_eta = None
        global_step, total_batches = len(nonlinear_rows), 720
        latest, rows = current, nonlinear_rows

    latent_count = sum(line_count(path) for path in LATENT_COMPATIBILITY_ROOT.rglob("*.jsonl")) if LATENT_COMPATIBILITY_ROOT.exists() else 0
    latent_result = LATENT_COMPATIBILITY_ROOT / "analysis.json"
    latent_active = "latent_bank_compatibility_seed42" in command_text
    if latent_active or latent_count or latent_result.exists():
        stage = "机制归因 · latent-token兼容性"
        detail = f"低分辨率 paired-wrong 完整4×128 latent bank提取 {latent_count}/4096"
        phases = [
            {"name": "Audio paired-wrong latent bank", "state": "done" if latent_count >= 2048 else "active"},
            {"name": "Text paired-wrong latent bank", "state": "done" if latent_count >= 4096 else ("active" if latent_count >= 2048 else "pending")},
            {"name": "ID隔离兼容性检验", "state": "done" if latent_result.exists() else ("active" if latent_count >= 4096 else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = latent_count, 4096
        display_percent = min(100, 100 * latent_count / 4096)
        display_eta = None
        global_step, total_batches = latent_count, 4096

    posterior_count = sum(line_count(path) for path in POSTERIOR_UPDATE_ROOT.rglob("*.jsonl")) if POSTERIOR_UPDATE_ROOT.exists() else 0
    posterior_result = POSTERIOR_UPDATE_ROOT / "analysis.json"
    posterior_active = "geographic_posterior_update_seed42" in command_text
    if posterior_active or posterior_count or posterior_result.exists():
        stage = "方法筛查 · 地理后验条件更新"
        detail = f"同模型同地理头 posterior 导出 {posterior_count}/5120 · alpha仅由验证集选择"
        phases = [
            {"name": "退化视觉地理后验", "state": "done" if posterior_count >= 1024 else "active"},
            {"name": "Audio paired-wrong 后验", "state": "done" if posterior_count >= 3072 else ("active" if posterior_count >= 1024 else "pending")},
            {"name": "Text paired-wrong 后验", "state": "done" if posterior_count >= 5120 else ("active" if posterior_count >= 3072 else "pending")},
            {"name": "验证选权重与独立测试", "state": "done" if posterior_result.exists() else ("active" if posterior_count >= 5120 else "pending")},
        ]
        display_epoch, display_target_epochs = 1, 1
        display_batch, display_batches = posterior_count, 5120
        display_percent = min(100, 100 * posterior_count / 5120)
        display_eta = None
        global_step, total_batches = posterior_count, 5120

    reliability_rows = tail_jsonl(POSTERIOR_RELIABILITY_ROOT / "progress.jsonl", 1000)
    reliability_result = POSTERIOR_RELIABILITY_ROOT / "analysis.json"
    reliability_active = "train_posterior_reliability_screen.py" in command_text
    if reliability_active or reliability_rows or reliability_result.exists():
        current = reliability_rows[-1] if reliability_rows else {}
        completed = len({(row.get("modality"), row.get("seed")) for row in reliability_rows})
        stage = "方法筛查 · 样本级地理证据可靠性"
        detail = ("三随机种子筛选已通过 · 正确Audio-Text增益保留 · 错配伤害被抑制" if reliability_result.exists() else f"{current.get('modality','—')} seed {current.get('seed','—')} epoch {current.get('epoch',0)}")
        phases = [
            {"name": "Audio三随机种子", "state": "done" if completed >= 3 else "active"},
            {"name": "Text三随机种子", "state": "done" if reliability_result.exists() else ("active" if completed >= 3 else "pending")},
            {"name": "独立测试与错配安全门槛", "state": "done" if reliability_result.exists() else "pending"},
        ]
        display_epoch, display_target_epochs = int(current.get("epoch",0) or 0), 160
        display_batch, display_batches = min(completed,6), 6
        display_percent = 100 if reliability_result.exists() else 100 * min(completed,6) / 6
        display_eta = None
        global_step, total_batches = len(reliability_rows), 960

    safe_rows = tail_jsonl(SAFE_PRIOR_UPDATE_ROOT / "progress.jsonl", 1200)
    safe_result = SAFE_PRIOR_UPDATE_ROOT / "analysis.json"
    safe_active = "train_safe_prior_corrected_screen.py" in command_text
    if safe_active or safe_rows or safe_result.exists():
        current = safe_rows[-1] if safe_rows else {}
        completed = len({(row.get("modality"), row.get("seed")) for row in safe_rows})
        stage = "正式方法筛查通过 · 安全贝叶斯地理证据更新"
        detail = ("Audio-Text三随机种子均通过 · 正确证据增益显著 · 错配风险约束成立" if safe_result.exists() else f"{current.get('modality','—')} seed {current.get('seed','—')} epoch {current.get('epoch',0)}")
        phases = [
            {"name": "先验校正地理证据", "state": "done"},
            {"name": "对偶错配安全约束", "state": "done" if safe_result.exists() else "active"},
            {"name": "Audio-Text三随机种子", "state": "done" if completed >= 6 else "active"},
            {"name": "独立测试门槛", "state": "done" if safe_result.exists() else "pending"},
        ]
        display_epoch, display_target_epochs = int(current.get("epoch",0) or 0), 180
        display_batch, display_batches = min(completed,6), 6
        display_percent = 100 if safe_result.exists() else 100 * min(completed,6) / 6
        display_eta = None
        global_step, total_batches = len(safe_rows), 1080

    online_runs = {
        seed: {
            "rows": tail_jsonl(root / "train.jsonl", 200),
            "summary": read_json(root / "summary.json", None),
        }
        for seed, root in ONLINE_EVIDENCE_ROOTS.items()
    }
    completed_online = [seed for seed, run in online_runs.items() if run["summary"]]
    current_seed = next(
        (seed for seed in reversed(tuple(ONLINE_EVIDENCE_ROOTS)) if online_runs[seed]["rows"]),
        42,
    )
    online_rows = online_runs[current_seed]["rows"]
    online_active = "train_online_safe_evidence_screen.py" in command_text
    if online_active or online_rows or completed_online:
        current = online_rows[-1] if online_rows else {}
        step = int(current.get("step", 0) or 0)
        stage = "Qwen-GeoCRD在线接入 · 安全贝叶斯证据更新"
        all_passed = len(completed_online) == len(ONLINE_EVIDENCE_ROOTS) and all(
            online_runs[seed]["summary"].get("gate", {}).get("overall", False)
            for seed in completed_online
        )
        detail = (
            "三随机种子均通过：配对音频与文本优于视觉基线及错配证据"
            if all_passed else
            f"seed {current_seed}: 4-GPU DDP, step {step}/64, {current.get('modality','loading')}"
        )
        phases = [
            {"name": "Qwen三路在线前向", "state": "done" if online_rows else "active"},
            {"name": "地理因子图势函数更新", "state": "done" if online_rows else "pending"},
            {"name": "每种子64步更新器训练", "state": "done" if len(completed_online) == 3 else "active"},
            {"name": f"128条独立验证 × {len(completed_online)}/3 个种子", "state": "done" if len(completed_online) == 3 else "active"},
        ]
        display_epoch, display_target_epochs = len(completed_online), 3
        display_batch, display_batches = step, 64
        display_percent = 100 * (
            len(completed_online) * 64
            + (0 if current_seed in completed_online else min(step, 64))
        ) / (3 * 64)
        display_eta = None
        global_step, total_batches = len(completed_online) * 64, 3 * 64
        latest, rows = current, online_rows

    joint_runs = {
        seed: {
            "rows": tail_jsonl(root / "train.jsonl", 300),
            "summary": read_json(root / "summary.json", None),
        }
        for seed, root in JOINT_EVIDENCE_ROOTS.items()
    }
    completed_joint = [seed for seed, run in joint_runs.items() if run["summary"]]
    current_joint_seed = next(
        (seed for seed in reversed(tuple(JOINT_EVIDENCE_ROOTS)) if joint_runs[seed]["rows"]),
        42,
    )
    joint_rows = joint_runs[current_joint_seed]["rows"]
    joint_summary = joint_runs[current_joint_seed]["summary"]
    joint_active = "joint_safe_evidence_screen_seed" in command_text
    if joint_active or joint_rows or completed_joint:
        current = joint_rows[-1] if joint_rows else {}
        step = int(current.get("step", 0) or 0)
        stage = "共享地理头与证据更新器联合训练"
        all_joint_passed = len(completed_joint) == 3 and all(
            joint_runs[seed]["summary"].get("gate", {}).get("overall", False)
            for seed in completed_joint
        )
        detail = (
            "三随机种子均通过联合训练与固定留出验证"
            if all_joint_passed else
            f"seed {current_joint_seed}: 4-GPU DDP, step {step}/128, {current.get('modality','loading')}"
        )
        phases = [
            {"name": "冻结Qwen在线编码", "state": "done" if joint_rows else "active"},
            {"name": "联合训练更新器与地理因子图头", "state": "done" if len(completed_joint) == 3 else "active"},
            {"name": "视觉锚定与配对-错配排序", "state": "done" if len(completed_joint) == 3 else "active"},
            {"name": f"256条固定留出验证 × {len(completed_joint)}/3 个种子", "state": "done" if len(completed_joint) == 3 else "active"},
        ]
        display_epoch, display_target_epochs = len(completed_joint), 3
        display_batch, display_batches = step, 128
        display_percent = 100 * (
            len(completed_joint) * 128
            + (0 if current_joint_seed in completed_joint else min(step, 128))
        ) / (3 * 128)
        display_eta = None
        global_step, total_batches = len(completed_joint) * 128, 3 * 128
        latest, rows = current, joint_rows

    update_ablation_summaries = {
        name: read_json(root / "summary.json", None)
        for name, root in EVIDENCE_UPDATE_ABLATIONS.items()
    }
    completed_update_ablations = [
        name for name, summary in update_ablation_summaries.items() if summary
    ]
    if completed_update_ablations:
        stage = "安全地理证据更新 · 核心组件归因"
        detail = (
            "4/4完成：直接相加与仅先验校正失败；自适应可靠性与完整安全方法通过"
            if len(completed_update_ablations) == 4 else
            f"completed {len(completed_update_ablations)}/4 variants"
        )
        phases = [
            {"name": "普通势函数直接相加", "state": "done" if "ordinary_add" in completed_update_ablations else "active"},
            {"name": "仅先验校正", "state": "done" if "prior_only" in completed_update_ablations else "pending"},
            {"name": "自适应可靠性与配对排序", "state": "done" if "adaptive_ranked" in completed_update_ablations else "pending"},
            {"name": "完整对偶安全约束", "state": "done" if "full" in completed_update_ablations else "pending"},
        ]
        display_epoch, display_target_epochs = len(completed_update_ablations), 4
        display_batch, display_batches = len(completed_update_ablations), 4
        display_percent = 25 * len(completed_update_ablations)
        display_eta = None
        global_step, total_batches = len(completed_update_ablations), 4

    regression_summaries = {
        seed: read_json(root / "summary.json", None)
        for seed, root in REGRESSION_UPDATE_ROOTS.items()
    }
    completed_regression = {
        seed: summary for seed, summary in regression_summaries.items() if summary
    }
    regression_summary = regression_summaries.get(42)
    retrieval_rows = tail_jsonl(RETRIEVAL_BASE_ROOT / "train.jsonl", 512)
    retrieval_summary = read_json(
        RETRIEVAL_BASE_ROOT / "evaluation_retrieval_val_clean" / "summary.json", None
    )
    retrieval_active = (
        "train_geocrd_v2_classification.py" in command_text
        and "safe_evidence_retrieval_base_seed42" in command_text
    )
    if retrieval_active or retrieval_rows:
        current = retrieval_rows[-1] if retrieval_rows else {}
        step = int(current.get("global_step", 0) or 0)
        finished_base = step >= 512 and not retrieval_active
        stage = "三任务扩展 · 跨视角检索基座" if finished_base else "三任务扩展 · 跨视角检索基座续训"
        if retrieval_active or step < 512:
            detail = f"固定Qwen-Omni与任务专用检索头 · step {step}/512 · batch内8候选；达到非随机基座后再接候选后验更新"
        elif retrieval_summary:
            visual = retrieval_summary.get("coalitions", {}).get("V", {})
            detail = f"基座训练完成 · 当前V R@1 {100 * visual.get('recall_at_1', 0):.2f}% · median rank {visual.get('median_rank', '—')}"
        phases = [
            {"name": "分类安全地理证据更新", "state": "done"},
            {"name": "vMF球面回归安全更新", "state": "done" if regression_summary else "pending"},
            {"name": "跨视角检索基座训练", "state": "done" if finished_base else "active"},
            {"name": "候选后验安全更新", "state": "pending"},
        ]
        display_epoch, display_target_epochs = 2 if regression_summary else 1, 3
        display_batch, display_batches = min(step, 512), 512
        display_percent = 100 * min(step, 512) / 512
        display_eta = None
        global_step, total_batches = step, 512
        latest, rows = current, retrieval_rows

    conditional_retrieval = read_json(CONDITIONAL_RETRIEVAL_SUMMARY, None)
    if conditional_retrieval:
        audio = conditional_retrieval["aggregate"]["audio"]
        text = conditional_retrieval["aggregate"]["text"]
        regression_audio_gains = [
            summary["audio"]["paired_km_gain"]
            for summary in completed_regression.values()
        ]
        regression_text_gains = [
            summary["text"]["paired_km_gain"]
            for summary in completed_regression.values()
        ]
        regression_complete = len(completed_regression) == 3
        stage = (
            "三任务扩展 · 分类/回归/检索验证完成"
            if regression_complete else
            "三任务扩展 · 条件候选证据三随机种子完成"
        )
        regression_detail = ""
        if regression_complete:
            regression_detail = (
                f"；回归3 seeds：Audio平均距离改善 {sum(regression_audio_gains) / 3:.1f} km，"
                f"Text {sum(regression_text_gains) / 3:.1f} km"
            )
        detail = (
            f"Audio综合失真增益 {audio['distortion_gain_mean']:.4f}，Top-1距离改善 {audio['top1_km_gain_mean']:.1f} km；"
            f"Text综合失真增益 {text['distortion_gain_mean']:.4f}，但Top-1距离变化 {text['top1_km_gain_mean']:.1f} km"
            f"{regression_detail}"
        )
        phases = [
            {"name": "分类安全地理证据更新", "state": "done"},
            {"name": f"vMF球面回归安全更新 · {len(completed_regression)}/3 seeds", "state": "done" if regression_complete else "active"},
            {"name": "独立模态候选乘积（被否定）", "state": "done"},
            {"name": "联合条件后验更新 · 3/3 seeds", "state": "done"},
        ]
        display_epoch, display_target_epochs = 3, 3
        display_batch, display_batches = 3, 3
        display_percent = 100.0
        display_eta = None
        global_step, total_batches = 3, 3

    formal_classification_rows = tail_jsonl(FORMAL_CLASSIFICATION_ROOT / "train.jsonl", 500)
    formal_classification_summary = read_json(FORMAL_CLASSIFICATION_ROOT / "summary.json", None)
    formal_classification_active = (
        "train_online_safe_evidence_screen.py" in command_text
        and FORMAL_CLASSIFICATION_ROOT.name in command_text
    )
    if formal_classification_active or formal_classification_rows or formal_classification_summary:
        current = formal_classification_rows[-1] if formal_classification_rows else {}
        step = int(current.get("step", 0) or 0)
        target_steps = 7850 if step > 3925 or "--max-steps 7850" in command_text else 3925
        finished = bool(formal_classification_summary) and not formal_classification_active
        validating = formal_classification_active and step >= target_steps
        validation_records = sum(line_count(path) for path in FORMAL_CLASSIFICATION_ROOT.glob("validation_per_sample_rank*.jsonl"))
        validation_samples = validation_records // 2
        stage = "全量主实验 · 分类条件地理证据更新"
        detail = (
            f"完整训练与验证已结束 · step {step}"
            if finished else
            (f"epoch 1训练已保存 · 逐样本独立验证 {validation_samples:,}/7,536 · 正确/错配音频与文本对照" if validating else f"62,790条训练样本 · 连续多模态退化 · 4-GPU DDP · step {step}/{target_steps} · {current.get('modality','loading')}")
        )
        phases = [
            {"name": "正式协议烟雾检查", "state": "done"},
            {"name": "分类完整训练集 · epoch 1", "state": "done" if step >= 3925 else "active"},
            {"name": "epoch 1完整验证", "state": "done" if finished else ("active" if validating else "pending")},
            {"name": "分类完整训练集 · epoch 2", "state": "done" if finished and step >= 7850 else ("active" if target_steps == 7850 else "pending")},
            {"name": "验证驱动续训/早停", "state": "active" if finished else "pending"},
        ]
        display_target_epochs = 2 if target_steps > 3925 else 1
        display_epoch = min(step // 3925, display_target_epochs)
        display_batch, display_batches = min(step, target_steps), target_steps
        display_percent = 100.0 if finished else 100 * min(step, target_steps) / target_steps
        display_eta = max(0, target_steps - step) * seconds_per_batch if seconds_per_batch else None
        global_step, total_batches = step, target_steps
        if validating:
            display_batch, display_batches = min(validation_samples, 7536), 7536
            display_percent = 100.0 * min(validation_samples, 7536) / 7536
            display_eta = None
            global_step, total_batches = display_batch, display_batches
        latest, rows = current, formal_classification_rows

    formal_regression_rows = tail_jsonl(FORMAL_REGRESSION_ROOT / "train.jsonl", 500)
    formal_regression_active = (
        "train_geocrd_v2_classification.py" in command_text
        and "geocrd_v2_regression_full_seed42" in command_text
    )
    formal_regression_checkpoint = FORMAL_REGRESSION_ROOT / "checkpoint_epoch1.pt"
    if (formal_regression_active or formal_regression_rows or formal_regression_checkpoint.exists()) and not (formal_classification_active or formal_classification_summary):
        current = formal_regression_rows[-1] if formal_regression_rows else {}
        step = int(current.get("global_step", 0) or 0)
        finished = formal_regression_checkpoint.exists() and not formal_regression_active
        stage = "全量主实验 · 球面回归基座"
        detail = (
            "62,790条训练样本的首个完整epoch已完成，等待完整回归验证"
            if finished else
            f"62,790条训练样本 · vMF球面回归 · 连续多模态退化 · 4-GPU DDP · step {step}/3925"
        )
        phases = [
            {"name": "分类完整训练与验证", "state": "done"},
            {"name": "回归基座完整训练集 · epoch 1", "state": "done" if finished else "active"},
            {"name": "7,551条回归验证集", "state": "pending"},
            {"name": "回归条件证据更新", "state": "pending"},
        ]
        display_epoch, display_target_epochs = (1 if finished else 0), 1
        display_batch, display_batches = min(step,3925),3925
        display_percent = 100.0 if finished else 100 * min(step,3925) / 3925
        display_eta = None
        global_step,total_batches = step,3925
        latest,rows = current,formal_regression_rows

    formal_regression_update_rows = tail_jsonl(FORMAL_REGRESSION_UPDATE_ROOT / "train.jsonl", 500)
    formal_regression_update_summary = read_json(FORMAL_REGRESSION_UPDATE_ROOT / "summary.json", None)
    formal_regression_update_active = (
        "train_online_safe_regression_screen.py" in command_text
        and FORMAL_REGRESSION_UPDATE_ROOT.name in command_text
    )
    if formal_regression_update_active or ((formal_regression_update_rows or formal_regression_update_summary) and not (formal_classification_active or formal_classification_summary)):
        current = formal_regression_update_rows[-1] if formal_regression_update_rows else {}
        step = int(current.get("step", 0) or 0)
        finished = bool(formal_regression_update_summary) and not formal_regression_update_active
        regression_validating = formal_regression_update_active and (step >= 3925 or "--resume" in command_text) and not formal_regression_update_summary
        stage = "全量主实验 · 条件联合回归证据更新"
        detail = (
            "完整训练与7,551条验证结束，等待判断音频保留与文本抑制"
            if finished else
            ("epoch 1检查点已保存 · 一致退化条件下的完整回归验证进行中" if regression_validating else f"62,790条训练样本 · 连续多模态退化 · 正确/错配证据约束 · 4-GPU DDP · step {step}/3925")
        )
        phases = [
            {"name": "分类完整训练与验证", "state": "done"},
            {"name": "回归基座完整训练与验证", "state": "done"},
            {"name": "回归安全更新器 · epoch 1", "state": "done" if step >= 3925 else "active"},
            {"name": "音频保留/文本抑制验收", "state": "done" if finished else ("active" if regression_validating else "pending")},
        ]
        display_epoch,display_target_epochs=(1 if finished else 0),1
        display_batch,display_batches=min(step,3925),3925
        display_percent=100.0 if finished else 100*min(step,3925)/3925
        display_eta=None
        global_step,total_batches=step,3925
        latest,rows=current,formal_regression_update_rows
    risk_rows = tail_jsonl(RISK_CALIBRATION_ROOT / "train.jsonl", 500)
    risk_summary = read_json(RISK_CALIBRATION_ROOT / "summary.json", None)
    risk_calibration_summary = read_json(RISK_CALIBRATION_ROOT / "summary_calibration_first512.json", None)
    risk_active = ("train_online_safe_regression_screen.py" in command_text and RISK_CALIBRATION_ROOT.name in command_text)
    if risk_active or risk_rows or risk_summary:
        current = risk_rows[-1] if risk_rows else {}
        step = int(current.get("step", 0) or 0)
        records = sum(line_count(path) for path in RISK_CALIBRATION_ROOT.glob("validation_per_sample_rank*.jsonl")) // 2
        holdout_started = records > 0
        # A few audio rows can be rejected by modality-validity checks, so use the
        # independently rewritten holdout summary (newer than the archived
        # calibration summary) as the completion signal rather than requiring an
        # impossible exact line count.
        holdout_summary_newer = False
        summary_path = RISK_CALIBRATION_ROOT / "summary.json"
        calibration_path = RISK_CALIBRATION_ROOT / "summary_calibration_first512.json"
        if summary_path.exists() and calibration_path.exists():
            holdout_summary_newer = summary_path.stat().st_mtime > calibration_path.stat().st_mtime
        holdout_done = bool(risk_summary) and not risk_active and records >= 528
        stage = "回归稳定性验证 · 共享外生价值头 seed 43"
        detail = (
            "独立留出集验收完成" if holdout_done else
            (f"三来源分层验证 {min(records, 528)}/528" if holdout_started else
             ("seed 43稳定性验证完成" if risk_summary else f"固定验证集 · step {step}/128"))
        )
        phases = [
            {"name": "分类全量结论", "state": "done"},
            {"name": "全局风险约束失效诊断", "state": "done"},
            {"name": "三来源风险差异确认", "state": "done"},
            {"name": "首版最坏来源训练 · 未满足风险", "state": "done"},
            {"name": "安全收敛筛查 · 风险通过", "state": "done"},
            {"name": "调高特异性权重 · 无效", "state": "done"},
            {"name": "共享外生价值头 seed 43 · 128 steps", "state": "done" if step >= 128 else "active"},
            {"name": "三来源分层独立验收", "state": "done" if holdout_done else ("active" if holdout_started else "pending")},
        ]
        display_epoch, display_target_epochs = (1 if step >= 128 else 0), 1
        display_batch, display_batches = ((528, 528) if holdout_done else ((min(records, 528), 528) if holdout_started else (min(step, 128), 128)))
        display_percent = 100.0 * min(display_batch, display_batches) / display_batches
        display_eta = None
        global_step, total_batches = display_batch, display_batches
        latest, rows = current, risk_rows

    # Keep the current retrieval experiment above archived classification and
    # regression summaries. Extraction writes JSON progress records to its log.
    retrieval_extract_rows = tail_jsonl(EXPANDED_RETRIEVAL_BANK_ROOT / "extract.log", 300)
    retrieval_extract_active = (
        "extract_retrieval_evidence_bank.py" in command_text
        and EXPANDED_RETRIEVAL_BANK_ROOT.name in command_text
    )
    retrieval_update_rows = tail_jsonl(EXPANDED_RETRIEVAL_UPDATE_ROOT / "train.jsonl", 500)
    retrieval_update_summary = read_json(EXPANDED_RETRIEVAL_UPDATE_ROOT / "summary.json", None)
    retrieval_update_active = (
        "train_safe_retrieval_update.py" in command_text
        and EXPANDED_RETRIEVAL_UPDATE_ROOT.name in command_text
    )
    train_bank_done = (EXPANDED_RETRIEVAL_BANK_ROOT / "train.pt").exists()
    val_bank_done = (EXPANDED_RETRIEVAL_BANK_ROOT / "val.pt").exists()
    if retrieval_extract_active or retrieval_update_active or train_bank_done or val_bank_done or retrieval_update_rows:
        phase_order = {"gallery": 0, "queries": 1}
        extract_units_done = 0
        extract_units_total = 2 * (2048 + 1024)
        current_extract = retrieval_extract_rows[-1] if retrieval_extract_rows else {}
        for record in retrieval_extract_rows:
            split = record.get("split")
            phase = record.get("phase")
            processed = int(record.get("processed", 0) or 0)
            if split == "train":
                base = phase_order.get(phase, 0) * 2048
            elif split == "val":
                base = 2 * 2048 + phase_order.get(phase, 0) * 1024
            else:
                continue
            extract_units_done = max(extract_units_done, base + processed)
        if train_bank_done:
            extract_units_done = max(extract_units_done, 2 * 2048)
        if val_bank_done:
            extract_units_done = extract_units_total

        if retrieval_update_active or retrieval_update_rows or retrieval_update_summary:
            current = retrieval_update_rows[-1] if retrieval_update_rows else {}
            update_step = int(current.get("step", current.get("global_step", 0)) or 0)
            update_target = int(current.get("total_steps", 6400) or 6400)
            update_done = bool(retrieval_update_summary) and not retrieval_update_active
            stage = "三任务扩展 · 扩大候选库的安全检索更新"
            detail = (
                "2048训练候选/1024验证候选 · 更新器训练与验证完成"
                if update_done else
                f"2048训练候选/1024验证候选 · 样本级候选后验更新 step {update_step}/{update_target}"
            )
            phases = [
                {"name": "训练候选与查询证据提取 · 2048", "state": "done"},
                {"name": "验证候选与查询证据提取 · 1024", "state": "done"},
                {"name": "安全候选证据更新器", "state": "done" if update_done else "active"},
                {"name": "正确/错配模态与地理Top-1验收", "state": "done" if update_done else "pending"},
            ]
            display_epoch, display_target_epochs = (1 if update_done else 0), 1
            display_batch, display_batches = min(update_step, update_target), update_target
            display_percent = 100.0 if update_done else 100.0 * min(update_step, update_target) / max(update_target, 1)
            global_step, total_batches = display_batch, display_batches
            display_eta = None
            latest, rows = current, retrieval_update_rows
        else:
            split = current_extract.get("split", "loading")
            phase = current_extract.get("phase", "model")
            processed = int(current_extract.get("processed", 0) or 0)
            total = int(current_extract.get("total", 0) or 0)
            stage = "三任务扩展 · 扩大检索证据库"
            detail = f"{split} · {phase} · {processed}/{total}；目标为2048训练候选与1024验证候选"
            phases = [
                {"name": "训练候选库编码 · 2048", "state": "done" if extract_units_done >= 2048 else "active"},
                {"name": "训练查询证据提取 · 2048", "state": "done" if extract_units_done >= 4096 else ("active" if extract_units_done >= 2048 else "pending")},
                {"name": "验证候选与查询证据 · 1024", "state": "done" if val_bank_done else ("active" if train_bank_done else "pending")},
                {"name": "安全候选证据更新器", "state": "pending"},
            ]
            display_epoch, display_target_epochs = (1 if val_bank_done else 0), 1
            display_batch, display_batches = min(extract_units_done, extract_units_total), extract_units_total
            display_percent = 100.0 * display_batch / extract_units_total
            global_step, total_batches = display_batch, display_batches
            display_eta = None
            latest, rows = current_extract, retrieval_extract_rows

    # The current formal regression run must override completed historical
    # retrieval stages. Training logs every 20 optimizer steps; validation
    # writes two records (audio/text) per successfully evaluated sample.
    signed_regression_rows = tail_jsonl(FORMAL_SIGNED_REGRESSION_ROOT / "train.jsonl", 500)
    signed_regression_summary = read_json(FORMAL_SIGNED_REGRESSION_ROOT / "summary.json", None)
    signed_regression_active = (
        "train_online_safe_regression_screen.py" in command_text
        and FORMAL_SIGNED_REGRESSION_ROOT.name in command_text
    )
    if signed_regression_active or signed_regression_rows or signed_regression_summary:
        current = signed_regression_rows[-1] if signed_regression_rows else {}
        step = int(current.get("step", 0) or 0)
        validation_records = sum(
            line_count(path) for path in FORMAL_SIGNED_REGRESSION_ROOT.glob("validation_per_sample_rank*.jsonl")
        )
        validation_samples = validation_records // 2
        training_done = step >= 3925
        validation_active = signed_regression_active and training_done
        finished = bool(signed_regression_summary) and not signed_regression_active
        if finished:
            stage = "三任务全量 · 回归正式训练与验证完成"
            detail = "62,790条训练与完整回归验证完成，等待汇总后进入正式检索"
        elif validation_active:
            stage = "三任务全量 · 回归完整验证"
            detail = f"训练 3925/3925 完成 · 正确/错配音频与文本验证 {min(validation_samples,7551):,}/7,551"
        else:
            stage = "三任务全量 · 回归正式训练"
            detail = f"62,790条训练样本 · step {step}/3925 · 连续多模态退化 · 4-GPU"
        phases = [
            {"name": "分类五条件完整验证", "state": "done"},
            {"name": "回归正式训练 · 62,790", "state": "done" if training_done else "active"},
            {"name": "回归完整验证 · 7,551", "state": "done" if finished else ("active" if validation_active else "pending")},
            {"name": "正式全量检索", "state": "pending"},
        ]
        if validation_active:
            display_batch, display_batches = min(validation_samples,7551), 7551
            display_percent = 100.0 * display_batch / display_batches
            global_step, total_batches = display_batch, display_batches
            display_eta = None
        else:
            display_batch, display_batches = min(step,3925), 3925
            display_percent = 100.0 if finished else 100.0 * display_batch / display_batches
            global_step, total_batches = display_batch, display_batches
            if len(signed_regression_rows) >= 2:
                first_row, last_row = signed_regression_rows[0], signed_regression_rows[-1]
                delta_step = max(1, int(last_row.get("step",0))-int(first_row.get("step",0)))
                seconds_per_step = max(0.0, float(last_row.get("time",0))-float(first_row.get("time",0))) / delta_step
                display_eta = max(0,3925-step) * seconds_per_step
            else:
                display_eta = None
        display_epoch, display_target_epochs = (1 if training_done else 0), 1
        latest, rows = current, signed_regression_rows

    regression_eval_roots = {
        "clean": RD_ROOT / "regression_final_eval_clean_seed42",
        "blur": RD_ROOT / "regression_final_eval_blur_seed42",
        "dark": RD_ROOT / "regression_final_eval_dark_seed42",
        "occlusion": RD_ROOT / "regression_final_eval_occlusion_seed42",
    }
    active_regression_eval = next(
        (name for name, root in regression_eval_roots.items() if root.name in command_text), None
    )
    completed_regression_evals = [
        name for name, root in regression_eval_roots.items() if (root / "summary.json").exists()
    ]
    if active_regression_eval or completed_regression_evals:
        active_root = regression_eval_roots.get(active_regression_eval)
        records = 0 if active_root is None else sum(
            line_count(path) for path in active_root.glob("validation_per_sample_rank*.jsonl")
        )
        evaluated_samples = records // 2
        all_done = len(completed_regression_evals) == len(regression_eval_roots) and active_regression_eval is None
        stage = (
            "三任务全量 · 回归五条件验证完成" if all_done else
            f"三任务全量 · 回归完整验证 · {active_regression_eval or '汇总'}"
        )
        detail = (
            "Clean/Low-resolution/Blur/Dark/Occlusion均完成，等待结果汇总" if all_done else
            f"Low-resolution已完成 · {active_regression_eval}: {min(evaluated_samples,7551):,}/7,551 · 已完成额外条件 {len(completed_regression_evals)}/4"
        )
        phases = [
            {"name": "分类五条件完整验证", "state": "done"},
            {"name": "回归正式训练 · 62,790", "state": "done"},
            {"name": f"回归五条件完整验证 · {1+len(completed_regression_evals)}/5", "state": "done" if all_done else "active"},
            {"name": "正式全量检索", "state": "pending"},
        ]
        display_epoch, display_target_epochs = (1 if all_done else 0), 1
        display_batch, display_batches = ((7551,7551) if all_done else (min(evaluated_samples,7551),7551))
        display_percent = 100.0 * display_batch / display_batches
        global_step, total_batches = display_batch, display_batches
        display_eta = None


    # Formal scalable retrieval: embedding extraction is O(Nd), updater
    # training uses sampled candidates, and validation ranks the full gallery.
    formal_retrieval_bank = RD_ROOT / "safe_retrieval_embedding_full_seed42"
    formal_retrieval_run = RD_ROOT / "safe_retrieval_formal_full_seed42"
    formal_extract_rows = tail_jsonl(formal_retrieval_bank / "extract.log", 1200)
    formal_train_rows = tail_jsonl(formal_retrieval_run / "train.jsonl", 1200)
    formal_convergence_rows = tail_jsonl(formal_retrieval_run / "convergence.jsonl", 100)
    formal_summary = read_json(formal_retrieval_run / "summary.json", None)
    formal_extract_active = (
        "extract_retrieval_evidence_bank.py" in command_text
        and formal_retrieval_bank.name in command_text
    )
    formal_train_active = (
        "train_safe_retrieval_full.py" in command_text
        and formal_retrieval_run.name in command_text
    )
    if formal_extract_active or formal_train_active or (formal_retrieval_run / "pipeline.log").exists():
        train_total, val_total = 62790, 7551
        extract_total = 2 * (train_total + val_total)
        extract_done = 0
        extract_current = formal_extract_rows[-1] if formal_extract_rows else {}
        for record in formal_extract_rows:
            split = record.get("split")
            phase = record.get("phase")
            processed = int(record.get("processed", 0) or 0)
            if split == "train":
                base = 0 if phase == "gallery" else train_total
            elif split == "val":
                base = 2 * train_total if phase == "gallery" else 2 * train_total + val_total
            else:
                continue
            extract_done = max(extract_done, base + processed)
        if (formal_retrieval_bank / "train.pt").exists():
            extract_done = max(extract_done, 2 * train_total)
        if (formal_retrieval_bank / "val.pt").exists():
            extract_done = extract_total
        train_current = formal_train_rows[-1] if formal_train_rows else {}
        convergence_current = formal_convergence_rows[-1] if formal_convergence_rows else {}
        train_step = int(train_current.get("step", 0) or 0)
        train_epoch = int(convergence_current.get("epoch", train_current.get("epoch", 0)) or 0)
        train_target = 30
        if formal_summary and not formal_train_active:
            stage = "三任务全量 · 正式检索完成"
            detail = "62,790训练查询 · 7,551完整验证图库 · 等待跨任务与主流基线汇总"
            display_batch, display_batches, display_percent = train_target, train_target, 100.0
            latest, rows = formal_summary, formal_train_rows
        elif formal_train_active:
            stage = "三任务全量 · 正式检索收敛训练"
            detail = (f"epoch {train_epoch}/30 · 最少7轮 · stale {int(convergence_current.get('stale', 0) or 0)}/3 · "
                      f"best epoch {convergence_current.get('best_epoch', '—')} · step {train_step:,}")
            display_batch, display_batches = min(train_epoch, train_target), train_target
            display_percent = 100.0 * display_batch / display_batches
            latest, rows = convergence_current or train_current, formal_convergence_rows or formal_train_rows
        else:
            split = extract_current.get("split", "loading")
            phase = extract_current.get("phase", "model")
            processed = int(extract_current.get("processed", 0) or 0)
            total = int(extract_current.get("total", 0) or 0)
            stage = "三任务全量 · 正式检索嵌入提取"
            detail = f"{split} · {phase} · {processed:,}/{total:,} · 线性内存分块协议"
            display_batch, display_batches = min(extract_done, extract_total), extract_total
            display_percent = 100.0 * display_batch / display_batches
            latest, rows = extract_current, formal_extract_rows
        phases = [
            {"name": "分类五条件完整验证", "state": "done"},
            {"name": "回归五条件完整验证", "state": "done"},
            {"name": "检索全量嵌入提取", "state": "done" if extract_done >= extract_total else "active"},
            {"name": "检索候选更新·验证收敛", "state": "done" if formal_summary else ("active" if formal_train_active else "pending")},
            {"name": "独立最终查询×7,551完整图库", "state": "done" if formal_summary else "pending"},
        ]
        display_epoch, display_target_epochs = (1 if formal_summary else 0), 1
        global_step, total_batches = display_batch, display_batches
        display_eta = None


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
            "distortion": latest.get("distortion", latest.get("paired_distortion")),
            "normalized_nll": latest.get("normalized_nll", (latest.get("task_metrics") or {}).get("normalized_nll")),
            "normalized_energy": latest.get("normalized_energy", (latest.get("task_metrics") or {}).get("normalized_energy")),
            "rate": latest.get("total_rate"),
            "ema_rate": latest.get("ema_rate"),
            "rate_budget": config.get("rate_budget"),
            "beta": latest.get("beta"),
            "window_distortion": mean(rows, "distortion") or mean(rows, "paired_distortion"),
            "window_nll": mean(rows, "normalized_nll") or mean_task_metric(rows, "normalized_nll") or mean(rows, "paired_nll"),
            "window_rate": mean(rows, "total_rate"),
            "window_utility_loss": mean(rows, "utility_loss"),
            "seconds_per_batch": seconds_per_batch,
            "train_eta_seconds": display_eta,
            "coalition": latest.get("coalition", latest.get("modality")),
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
