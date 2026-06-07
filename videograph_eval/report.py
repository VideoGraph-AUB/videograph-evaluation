"""
Report generation for evaluation results.

Outputs both machine-readable JSON and a human-readable markdown report.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def save_results_json(results: dict, output_path: str):
    """Save all results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results JSON saved to {output_path}")


def _ordered_dataset_keys(results: dict) -> list:
    preferred_order = [
        "egoschema",
        "nextqa-test",
        "nextqa-val",
        "video-mme-long",
        "video-mme-medium",
        "video-mme-short",
    ]
    dataset_keys = [
        key for key in results.keys()
        if key != "_meta" and isinstance(results.get(key), dict)
    ]
    ordered = [key for key in preferred_order if key in dataset_keys]
    ordered += sorted(key for key in dataset_keys if key not in preferred_order)
    return ordered


def _question_count(ds: dict, acc: dict) -> int:
    if "count" in acc:
        return int(acc.get("count", 0) or 0)
    if isinstance(acc.get("counts"), dict):
        return int(acc.get("counts", {}).get("total", 0) or 0)
    return int(ds.get("total_predictions", 0) or 0)


def _failed_count(ds: dict, acc: dict) -> int:
    if "failed" in acc:
        return int(acc.get("failed", 0) or 0)
    if "failed_predictions" in ds:
        return int(ds.get("failed_predictions", 0) or 0)
    total_preds = int(ds.get("total_predictions", 0) or 0)
    valid_preds = int(ds.get("valid_predictions", 0) or 0)
    return max(0, total_preds - valid_preds)


def _append_metric_table(lines_out: list, title: str, metric_map: dict, count_map: dict):
    if not metric_map:
        return
    lines_out.append(f"### {title}")
    lines_out.append("")
    lines_out.append("| Group | Accuracy | Count |")
    lines_out.append("|-------|----------|-------|")
    for key, value in sorted(metric_map.items()):
        count = count_map.get(key, 0) if isinstance(count_map, dict) else 0
        lines_out.append(f"| {key} | {value:.2%} | {count} |")
    lines_out.append("")


def generate_report(
    results: dict,
    output_path: str,
    version: str = "unknown",
):
    """
    Generate a human-readable markdown report.

    Args:
        results: Full results dict from evaluation
        output_path: Path to save the report.md
        version: Version label (e.g., 'v0')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_dataset_keys(results)

    lines = []
    lines.append(f"# VideoGraph Evaluation Report - {version}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append(f"**Version**: {version}")
    lines.append("")

    # --- Accuracy Summary ---
    lines.append("## Accuracy Summary")
    lines.append("")
    lines.append("| Dataset | Accuracy | Questions | Failed |")
    lines.append("|---------|----------|-----------|--------|")

    for dataset_key in ordered:
        ds = results.get(dataset_key, {})
        acc = ds.get("accuracy", {})
        overall = acc.get("overall", 0.0)
        count = _question_count(ds, acc)
        failed = _failed_count(ds, acc)
        lines.append(f"| {dataset_key} | {overall:.2%} | {count} | {failed} |")

    lines.append("")

    # --- NExT-QA Breakdown ---
    for split in [k for k in ordered if k.startswith("nextqa-")]:
        ds = results.get(split, {})
        acc = ds.get("accuracy", {})
        if not acc:
            continue

        lines.append(f"### {split} - Per-Category Accuracy")
        lines.append("")
        lines.append("| Category | Accuracy | Count |")
        lines.append("|----------|----------|-------|")

        counts = acc.get("counts", {})
        for cat in ["causal", "temporal", "descriptive"]:
            cat_acc = acc.get(cat, 0.0)
            cat_count = counts.get(cat, 0)
            lines.append(f"| {cat.capitalize()} | {cat_acc:.2%} | {cat_count} |")

        lines.append("")

        per_type = acc.get("per_type", {})
        if per_type:
            lines.append(f"### {split} - Per-Type Accuracy")
            lines.append("")
            lines.append("| Type | Accuracy |")
            lines.append("|------|----------|")
            for t, t_acc in sorted(per_type.items()):
                lines.append(f"| {t} | {t_acc:.2%} |")
            lines.append("")

    # --- VIDEO-MME Breakdown ---
    for split in [k for k in ordered if k.startswith("video-mme-")]:
        ds = results.get(split, {})
        acc = ds.get("accuracy", {})
        if not acc:
            continue

        counts = acc.get("counts", {})
        _append_metric_table(
            lines,
            f"{split} - Task Type Accuracy",
            acc.get("by_task_type", {}),
            counts.get("task_type", {}),
        )
        _append_metric_table(
            lines,
            f"{split} - Domain Accuracy",
            acc.get("by_domain", {}),
            counts.get("domain", {}),
        )

    # --- Performance Metrics ---
    has_performance = any(results.get(k, {}).get("performance") for k in ordered)

    if has_performance:
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("| Dataset | Sample Videos | Total API Calls | Avg Calls/Video | Total Cost (USD) | Avg Cost/Video (USD) | Avg Construction Wall Time/Video (s) | Avg Answer Time/Question (s) |")
        lines.append("|---------|--------------:|----------------:|----------------:|-----------------:|---------------------:|-------------------------------------:|----------------------------:|")

        for dataset_key in ordered:
            perf = results.get(dataset_key, {}).get("performance", {})
            if perf:
                lines.append(
                    f"| {dataset_key}"
                    f" | {perf.get('sample_videos', 0)}"
                    f" | {perf.get('total_api_calls', perf.get('total_llm_calls', 0))}"
                    f" | {perf.get('avg_api_calls_per_video', perf.get('avg_llm_calls_per_video', 0)):.1f}"
                    f" | ${perf.get('total_cost_usd', 0):.4f}"
                    f" | ${perf.get('avg_cost_per_video_usd', 0):.4f}"
                    f" | {perf.get('avg_processing_time_per_video_s', 0):.1f}"
                    f" | {perf.get('avg_answer_time_per_question_s', 0):.3f} |"
                )
        lines.append("")
        lines.append("Performance cost uses tracked OpenAI API calls during graph construction. Construction wall time is measured end-to-end per video and is distinct from summed API call latency.")
        lines.append("")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(f"*Generated by VideoGraph Evaluation on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Report saved to {output_path}")


