"""
Accuracy and performance metrics computation.
"""

import logging
from typing import Dict, List

from .datasets import get_nextqa_category

logger = logging.getLogger(__name__)


def _numeric(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: List[float]) -> float:
    vals = sorted(_numeric(value) for value in values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _sum_map(rows: List[dict], key: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for row in rows:
        value = row.get(key, {})
        if not isinstance(value, dict):
            continue
        for label, amount in value.items():
            totals[str(label)] = totals.get(str(label), 0.0) + _numeric(amount)
    return totals


def _sum_nested_map(rows: List[dict], key: str) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        value = row.get(key, {})
        if not isinstance(value, dict):
            continue
        for outer, inner_map in value.items():
            if not isinstance(inner_map, dict):
                continue
            outer_key = str(outer)
            totals.setdefault(outer_key, {})
            for inner, amount in inner_map.items():
                inner_key = str(inner)
                totals[outer_key][inner_key] = (
                    totals[outer_key].get(inner_key, 0.0) + _numeric(amount)
                )
    return totals


def _round_float_map(values: Dict[str, float], digits: int = 6) -> Dict[str, float]:
    return {key: round(value, digits) for key, value in sorted(values.items())}


def _round_count_map(values: Dict[str, float]) -> Dict[str, int]:
    return {key: int(round(value)) for key, value in sorted(values.items())}


def _round_nested_float_map(
    values: Dict[str, Dict[str, float]],
    digits: int = 6,
) -> Dict[str, Dict[str, float]]:
    return {
        outer: {inner: round(amount, digits) for inner, amount in sorted(inner_map.items())}
        for outer, inner_map in sorted(values.items())
    }


def _round_nested_count_map(values: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, int]]:
    return {
        outer: {inner: int(round(amount)) for inner, amount in sorted(inner_map.items())}
        for outer, inner_map in sorted(values.items())
    }


def compute_accuracy(predictions: List[dict]) -> float:
    """
    Compute top-1 MC accuracy.

    Args:
        predictions: List of dicts with 'answer' (ground-truth) and 'predicted'

    Returns:
        Accuracy as a float [0, 1]
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p["predicted"] == p["answer"])
    return correct / len(predictions)


def compute_nextqa_accuracy(predictions: List[dict]) -> dict:
    """
    Compute NExT-QA accuracy broken down by category.

    Args:
        predictions: List of dicts with 'answer', 'predicted', 'question_type'

    Returns:
        Dict with 'overall', 'causal', 'temporal', 'descriptive', 'per_type'
    """
    if not predictions:
        return {"overall": 0.0, "causal": 0.0, "temporal": 0.0, "descriptive": 0.0, "per_type": {}}

    overall = compute_accuracy(predictions)

    # Group by category
    by_category: Dict[str, List[dict]] = {}
    by_type: Dict[str, List[dict]] = {}

    for p in predictions:
        qtype = p.get("question_type", "")
        category = get_nextqa_category(qtype)

        by_category.setdefault(category, []).append(p)
        by_type.setdefault(qtype, []).append(p)

    return {
        "overall": round(overall, 4),
        "causal": round(compute_accuracy(by_category.get("causal", [])), 4),
        "temporal": round(compute_accuracy(by_category.get("temporal", [])), 4),
        "descriptive": round(compute_accuracy(by_category.get("descriptive", [])), 4),
        "per_type": {
            t: round(compute_accuracy(preds), 4)
            for t, preds in sorted(by_type.items())
        },
        "counts": {
            "total": len(predictions),
            "causal": len(by_category.get("causal", [])),
            "temporal": len(by_category.get("temporal", [])),
            "descriptive": len(by_category.get("descriptive", [])),
        },
        "failed": sum(1 for p in predictions if p.get("predicted", -1) < 0),
    }


def compute_egoschema_accuracy(predictions: List[dict]) -> dict:
    """
    Compute EgoSchema subset accuracy.

    Args:
        predictions: List of dicts with 'answer' and 'predicted'

    Returns:
        Dict with 'overall' and 'count'
    """
    return {
        "overall": round(compute_accuracy(predictions), 4),
        "count": len(predictions),
        "failed": sum(1 for p in predictions if p.get("predicted", -1) < 0),
    }


def compute_performance_metrics(
    video_stats: List[dict],
    answer_times: List[float],
) -> dict:
    """
    Compute performance metrics from per-video tracker data.

    Args:
        video_stats: List of per-video TrackerStats dicts
        answer_times: List of per-question answer times in seconds

    Returns:
        Dict with per-dataset totals and averages for calls, cost, processing,
        and answer time.
    """
    if not video_stats:
        return {
            "sample_videos": 0,
            "total_api_calls": 0,
            "total_llm_calls": 0,
            "total_cost_usd": 0.0,
            "total_api_duration_s": 0.0,
            "total_wall_time_s": 0.0,
            "total_audio_duration_s": 0.0,
            "avg_api_calls_per_video": 0,
            "avg_llm_calls_per_video": 0,
            "median_api_calls_per_video": 0,
            "median_llm_calls_per_video": 0,
            "avg_cost_per_video_usd": 0.0,
            "median_cost_per_video_usd": 0.0,
            "avg_processing_time_per_video_s": 0.0,
            "median_processing_time_per_video_s": 0.0,
            "avg_api_time_per_video_s": 0.0,
            "median_api_time_per_video_s": 0.0,
            "avg_answer_time_per_question_s": 0.0,
            "calls_by_type": {},
            "cost_by_type": {},
            "calls_by_stage": {},
            "cost_by_stage": {},
        }

    n_videos = len(video_stats)
    calls_per_video = [_numeric(s.get("total_calls", 0)) for s in video_stats]
    costs_per_video = [_numeric(s.get("total_cost_usd", 0.0)) for s in video_stats]
    api_times = [_numeric(s.get("total_duration_s", 0.0)) for s in video_stats]
    wall_times = [_numeric(s.get("wall_time_s", 0.0)) for s in video_stats]
    audio_durations = [
        _numeric(s.get("total_audio_duration_s", 0.0)) for s in video_stats
    ]

    total_calls = int(round(sum(calls_per_video)))
    total_cost = sum(costs_per_video)
    total_api_time = sum(api_times)
    total_wall_time = sum(wall_times)
    total_audio_duration = sum(audio_durations)

    avg_answer_time = sum(answer_times) / len(answer_times) if answer_times else 0.0

    return {
        "sample_videos": n_videos,
        "total_api_calls": total_calls,
        "total_llm_calls": total_calls,
        "total_cost_usd": round(total_cost, 6),
        "total_api_duration_s": round(total_api_time, 2),
        "total_wall_time_s": round(total_wall_time, 2),
        "total_audio_duration_s": round(total_audio_duration, 2),
        "avg_api_calls_per_video": round(total_calls / n_videos, 2),
        "avg_llm_calls_per_video": round(total_calls / n_videos, 2),
        "median_api_calls_per_video": round(_median(calls_per_video), 2),
        "median_llm_calls_per_video": round(_median(calls_per_video), 2),
        "avg_cost_per_video_usd": round(total_cost / n_videos, 6),
        "median_cost_per_video_usd": round(_median(costs_per_video), 6),
        "avg_processing_time_per_video_s": round(total_wall_time / n_videos, 2),
        "median_processing_time_per_video_s": round(_median(wall_times), 2),
        "avg_api_time_per_video_s": round(total_api_time / n_videos, 2),
        "median_api_time_per_video_s": round(_median(api_times), 2),
        "avg_audio_duration_per_video_s": round(total_audio_duration / n_videos, 2),
        "median_audio_duration_per_video_s": round(_median(audio_durations), 2),
        "avg_answer_time_per_question_s": round(avg_answer_time, 4),
        "calls_by_type": _round_count_map(_sum_map(video_stats, "calls_by_type")),
        "cost_by_type": _round_float_map(_sum_map(video_stats, "cost_by_type")),
        "duration_by_type": _round_float_map(
            _sum_map(video_stats, "duration_by_type"), digits=2
        ),
        "input_tokens_by_type": _round_count_map(
            _sum_map(video_stats, "input_tokens_by_type")
        ),
        "output_tokens_by_type": _round_count_map(
            _sum_map(video_stats, "output_tokens_by_type")
        ),
        "audio_duration_by_type": _round_float_map(
            _sum_map(video_stats, "audio_duration_by_type"), digits=2
        ),
        "calls_by_stage": _round_count_map(_sum_map(video_stats, "calls_by_stage")),
        "cost_by_stage": _round_float_map(_sum_map(video_stats, "cost_by_stage")),
        "duration_by_stage": _round_float_map(
            _sum_map(video_stats, "duration_by_stage"), digits=2
        ),
        "input_tokens_by_stage": _round_count_map(
            _sum_map(video_stats, "input_tokens_by_stage")
        ),
        "output_tokens_by_stage": _round_count_map(
            _sum_map(video_stats, "output_tokens_by_stage")
        ),
        "audio_duration_by_stage": _round_float_map(
            _sum_map(video_stats, "audio_duration_by_stage"), digits=2
        ),
        "calls_by_model": _round_count_map(_sum_map(video_stats, "calls_by_model")),
        "cost_by_model": _round_float_map(_sum_map(video_stats, "cost_by_model")),
        "calls_by_stage_and_type": _round_nested_count_map(
            _sum_nested_map(video_stats, "calls_by_stage_and_type")
        ),
        "cost_by_stage_and_type": _round_nested_float_map(
            _sum_nested_map(video_stats, "cost_by_stage_and_type")
        ),
    }


def _compute_grouped_accuracy(predictions: List[dict], *keys: str) -> Dict[str, dict]:
    """
    Group predictions by the first non-empty key and compute per-group accuracy.

    Returns:
        Dict with `accuracy` and `counts` maps.
    """
    groups: Dict[str, List[dict]] = {}

    for p in predictions:
        label = "unknown"
        for key in keys:
            value = p.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                label = value_str
                break
        groups.setdefault(label, []).append(p)

    return {
        "accuracy": {
            group: round(compute_accuracy(group_preds), 4)
            for group, group_preds in sorted(groups.items())
        },
        "counts": {
            group: len(group_preds)
            for group, group_preds in sorted(groups.items())
        },
    }


def compute_video_mme_accuracy(predictions: List[dict]) -> dict:
    """
    Compute VIDEO-MME accuracy with dataset-specific breakdowns.

    Breakdowns:
    - duration (long/medium/short)
    - domain
    - task_type
    """
    if not predictions:
        return {
            "overall": 0.0,
            "by_duration": {},
            "by_domain": {},
            "by_task_type": {},
            "counts": {"total": 0},
            "failed": 0,
        }

    overall = round(compute_accuracy(predictions), 4)
    duration_stats = _compute_grouped_accuracy(predictions, "duration")
    domain_stats = _compute_grouped_accuracy(predictions, "domain")
    task_type_stats = _compute_grouped_accuracy(predictions, "task_type", "question_type")

    return {
        "overall": overall,
        "by_duration": duration_stats["accuracy"],
        "by_domain": domain_stats["accuracy"],
        "by_task_type": task_type_stats["accuracy"],
        "counts": {
            "total": len(predictions),
            "duration": duration_stats["counts"],
            "domain": domain_stats["counts"],
            "task_type": task_type_stats["counts"],
        },
        "failed": sum(1 for p in predictions if p.get("predicted", -1) < 0),
    }


