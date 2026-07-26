"""
Evaluation pipeline orchestrator.

Processes local videos through the VideoGraph pipeline, answers
multiple-choice questions, and collects metrics.
"""

import json
import logging
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from videograph.config_loader import deep_update, resolve_evidence_construction

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def resolve_retrieval_settings(config: dict) -> dict:
    """Validate and normalize retrieval settings used by evaluation ablations."""
    from videograph.graph.models import EdgeType, NodeType

    section = config.get("retrieval", {})
    if not isinstance(section, dict):
        raise ValueError("retrieval must be a YAML mapping")

    top_k = section.get("top_k", 10)
    hop_expansion = section.get("hop_expansion", 2)
    hybrid_alpha = section.get("hybrid_alpha", 0.7)
    use_state_change_channel = section.get("use_state_change_channel", True)

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k must be a positive integer")
    if (
        isinstance(hop_expansion, bool)
        or not isinstance(hop_expansion, int)
        or hop_expansion < 0
    ):
        raise ValueError("retrieval.hop_expansion must be a non-negative integer")
    if isinstance(hybrid_alpha, bool) or not isinstance(hybrid_alpha, (int, float)):
        raise ValueError("retrieval.hybrid_alpha must be numeric")
    hybrid_alpha = float(hybrid_alpha)
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("retrieval.hybrid_alpha must be between 0 and 1")
    if not isinstance(use_state_change_channel, bool):
        raise ValueError("retrieval.use_state_change_channel must be true or false")

    return {
        "top_k": top_k,
        "hop_expansion": hop_expansion,
        "hybrid_alpha": hybrid_alpha,
        "allowed_node_types": _normalize_enum_list(
            section.get("allowed_node_types"),
            NodeType,
            "retrieval.allowed_node_types",
        ),
        "use_state_change_channel": use_state_change_channel,
        "expansion_edge_types": _normalize_enum_list(
            section.get("expansion_edge_types"),
            EdgeType,
            "retrieval.expansion_edge_types",
        ),
    }


def _normalize_enum_list(value, enum_type, field_name: str):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be null or a YAML list")

    normalized = []
    for item in value:
        raw = str(item).strip()
        try:
            member = enum_type(raw)
        except ValueError:
            try:
                member = enum_type[raw.upper()]
            except KeyError as exc:
                valid = ", ".join(member.value for member in enum_type)
                raise ValueError(
                    f"Unsupported value {raw!r} in {field_name}; expected one of: {valid}"
                ) from exc
        normalized.append(member.value)
    return normalized


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load the default configuration and merge an optional YAML overlay."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if config_path is None:
        return config

    overlay_path = Path(config_path).expanduser().resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {overlay_path}")
    with open(overlay_path, "r", encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle) or {}
    if not isinstance(overlay, dict):
        raise ValueError(f"Configuration root must be a YAML mapping: {overlay_path}")
    return deep_update(config, overlay)


def save_effective_config(config: dict, output_dir: str | Path) -> Path:
    """Persist the exact merged configuration and protect resumed runs from drift."""
    output_path = Path(output_dir) / "effective_config.yaml"
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
        if existing != config:
            raise RuntimeError(
                "The output directory contains a different effective_config.yaml. "
                "Use a new output directory to avoid mixing incompatible artifacts."
            )
        return output_path

    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path


def process_video(
    video_path: str,
    output_dir: str,
    video_id: str,
    config: dict,
    track_performance: bool = False,
    max_parallel_vision: Optional[int] = None,
) -> Optional[dict]:
    """
    Run the full VideoGraph pipeline on a local video.

    Skips if graph.json + embeddings.json already exist (resume support).

    Args:
        video_path: Path to local .mp4 file
        output_dir: Output directory for this video
        video_id: Video identifier
        config: VideoGraph config dict
        track_performance: Whether to track API calls/cost
        max_parallel_vision: Override for parallel vision workers

    Returns:
        Tracker stats dict if track_performance, else None
    """
    from videograph.cache.openai_cache import get_cache

    output_dir = Path(output_dir)
    graph_path = output_dir / "graph.json"
    embeddings_path = output_dir / "embeddings.json"

    # Resume support: skip only when both expected outputs exist and contain data.
    if (
        graph_path.exists() and graph_path.stat().st_size > 0 and
        embeddings_path.exists() and embeddings_path.stat().st_size > 0
    ):
        logger.info(f"  Skipping {video_id}: graph already exists")
        return None

    # Configure caching
    cache = get_cache()
    if track_performance:
        cache.enabled = False
    else:
        cache.enabled = True

    tracker_stats = None

    # Optionally wrap with tracker
    if track_performance:
        from .tracker import APITracker

        tracker = APITracker()
        with tracker:
            _run_pipeline(
                video_path,
                output_dir,
                video_id,
                config,
                max_parallel_vision=max_parallel_vision,
                tracker=tracker,
            )
        tracker_stats = tracker.get_stats().to_dict()
        tracker_stats["wall_time_s"] = tracker.get_wall_time()
        tracker_stats["video_id"] = video_id
        tracker_stats.update(_read_processing_artifact_stats(output_dir))
    else:
        _run_pipeline(video_path, output_dir, video_id, config, max_parallel_vision=max_parallel_vision)

    # Re-enable cache
    cache.enabled = True

    return tracker_stats


def cleanup_intermediates(output_dir: str):
    """
    Delete intermediate files after graph building to save storage.

    Keeps only: graph.json, embeddings.json, graph.graphml, metadata.json
    Deletes: keyframes/, clips/, audio.wav, audio_compressed.mp3, visual.json, transcript.json
    """
    output_dir = Path(output_dir)
    removed = 0

    # Remove directories
    for dirname in ["keyframes", "clips", "frames"]:
        dirpath = output_dir / dirname
        if dirpath.exists():
            shutil.rmtree(dirpath)
            removed += 1

    # Remove intermediate files
    for filename in ["audio.wav", "audio_compressed.mp3", "visual.json", "transcript.json"]:
        filepath = output_dir / filename
        if filepath.exists():
            filepath.unlink()
            removed += 1

    if removed > 0:
        logger.info(f"  Cleaned up intermediate files for {output_dir.name}")


def _read_processing_artifact_stats(output_dir: Path) -> dict:
    """Read lightweight metadata for the processed video, if available."""
    stats = {}

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            nested = metadata.get("metadata") if isinstance(metadata, dict) else {}
            if isinstance(nested, dict):
                stats["video_duration_s"] = nested.get("duration")
                stats["raw_video_file_size_bytes"] = nested.get("file_size_bytes")
        except Exception:
            pass

    graph_path = output_dir / "graph.json"
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            stats["graph_nodes"] = len(graph.get("nodes", []))
            stats["graph_edges"] = len(graph.get("edges", []))
        except Exception:
            pass

    return stats


def update_dataset_progress(
    output_dir: Path,
    dataset_name: str,
    progress_data: dict,
):
    """Update one dataset entry in the shared progress file."""
    progress_file = Path(output_dir) / "progress.json"
    current_progress = {}

    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                current_progress = json.load(f)
        except Exception:
            current_progress = {}

    current_progress[dataset_name] = progress_data

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(current_progress, f, indent=2)


def _tracker_stage(tracker, name: str):
    return tracker.stage(name) if tracker is not None else nullcontext()


def _run_pipeline(
    video_path: str,
    output_dir: Path,
    video_id: str,
    config: dict,
    max_parallel_vision: Optional[int] = None,
    tracker=None,
):
    """Run the full processing pipeline for a single video."""
    from videograph.video.adaptive_ingest import process_local_video_adaptive
    from videograph.visual.adaptive_processing import (
        analyze_adaptive_clips,
        update_adaptive_visual_json_with_ocr,
    )
    from videograph.video.transcribe import transcribe_audio
    from videograph.graph.builder import build_video_graph

    openai_config = config.get("openai", {})
    visual_config = config.get("visual", {})
    trans_config = config.get("transcription", {})
    processing_config = config.get("processing", {})
    evidence_construction = resolve_evidence_construction(config)
    openai_temperature = float(openai_config.get("temperature", 0.0))
    vision_workers = (
        max_parallel_vision
        if max_parallel_vision is not None
        else int(processing_config.get("max_parallel_vision", 5))
    )
    ocr_enabled = visual_config.get("ocr_enabled", True)
    prompt_style = visual_config.get("vision_prompt_style", "detailed")
    append_state_change_to_description = bool(
        visual_config.get("append_state_change_to_description", False)
    )

    # Step 1: Process local video (audio, frames, clips)
    logger.info(f"  [1/5] Processing local video: {video_id}")
    with _tracker_stage(tracker, "video_preprocessing"):
        process_local_video_adaptive(
            video_path=str(video_path),
            output_dir=str(output_dir),
            video_id=video_id,
            config=config,
        )

    # Step 2: Transcribe
    logger.info(f"  [2/5] Transcribing audio: {video_id}")
    audio_path = output_dir / "audio.wav"
    if audio_path.exists():
        with _tracker_stage(tracker, "transcription"):
            transcribe_audio(
                str(audio_path),
                output_dir=str(output_dir),
                model=config.get("openai", {}).get("transcription_model", "whisper-1"),
                language=trans_config.get("language"),
                timestamp_granularity=trans_config.get("timestamp_granularity", "segment"),
                filter_hallucinations=(
                    evidence_construction["transcript_filtering"]
                    and trans_config.get("filter_hallucinations", True)
                ),
                no_speech_threshold=trans_config.get("no_speech_threshold", 0.6),
                logprob_threshold=trans_config.get("logprob_threshold", -1.0),
                compression_ratio_threshold=trans_config.get(
                    "compression_ratio_threshold", 2.4
                ),
            )
    else:
        logger.warning(f"  Audio file not found for {video_id}, skipping transcription")

    # Step 3: Visual captioning
    logger.info(f"  [3/5] Visual captioning: {video_id}")
    with _tracker_stage(tracker, "visual_captioning"):
        analyze_adaptive_clips(
            str(output_dir),
            model=openai_config.get("vision_model", "gpt-4o"),
            prompt_style=prompt_style,
            temperature=openai_temperature,
            max_parallel=vision_workers,
            append_state_change_to_description=append_state_change_to_description,
            use_previous_clip_context=evidence_construction["cross_clip_continuity"],
        )

    # Step 4: OCR
    logger.info(f"  [4/5] OCR: {video_id}")
    if ocr_enabled:
        with _tracker_stage(tracker, "ocr"):
            try:
                update_adaptive_visual_json_with_ocr(
                    str(output_dir),
                    model=openai_config.get("vision_model", "gpt-4o"),
                    max_parallel=vision_workers,
                    gate_on_readable_text=evidence_construction["ocr_gating"],
                )
            except Exception as e:
                logger.warning(f"  OCR failed (non-fatal) for {video_id}: {e}")

    # Step 4b: EGC targeted re-perception with gated evidence write-back.
    reinforcement_config = config.get("graph", {}).get("reinforcement", {})
    if (
        evidence_construction["targeted_reperception"]
        and reinforcement_config.get("enabled", False)
    ):
        from videograph.graph.reinforce import reinforce_video_graph
        try:
            with _tracker_stage(tracker, "egc_targeted_reperception"):
                reinforce_video_graph(
                    str(output_dir),
                    text_model=config.get("openai", {}).get("text_model", "gpt-4o"),
                    vision_model=config.get("openai", {}).get("vision_model", "gpt-4o"),
                    max_probes=int(reinforcement_config.get("max_probes", 5)),
                    frames_per_probe=int(
                        reinforcement_config.get("frames_per_probe", 8)
                    ),
                    rebuild=False,
                )
        except Exception as e:
            logger.warning(
                f"  EGC targeted re-perception failed (non-fatal) for {video_id}: {e}"
            )

    # Step 4c: Multi-granularity — whole-video summary node (coarse level alongside
    # event-granular clips; retrieval routes by similarity, serving both fine and
    # holistic questions in one graph)
    if evidence_construction["whole_video_summary"]:
        from videograph.visual.adaptive_processing import append_video_summary_node
        try:
            with _tracker_stage(tracker, "summary_node"):
                append_video_summary_node(
                    str(output_dir),
                    model=config.get("openai", {}).get("text_model", "gpt-4o"),
                )
        except Exception as e:
            logger.warning(f"  Summary node failed (non-fatal) for {video_id}: {e}")

    # Step 5: Build graph
    logger.info(f"  [5/5] Building graph: {video_id}")
    with _tracker_stage(tracker, "graph_building"):
        build_video_graph(str(output_dir), config=config)


def answer_questions(
    questions: list,
    graphs_dir: str,
    config: dict,
    dataset_name: str = "",
    output_dir: Optional[str] = None,
    trace_records: Optional[list] = None,
    read_only_graphs: bool = False,
) -> list:
    """
    Answer a list of MC questions using pre-built graphs.

    Args:
        questions: List of Question objects
        graphs_dir: Directory containing per-video graph folders
        config: VideoGraph config dict
        dataset_name: Dataset name (for dataset-specific prompt hints)

    Returns:
        List of prediction dicts
    """
    from .mc_answer import answer_mc

    graphs_dir = Path(graphs_dir)
    openai_config = config.get("openai", {})
    text_model = openai_config.get("text_model", "gpt-4o")
    qa_temperature = float(openai_config.get("temperature", 0.0))
    retrieval = resolve_retrieval_settings(config)

    predictions = []

    for i, q in enumerate(questions):
        graph_path = graphs_dir / q.video_id / "graph.json"

        if not graph_path.exists():
            logger.warning(f"  Graph not found for {q.video_id}, skipping question {q.qid}")
            continue

        logger.info(f"  Answering question {i+1}/{len(questions)} (video={q.video_id}, qid={q.qid})")

        try:
            result = answer_mc(
                question=q.question,
                options=q.options,
                graph_path=str(graph_path),
                text_model=text_model,
                top_k=retrieval["top_k"],
                hop_expansion=retrieval["hop_expansion"],
                hybrid_alpha=retrieval["hybrid_alpha"],
                allowed_node_types=retrieval["allowed_node_types"],
                use_state_change_channel=retrieval["use_state_change_channel"],
                expansion_edge_types=retrieval["expansion_edge_types"],
                persist_visual_channel_embeddings=not read_only_graphs,
                temperature=qa_temperature,
                dataset=dataset_name,
            )

            predictions.append({
                "qid": q.qid,
                "video_id": q.video_id,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "predicted": result["predicted"],
                "raw_response": result["raw_response"],
                "answer_time_s": result["answer_time_s"],
                "question_type": q.question_type,
                "duration": getattr(q, "duration", None),
                "domain": getattr(q, "domain", None),
                "sub_category": getattr(q, "sub_category", None),
                "task_type": getattr(q, "task_type", None),
                "failure_reason": result.get("failure_reason"),
            })
            if trace_records is not None:
                trace_records.append({
                    "qid": q.qid,
                    "video_id": q.video_id,
                    "question": q.question,
                    "options": q.options,
                    "answer": q.answer,
                    "predicted": result["predicted"],
                    "raw_response": result["raw_response"],
                    "failure_reason": result.get("failure_reason"),
                    "retrieval_context": result.get("retrieval_context", ""),
                    "qa_user_prompt": result.get("qa_user_prompt", ""),
                })

        except Exception as e:
            logger.error(f"  Failed to answer question {q.qid}: {e}")
            predictions.append({
                "qid": q.qid,
                "video_id": q.video_id,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "predicted": -1,
                "raw_response": f"ERROR: {e}",
                "answer_time_s": 0.0,
                "question_type": q.question_type,
                "duration": getattr(q, "duration", None),
                "domain": getattr(q, "domain", None),
                "sub_category": getattr(q, "sub_category", None),
                "task_type": getattr(q, "task_type", None),
                "failure_reason": "answer_error",
            })
            if trace_records is not None:
                trace_records.append({
                    "qid": q.qid,
                    "video_id": q.video_id,
                    "question": q.question,
                    "options": q.options,
                    "answer": q.answer,
                    "predicted": -1,
                    "raw_response": f"ERROR: {e}",
                    "failure_reason": "answer_error",
                    "retrieval_context": "",
                    "qa_user_prompt": "",
                })

    return predictions


def save_predictions(predictions: list, preds_file: Path):
    """Persist predictions so QA can resume after interruption."""
    preds_file = Path(preds_file)
    preds_file.parent.mkdir(parents=True, exist_ok=True)
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def _qid_sort_key(qid) -> tuple:
    try:
        return (0, int(qid))
    except Exception:
        return (1, str(qid))


def _format_answer_label(options: list, answer_idx) -> str:
    try:
        idx = int(answer_idx)
    except Exception:
        return str(answer_idx)

    if idx < 0:
        return str(idx)

    if isinstance(options, list) and 0 <= idx < len(options):
        return f"{idx}: {options[idx]}"
    return str(idx)


def _prediction_to_trace_record(pred: dict) -> dict:
    return {
        "qid": pred.get("qid"),
        "video_id": pred.get("video_id"),
        "question": pred.get("question", ""),
        "options": pred.get("options", []),
        "answer": pred.get("answer"),
        "predicted": pred.get("predicted", -1),
        "raw_response": pred.get("raw_response", ""),
        "failure_reason": pred.get("failure_reason"),
        "retrieval_context": pred.get("retrieval_context", ""),
        "qa_user_prompt": pred.get("qa_user_prompt", ""),
    }


def save_dataset_qa_trace_markdown(
    dataset_name: str,
    trace_records: list,
    output_path: Path,
):
    """
    Save a per-dataset QA trace markdown grouped by video and question.

    Includes question text, options, correct/predicted answers, and the context
    passed to the LLM for answering.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for row in trace_records:
        video_id = str(row.get("video_id", "unknown"))
        grouped.setdefault(video_id, []).append(row)

    lines = [
        f"# QA Trace - {dataset_name}",
        "",
        f"Total questions: {len(trace_records)}",
        "",
    ]

    for video_id in sorted(grouped.keys()):
        lines.append(f"## Video `{video_id}`")
        lines.append("")
        video_rows = sorted(grouped[video_id], key=lambda r: _qid_sort_key(r.get("qid")))
        for row in video_rows:
            qid = row.get("qid")
            question = row.get("question", "")
            options = row.get("options", []) or []
            correct_label = _format_answer_label(options, row.get("answer"))
            predicted_label = _format_answer_label(options, row.get("predicted"))
            context_text = row.get("retrieval_context") or row.get("qa_user_prompt") or "<context unavailable>"

            lines.append(f"### QID `{qid}`")
            lines.append("")
            lines.append(f"Question: {question}")
            lines.append("")
            lines.append("Options:")
            for idx, opt in enumerate(options):
                lines.append(f"- {idx}: {opt}")
            lines.append("")
            lines.append(f"Correct answer: {correct_label}")
            lines.append(f"Predicted answer: {predicted_label}")
            lines.append(f"Raw response: `{row.get('raw_response', '')}`")
            if row.get("failure_reason"):
                lines.append(f"Failure reason: `{row.get('failure_reason')}`")
            lines.append("")
            lines.append("Context provided to LLM:")
            lines.append("```text")
            lines.append(str(context_text))
            lines.append("```")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def evaluate_dataset(
    dataset_name: str,
    questions: list,
    videos_dir: str,
    output_dir: str,
    config: dict,
    track_performance: bool = False,
    skip_processing: bool = False,
    cleanup: bool = False,
    max_parallel_vision: Optional[int] = None,
    graphs_root: Optional[str | Path] = None,
) -> dict:
    """
    Evaluate a single dataset end-to-end.

    Args:
        dataset_name: Name of the dataset (e.g. 'egoschema')
        questions: List of Question objects
        videos_dir: Path to video files folder
        output_dir: Output directory root
        config: VideoGraph config dict
        track_performance: Whether to track API metrics
        skip_processing: Skip video processing, only run QA
        cleanup: Delete intermediate files after each video to save storage
        max_parallel_vision: Override for parallel vision workers
        graphs_root: Read-only graph root containing a dataset subdirectory

    Returns:
        Dict with predictions, accuracy, and optional performance metrics
    """
    from .datasets import group_by_video
    from .metrics import (
        compute_egoschema_accuracy,
        compute_nextqa_accuracy,
        compute_video_mme_accuracy,
        compute_performance_metrics,
    )

    output_dir = Path(output_dir)
    if graphs_root is None:
        graphs_dir = output_dir / "graphs" / dataset_name
        graphs_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not skip_processing:
            raise ValueError("graphs_root requires skip_processing=True")
        graphs_dir = Path(graphs_root).expanduser().resolve() / dataset_name
        if not graphs_dir.is_dir():
            raise FileNotFoundError(
                f"Source graph directory not found for {dataset_name}: {graphs_dir}"
            )
    preds_file = output_dir / "predictions" / f"{dataset_name}.json"
    qa_trace_file = output_dir / "predictions" / f"{dataset_name}_qa_trace.md"

    preds_file.parent.mkdir(parents=True, exist_ok=True)

    videos_dir = Path(videos_dir)
    grouped = group_by_video(questions)

    logger.info(f"{'='*60}")
    logger.info(f"EVALUATING: {dataset_name}")
    logger.info(f"  Videos: {len(grouped)}, Questions: {len(questions)}")
    logger.info(f"{'='*60}")

    # --- Phase 1: Process videos & build graphs ---
    video_stats = []
    qa_stats = []
    failures = []
    
    # Check for existing failures to resume
    failures_file = output_dir / "failures.json"
    if failures_file.exists():
        try:
            with open(failures_file, "r") as f:
                failures = json.load(f)
        except Exception:
            pass

    if not skip_processing:
        logger.info(f"\nPhase 1: Processing videos for {dataset_name}")
        
        video_ids = list(grouped.keys())
        total_videos = len(video_ids)

        for idx, video_id in enumerate(video_ids):
            video_file = videos_dir / f"{video_id}.mp4"
            video_output = graphs_dir / video_id

            logger.info(f"\n[{idx+1}/{total_videos}] Processing video: {video_id}")

            ds_progress = {
                "dataset": dataset_name,
                "current_video": video_id,
                "index": idx + 1,
                "total": total_videos,
                "percentage": round(((idx + 1) / total_videos) * 100, 1),
                "phase": "processing",
                "updated_at": datetime.now().isoformat()
            }
            update_dataset_progress(output_dir, dataset_name, ds_progress)

            if not video_file.exists():
                error_msg = f"Video file not found: {video_file}"
                logger.warning(f"  {error_msg}")
                failures.append({
                    "video_id": video_id,
                    "dataset": dataset_name,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                with open(failures_file, "w") as f:
                    json.dump(failures, f, indent=2)
                continue

            try:
                stats = process_video(
                    video_path=str(video_file),
                    output_dir=str(video_output),
                    video_id=video_id,
                    config=config,
                    track_performance=track_performance,
                    max_parallel_vision=max_parallel_vision,
                )
                if stats is not None:
                    video_stats.append(stats)

                # Clean up intermediate files to save storage
                if cleanup:
                    cleanup_intermediates(str(video_output))

            except Exception as e:
                error_msg = str(e)
                logger.error(f"  Failed to process video {video_id}: {error_msg}")
                failures.append({
                    "video_id": video_id,
                    "dataset": dataset_name,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                with open(failures_file, "w") as f:
                    json.dump(failures, f, indent=2)

    # --- Phase 2: Answer questions ---
    logger.info(f"\nPhase 2: Answering questions for {dataset_name}")

    # Load existing predictions for resume
    existing_preds = []
    answered_qids = set()
    if preds_file.exists():
        try:
            with open(preds_file, "r", encoding="utf-8") as f:
                existing_preds = json.load(f)
            answered_qids = {(p["video_id"], p["qid"]) for p in existing_preds}
            logger.info(f"  Loaded {len(existing_preds)} existing predictions")
        except Exception:
            pass

    remaining = [q for q in questions if (q.video_id, q.qid) not in answered_qids]
    logger.info(f"  Remaining questions to answer: {len(remaining)}")

    all_preds = list(existing_preds)
    qa_traces = [_prediction_to_trace_record(p) for p in existing_preds]

    # Save incrementally per video so QA can resume mid-phase after interruption.
    remaining_by_video = group_by_video(remaining)
    total_videos_with_questions = len(remaining_by_video)
    total_questions = len(questions)
    answered_count = len(existing_preds)

    if remaining:
        first_video_id = next(iter(remaining_by_video))
        update_dataset_progress(output_dir, dataset_name, {
            "dataset": dataset_name,
            "current_video": first_video_id,
            "index": answered_count,
            "total": total_questions,
            "percentage": round((answered_count / total_questions) * 100, 1) if total_questions else 100.0,
            "phase": "answering",
            "updated_at": datetime.now().isoformat()
        })

    for idx, (video_id, video_questions) in enumerate(remaining_by_video.items(), start=1):
        logger.info(
            f"  Answering remaining questions for video {idx}/{total_videos_with_questions}: "
            f"{video_id} ({len(video_questions)} questions)"
        )

        update_dataset_progress(output_dir, dataset_name, {
            "dataset": dataset_name,
            "current_video": video_id,
            "index": answered_count,
            "total": total_questions,
            "percentage": round((answered_count / total_questions) * 100, 1) if total_questions else 100.0,
            "phase": "answering",
            "updated_at": datetime.now().isoformat()
        })

        if track_performance:
            from videograph.cache.openai_cache import get_cache

            from .tracker import APITracker

            cache = get_cache()
            cache.enabled = False
            qa_tracker = APITracker()
            try:
                with qa_tracker, qa_tracker.stage("question_answering"):
                    video_preds = answer_questions(
                        video_questions,
                        str(graphs_dir),
                        config,
                        dataset_name=dataset_name,
                        output_dir=None,  # Disable per-question tracking
                        trace_records=qa_traces,
                        read_only_graphs=graphs_root is not None,
                    )
            finally:
                cache.enabled = True

            stats = qa_tracker.get_stats().to_dict()
            stats["wall_time_s"] = qa_tracker.get_wall_time()
            stats["video_id"] = video_id
            qa_stats.append(stats)
        else:
            video_preds = answer_questions(
                video_questions,
                str(graphs_dir),
                config,
                dataset_name=dataset_name,
                output_dir=None,  # Disable per-question tracking
                trace_records=qa_traces,
                read_only_graphs=graphs_root is not None,
            )

        if not video_preds:
            continue

        all_preds.extend(video_preds)
        answered_count += len(video_preds)
        save_predictions(all_preds, preds_file)
        save_dataset_qa_trace_markdown(dataset_name, qa_traces, qa_trace_file)
        logger.info(f"  Saved {len(all_preds)} predictions to {preds_file}")
        update_dataset_progress(output_dir, dataset_name, {
            "dataset": dataset_name,
            "current_video": video_id,
            "index": answered_count,
            "total": total_questions,
            "percentage": round((answered_count / total_questions) * 100, 1) if total_questions else 100.0,
            "phase": "answering",
            "updated_at": datetime.now().isoformat()
        })

    update_dataset_progress(output_dir, dataset_name, {
        "dataset": dataset_name,
        "current_video": None,
        "index": answered_count,
        "total": total_questions,
        "percentage": round((answered_count / total_questions) * 100, 1) if total_questions else 100.0,
        "phase": "completed",
        "updated_at": datetime.now().isoformat()
    })

    # Ensure trace markdown exists even when there were no new questions to answer.
    save_dataset_qa_trace_markdown(dataset_name, qa_traces, qa_trace_file)

    if dataset_name == "egoschema":
        accuracy = compute_egoschema_accuracy(all_preds)
    elif dataset_name.startswith("video-mme-"):
        accuracy = compute_video_mme_accuracy(all_preds)
    else:
        accuracy = compute_nextqa_accuracy(all_preds)

    valid_preds = [p for p in all_preds if p["predicted"] >= 0]

    result = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total_predictions": len(all_preds),
        "valid_predictions": len(valid_preds),
        "failed_predictions": len(all_preds) - len(valid_preds),
    }

    if track_performance and video_stats:
        answer_times = [p["answer_time_s"] for p in valid_preds if p["answer_time_s"] > 0]
        result["performance"] = compute_performance_metrics(video_stats, answer_times)
        result["performance_video_stats"] = video_stats

    if track_performance and qa_stats:
        answer_times = [p["answer_time_s"] for p in valid_preds if p["answer_time_s"] > 0]
        result["qa_performance"] = compute_performance_metrics(qa_stats, answer_times)
        result["qa_performance_video_stats"] = qa_stats

    return result



