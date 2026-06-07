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

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load VideoGraph configuration."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


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
                )
            except Exception as e:
                logger.warning(f"  OCR failed (non-fatal) for {video_id}: {e}")

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
    retrieval_config = config.get("retrieval", {})
    top_k = retrieval_config.get("top_k", 10)
    hop_expansion = retrieval_config.get("hop_expansion", 2)
    hybrid_alpha = float(retrieval_config.get("hybrid_alpha", 0.7))

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
                top_k=top_k,
                hop_expansion=hop_expansion,
                hybrid_alpha=hybrid_alpha,
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
    graphs_dir = output_dir / "graphs" / dataset_name
    preds_file = output_dir / "predictions" / f"{dataset_name}.json"
    qa_trace_file = output_dir / "predictions" / f"{dataset_name}_qa_trace.md"

    graphs_dir.mkdir(parents=True, exist_ok=True)
    preds_file.parent.mkdir(parents=True, exist_ok=True)

    videos_dir = Path(videos_dir)
    grouped = group_by_video(questions)

    logger.info(f"{'='*60}")
    logger.info(f"EVALUATING: {dataset_name}")
    logger.info(f"  Videos: {len(grouped)}, Questions: {len(questions)}")
    logger.info(f"{'='*60}")

    # --- Phase 1: Process videos & build graphs ---
    video_stats = []
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

        video_preds = answer_questions(
            video_questions,
            str(graphs_dir),
            config,
            dataset_name=dataset_name,
            output_dir=None,  # Disable per-question tracking
            trace_records=qa_traces,
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

    return result



