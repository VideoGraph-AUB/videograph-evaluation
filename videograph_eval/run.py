"""
CLI entry point for the VideoGraph evaluation pipeline.

Usage:
    python -m videograph_eval.run \
        --data-dir /data \
        --output-dir /results/v0 \
        --version v0

    # With performance tracking (disables cache):
    python -m videograph_eval.run \
        --data-dir /data \
        --output-dir /results/v0 \
        --version v0 \
        --track-performance

    # Debug run (1 video per dataset):
    python -m videograph_eval.run \
        --data-dir /data \
        --output-dir /results/test \
        --version test \
        --max-videos 1

    # Run only specific datasets:
    python -m videograph_eval.run \
        --data-dir /data \
        --output-dir /results/v0 \
        --version v0 \
        --datasets egoschema nextqa-test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from videograph_eval.datasets import load_egoschema, load_nextqa, load_video_mme
from videograph.config_loader import resolve_evidence_construction
from videograph_eval.pipeline import evaluate_dataset, load_config, save_effective_config
from videograph_eval.report import generate_report, save_results_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset folder and file names
DATASET_CONFIG = {
    "egoschema": {
        "videos_folder": "EgoSchema Subset Videos",
        "qa_file": "EgoSchema Subset QAs.json",
        "loader": "egoschema",
    },
    "nextqa-test": {
        "videos_folder": "NExT-QA Test Videos",
        "qa_file": "NExT-QA Test QAs.csv",
        "loader": "nextqa",
    },
    "nextqa-val": {
        "videos_folder": "NExT-QA Val Videos",
        "qa_file": "NExT-QA Val QAs.csv",
        "loader": "nextqa",
    },
    "video-mme-long": {
        "videos_folder": "VIDEO-MME Long Videos",
        "qa_file": "VIDEO-MME Long QAs.csv",
        "loader": "video-mme",
    },
    "video-mme-medium": {
        "videos_folder": "VIDEO-MME Medium Videos",
        "qa_file": "VIDEO-MME Medium QAs.csv",
        "loader": "video-mme",
    },
    "video-mme-short": {
        "videos_folder": "VIDEO-MME Short Videos",
        "qa_file": "VIDEO-MME Short QAs.csv",
        "loader": "video-mme",
    },
}

ALL_DATASETS = list(DATASET_CONFIG.keys())


def main():
    parser = argparse.ArgumentParser(
        description="VideoGraph Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory containing video folders and QA files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for graphs, predictions, and results",
    )
    parser.add_argument(
        "--version", default="Unspecified",
        help="Version label for the report (default: Unspecified)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML overlay merged onto config/default.yaml",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip-processing", action="store_true",
        help="Skip video processing + graph building, only run QA",
    )
    parser.add_argument(
        "--track-performance", action="store_true",
        help="Disable cache and track API calls, cost, and timing",
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max videos per dataset (for debugging)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete intermediate files (frames, clips, audio) after each video to save storage",
    )
    parser.add_argument(
        "--max-parallel-vision", type=int, default=None,
        help="Override max parallel workers for visual captioning/OCR (default: config value)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    effective_config_path = save_effective_config(config, output_dir)
    evidence_construction = resolve_evidence_construction(config)
    default_vision_workers = int(config.get("processing", {}).get("max_parallel_vision", 5))
    effective_vision_workers = (
        args.max_parallel_vision
        if args.max_parallel_vision is not None
        else default_vision_workers
    )

    logger.info("=" * 60)
    logger.info("VIDEOGRAPH EVALUATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Data dir:     {data_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Version:      {args.version}")
    logger.info(f"  Config:       {args.config or 'config/default.yaml'}")
    logger.info(
        f"  EGC:          {'ON' if evidence_construction['enabled'] else 'OFF'}"
    )
    logger.info(f"  Datasets:     {args.datasets}")
    logger.info(f"  Performance:  {'ON' if args.track_performance else 'OFF (cache enabled)'}")
    logger.info(f"  Max videos:   {args.max_videos or 'all'}")
    logger.info(f"  Skip processing: {args.skip_processing}")
    logger.info(f"  Cleanup:      {'ON' if args.cleanup else 'OFF'}")
    logger.info(f"  Max parallel vision: {effective_vision_workers}")
    logger.info("=" * 60)

    all_results = {}
    start_time = time.time()

    for ds_name in args.datasets:
        ds_config = DATASET_CONFIG[ds_name]
        videos_dir = data_dir / ds_config["videos_folder"]
        qa_path = data_dir / ds_config["qa_file"]

        if not videos_dir.exists():
            logger.error(f"Videos folder not found: {videos_dir}")
            continue
        if not qa_path.exists():
            logger.error(f"QA file not found: {qa_path}")
            continue

        # Load dataset
        if ds_config["loader"] == "egoschema":
            questions = load_egoschema(
                str(qa_path), str(videos_dir), max_videos=args.max_videos
            )
        elif ds_config["loader"] == "nextqa":
            questions = load_nextqa(
                str(qa_path), str(videos_dir), max_videos=args.max_videos
            )
        else:
            questions = load_video_mme(
                str(qa_path), str(videos_dir), max_videos=args.max_videos
            )

        if not questions:
            logger.warning(f"No questions loaded for {ds_name}, skipping")
            continue

        # Evaluate
        result = evaluate_dataset(
            dataset_name=ds_name,
            questions=questions,
            videos_dir=str(videos_dir),
            output_dir=str(output_dir),
            config=config,
            track_performance=args.track_performance,
            skip_processing=args.skip_processing,
            cleanup=args.cleanup,
            max_parallel_vision=args.max_parallel_vision,
        )

        all_results[ds_name] = result

        # Log accuracy summary
        acc = result.get("accuracy", {})
        logger.info(f"\n{ds_name} accuracy: {acc.get('overall', 0):.2%}")

    # Save combined results
    total_time = time.time() - start_time
    all_results["_meta"] = {
        "version": args.version,
        "total_time_s": round(total_time, 2),
        "datasets": args.datasets,
        "track_performance": args.track_performance,
        "max_videos": args.max_videos,
        "max_parallel_vision": effective_vision_workers,
        "config": str(effective_config_path),
        "evidence_construction": evidence_construction,
    }

    results_path = output_dir / "results.json"
    save_results_json(all_results, str(results_path))

    report_path = output_dir / "report.md"
    generate_report(all_results, str(report_path), version=args.version)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Results:    {results_path}")
    logger.info(f"  Report:     {report_path}")

    for ds_name in args.datasets:
        if ds_name in all_results:
            acc = all_results[ds_name].get("accuracy", {}).get("overall", 0)
            logger.info(f"  {ds_name}: {acc:.2%}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()


