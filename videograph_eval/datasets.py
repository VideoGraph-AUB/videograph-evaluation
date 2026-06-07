"""
Dataset loaders for EgoSchema, NExT-QA, and VIDEO-MME benchmarks.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A single evaluation question."""
    qid: str
    video_id: str
    question: str
    options: List[str]
    answer: int          # ground-truth index (0-based)
    question_type: str   # raw type code from dataset
    duration: Optional[str] = None
    domain: Optional[str] = None
    sub_category: Optional[str] = None
    task_type: Optional[str] = None


# NExT-QA type groupings
NEXTQA_CAUSAL_TYPES = {"CW", "CH"}
NEXTQA_TEMPORAL_TYPES = {"TN", "TC"}
NEXTQA_DESCRIPTIVE_TYPES = {"DB", "DC", "DL", "DO"}


def get_nextqa_category(question_type: str) -> str:
    """Map a NExT-QA raw type code to its category."""
    if question_type in NEXTQA_CAUSAL_TYPES:
        return "causal"
    elif question_type in NEXTQA_TEMPORAL_TYPES:
        return "temporal"
    elif question_type in NEXTQA_DESCRIPTIVE_TYPES:
        return "descriptive"
    return "unknown"


def load_egoschema(
    json_path: str,
    videos_dir: str,
    max_videos: Optional[int] = None
) -> List[Question]:
    """
    Load EgoSchema subset questions.

    Args:
        json_path: Path to 'EgoSchema Subset QAs.json'
        videos_dir: Path to 'EgoSchema Subset Videos/' folder
        max_videos: Limit number of unique videos (for debugging)

    Returns:
        List of Question objects
    """
    json_path = Path(json_path)
    videos_dir = Path(videos_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    seen_videos = set()

    for entry in data:
        q_uid = entry["q_uid"]
        video_file = videos_dir / f"{q_uid}.mp4"

        if max_videos is not None and q_uid not in seen_videos:
            if len(seen_videos) >= max_videos:
                continue

        if not video_file.exists():
            logger.warning(f"Video not found, skipping: {video_file}")
            continue

        seen_videos.add(q_uid)

        options = [
            entry["option 0"],
            entry["option 1"],
            entry["option 2"],
            entry["option 3"],
            entry["option 4"],
        ]

        questions.append(Question(
            qid=q_uid,
            video_id=q_uid,
            question=entry["question"],
            options=options,
            answer=int(entry["_answer"]),
            question_type="ego",
        ))

    logger.info(f"Loaded {len(questions)} EgoSchema questions from {len(seen_videos)} videos")
    return questions


def load_nextqa(
    csv_path: str,
    videos_dir: str,
    max_videos: Optional[int] = None
) -> List[Question]:
    """
    Load NExT-QA questions from a CSV file (test or val split).

    Args:
        csv_path: Path to NExT-QA CSV file
        videos_dir: Path to folder containing .mp4 video files
        max_videos: Limit number of unique videos (for debugging)

    Returns:
        List of Question objects
    """
    csv_path = Path(csv_path)
    videos_dir = Path(videos_dir)

    questions = []
    seen_videos = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _build_safe_csv_reader(f)
        for row in reader:
            video_id = str(row["video"])
            video_file = videos_dir / f"{video_id}.mp4"

            if max_videos is not None and video_id not in seen_videos:
                if len(seen_videos) >= max_videos:
                    continue

            if not video_file.exists():
                logger.warning(f"Video not found, skipping: {video_file}")
                continue

            seen_videos.add(video_id)

            options = [
                row["a0"],
                row["a1"],
                row["a2"],
                row["a3"],
                row["a4"],
            ]

            questions.append(Question(
                qid=str(row["qid"]),
                video_id=video_id,
                question=row["question"],
                options=options,
                answer=int(row["answer"]),
                question_type=row["type"],
            ))

    logger.info(f"Loaded {len(questions)} NExT-QA questions from {len(seen_videos)} videos")
    return questions


def _get_row_value(row: dict, *candidate_keys: str) -> str:
    """Get the first matching row value using case-insensitive key lookup."""
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in candidate_keys:
        value = lowered.get(key.strip().lower())
        if value is not None:
            return str(value).strip()
    return ""


def _build_safe_csv_reader(file_obj):
    """
    Build a CSV DictReader with stable quote parsing.

    We only sniff delimiter (comma/tab). Quoting is forced to Python's
    standard CSV behavior so doubled quotes inside fields are parsed correctly.
    """
    sample = file_obj.read(2048)
    file_obj.seek(0)

    delimiter = ","
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=",\t")
        if sniffed.delimiter in {",", "\t"}:
            delimiter = sniffed.delimiter
    except csv.Error:
        pass

    return csv.DictReader(
        file_obj,
        delimiter=delimiter,
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
    )


def _collect_video_mme_options(row: dict) -> List[str]:
    """Collect and order option columns (e.g., optionA..optionD or option0..option4)."""
    option_items = []
    for key, value in row.items():
        key_str = str(key).strip()
        low = key_str.lower()
        if not low.startswith("option"):
            continue
        suffix = low[len("option"):].strip()
        if not suffix:
            continue
        option_items.append((suffix, str(value or "").strip()))

    def _option_sort_key(suffix: str):
        if suffix.isdigit():
            return (0, int(suffix))
        if len(suffix) == 1 and suffix.isalpha():
            return (1, ord(suffix.upper()) - ord("A"))
        return (2, suffix)

    option_items.sort(key=lambda x: _option_sort_key(x[0]))
    return [value for _, value in option_items if value]


def _parse_video_mme_answer(raw_answer: str, options: List[str]) -> Optional[int]:
    """Parse VIDEO-MME answer value into 0-based option index."""
    raw = str(raw_answer or "").strip()
    if not raw or not options:
        return None

    # Numeric indices (support both 0-based and 1-based)
    if raw.isdigit():
        val = int(raw)
        if 0 <= val < len(options):
            return val
        if 1 <= val <= len(options):
            return val - 1

    # Letter labels (A/B/C/...)
    if len(raw) == 1 and raw.isalpha():
        idx = ord(raw.upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx

    # Full option text fallback
    raw_lower = raw.lower()
    for idx, option in enumerate(options):
        if option.strip().lower() == raw_lower:
            return idx

    return None


def load_video_mme(
    csv_path: str,
    videos_dir: str,
    max_videos: Optional[int] = None
) -> List[Question]:
    """
    Load VIDEO-MME questions from a CSV file.

    Supports the provided schema with columns such as:
    video_id, duration, domain, sub_category, videoID, question_id,
    task_type, question, optionA..optionD, answer.
    """
    csv_path = Path(csv_path)
    videos_dir = Path(videos_dir)

    questions = []
    seen_videos = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _build_safe_csv_reader(f)
        for row in reader:
            video_id = _get_row_value(row, "videoID", "video_id", "video")
            if not video_id:
                logger.warning(f"Missing video ID in row, skipping: {row}")
                continue

            video_file = videos_dir / f"{video_id}.mp4"
            if max_videos is not None and video_id not in seen_videos:
                if len(seen_videos) >= max_videos:
                    continue

            if not video_file.exists():
                logger.warning(f"Video not found, skipping: {video_file}")
                continue

            seen_videos.add(video_id)

            options = _collect_video_mme_options(row)
            answer = _parse_video_mme_answer(_get_row_value(row, "answer"), options)
            if answer is None:
                qid_for_log = _get_row_value(row, "question_id", "qid")
                logger.warning(
                    f"Could not parse VIDEO-MME answer for qid={qid_for_log}, "
                    f"answer='{_get_row_value(row, 'answer')}', skipping"
                )
                continue

            qid = _get_row_value(row, "question_id", "qid")
            if not qid:
                qid = f"{video_id}-{len(questions)}"

            question_type = _get_row_value(row, "task_type") or "video_mme"

            questions.append(Question(
                qid=qid,
                video_id=video_id,
                question=_get_row_value(row, "question"),
                options=options,
                answer=answer,
                question_type=question_type,
                duration=_get_row_value(row, "duration"),
                domain=_get_row_value(row, "domain"),
                sub_category=_get_row_value(row, "sub_category"),
                task_type=_get_row_value(row, "task_type"),
            ))

    logger.info(f"Loaded {len(questions)} VIDEO-MME questions from {len(seen_videos)} videos")
    return questions


def group_by_video(questions: List[Question]) -> Dict[str, List[Question]]:
    """Group questions by video_id."""
    groups: Dict[str, List[Question]] = {}
    for q in questions:
        groups.setdefault(q.video_id, []).append(q)
    return groups


