#!/usr/bin/env python3
"""
run_nextgqa_grounding.py  (paper-comparable denominators + seed verification
                                + graph coverage + visual-only + expansion gain)

Capstone experiment: compare flat vs graph-hop retrieval on NExT-GQA.

Three conditions (all use top_k=7):
  flat     — hop_expansion=0  (seed-only retrieval, no graph traversal)
  graph-1  — hop_expansion=1  (seeds + 1-hop neighbours)
  graph-2  — hop_expansion=2  (seeds + 2-hop neighbourhood)

Metric design (see "Which metrics are fair?" at the bottom of this docstring)
-----------------------------------------------------------------------
We NEVER compute a global union span
[min(node.start), max(node.end)] and use it as a single predicted interval
for IoP or Acc@GQA.  That unfairly penalises graph-hop conditions because
expansion adds temporally scattered nodes, making the union span arbitrarily
wide and the IoP artificially low.

Instead every grounding metric is computed PER NODE, and we report:
  top1_iop / top1_iou      — IoP/IoU of the single highest-ranked node
  best_node_iop / _iou     — max IoP/IoU across every retrieved node
  iog_coverage             — fraction of the total gold evidence covered
                             by the union of all retrieved node spans
  hit_at_1/5/7_seed        — does any of the first K *seed* nodes (is_expanded=False) overlap gold?
  hit_at_seed_budget       — hit_at_k_seed where k = current --top-k (always valid regardless of top_k)
  hit_any                  — does ANY retrieved node (seeds + expanded) overlap?
  hit_beyond_seed_budget   — hit_any AND NOT hit_at_seed_budget
                             (any extra evidence beyond the seed budget; can be >0 for flat)
  hit_expanded_only        — NOT hit_at_seed_budget AND at least one is_expanded=True node overlaps gold
                             (true graph expansion gain; structurally 0 for flat/hop=0)

Primary Acc@GQA metrics
  acc_gqa_top1             — answer_correct AND top1_iop >= 0.5
  acc_gqa_oracle_retrieved — answer_correct AND best_node_iop >= 0.5

Paper-comparable denominators 
  accuracy_all_split       — n_correct / n_total_split_questions
                             (missing-graph questions score 0; matches paper)
  acc_gqa_all_with_gold    — n_acc_gqa_top1 / n_total_with_gold
                             (all questions with gold intervals, not just graph-covered)

acc_gqa_top1 is the fair primary metric for the flat-vs-graph comparison.
acc_gqa_oracle_retrieved is an ORACLE upper-bound (uses the best node in
hindsight); it shows the retrieval ceiling, not the localization ability.

Windows PowerShell run examples
  # Val split — full run
  python run_nextgqa_grounding.py `
    --data-dir data `
    --graphs-dir results\\v1\\graphs\\nextqa-val `
    --output-dir results\\nextgqa_val `
    --split val

  # Test split — full run
  python run_nextgqa_grounding.py `
    --data-dir data `
    --graphs-dir results\\v1_test\\graphs\\nextqa-test `
    --output-dir results\\nextgqa_test `
    --split test

  # Debug: 20 questions only (val)
  python run_nextgqa_grounding.py `
    --data-dir data `
    --graphs-dir results\\v1\\graphs\\nextqa-val `
    --output-dir results\\nextgqa_debug `
    --split val `
    --max-questions 20

  # Retrieval-only with seed-order verification
  python run_nextgqa_grounding.py `
    --data-dir data `
    --graphs-dir results\\v1\\graphs\\nextqa-val `
    --output-dir results\\nextgqa_retrieval `
    --split val `
    --retrieval-only `
    --verify-seed-order

  # Visual-only grounding variant
  python run_nextgqa_grounding.py `
    --data-dir data `
    --graphs-dir results\\v1\\graphs\\nextqa-val `
    --output-dir results\\nextgqa_visual_only `
    --split val `
    --node-types visual-only `
    --retrieval-only

Which metrics are fair for the flat-vs-graph comparison?
---------------------------------------------------------
PAPER-COMPARABLE (denominators cover ALL split questions):
  accuracy_all_split    — n_correct / n_total_split_questions
  acc_gqa_all_with_gold — n_acc_gqa_top1 / n_total_with_gold

FAIR for flat-vs-graph (denominators equal across conditions):
  accuracy              — QA accuracy, denom = n_answered (graph-covered subset)
  acc_gqa_top1          — uses only the top-1 seed node; denom = n_answered
  hit_at_seed_budget    — seed-only hit at k=top_k; identical across conditions
  hit_at_1/5/7_seed     — seed-only hits at fixed K; fair only when top_k >= K
  iog_coverage          — fair; more nodes can only improve recall

FAIR but influenced by set size:
  hit_any               — includes expanded nodes; show alongside hit_at_seed_budget
  hit_beyond_seed_budget — hit_any AND NOT hit_at_seed_budget (general extra evidence)
  hit_expanded_only     — true graph expansion gain; always 0 for flat

ORACLE / upper-bound (use for analysis, not primary comparison):
  best_node_iop/iou          — max over ALL nodes; graph-hop has more nodes
  acc_gqa_oracle_retrieved   — same issue; label as oracle in the report
  best_cluster_iop           — window around top-1; can grow with expanded nodes
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Retrieval conditions ─────────────────────────────────────────────────────

CONDITIONS: Dict[str, dict] = {
    "flat":    {"top_k": 7, "hop_expansion": 0},
    "graph-1": {"top_k": 7, "hop_expansion": 1},
    "graph-2": {"top_k": 7, "hop_expansion": 2},
}

# Threshold for Hits@K, acc_gqa_top1, acc_gqa_oracle_retrieved
GROUNDING_THRESHOLD = 0.5

# Cluster window: nodes within this many seconds of the top-1 node's span
# are merged into the "best cluster" prediction
CLUSTER_WINDOW_S = 5.0

# Sanity check: warn when graph max-end and gsub duration differ by this many seconds
TIMEBASE_WARN_S = 3.0

# Node types used in grounding metrics (transcript+visual mode)
GROUNDING_NODE_TYPES = {"TranscriptNode", "VisualNode"}

# Node types used in visual-only mode
VISUAL_ONLY_NODE_TYPES = {"VisualNode"}


# ── §1  Multi-interval grounding helpers ─────────────────────────────────────
#
# Gold evidence is always  [[s1, e1], [s2, e2], ...]  (list-of-lists).
# Predicted interval is a single [p_start, p_end] from one retrieved node.
#
# Official NExT-GQA evaluation protocol (TempGQA/eval_ground.py):
#   For each gold interval independently compute IoP and IoU, then take MAX.
#   This matches the official papers (FrozenGQA, SeViLA, TempGQA).
#
# IoG coverage is a separate recall metric that intentionally merges/sums
# over all gold intervals — kept as-is.

def interval_overlap(a_start: float, a_end: float,
                     b_start: float, b_end: float) -> float:
    """Raw overlap in seconds between two intervals. Zero if disjoint."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def total_gold_length(gold_intervals: list) -> float:
    """Sum of durations of all gold intervals."""
    return sum(max(0.0, g[1] - g[0]) for g in gold_intervals)


def overlap_with_gold(pred_start: float, pred_end: float,
                      gold_intervals: list) -> float:
    """
    Total overlap (seconds) of the predicted interval with the gold evidence.
    Sums overlaps across all gold intervals.
    Used ONLY by _iog_coverage (recall metric), NOT by iop_against_gold or
    iou_against_gold (which use max-per-segment per the official protocol).
    """
    return sum(
        interval_overlap(pred_start, pred_end, g[0], g[1])
        for g in gold_intervals
    )


def iop_against_gold(pred_start: float, pred_end: float,
                     gold_intervals: list) -> float:
    """
    Intersection-over-Prediction against multi-interval gold.
    Matches official NExT-GQA protocol: compute IoP against each gold interval
    independently, return the maximum.
    IoP_i = overlap(pred, gold_i) / pred_duration
    Range [0, 1].  Returns 0 if prediction has zero duration or no gold.
    """
    pred_dur = pred_end - pred_start
    if pred_dur <= 0.0 or not gold_intervals:
        return 0.0
    best = 0.0
    for g in gold_intervals:
        ov = interval_overlap(pred_start, pred_end, g[0], g[1])
        iop = ov / pred_dur
        if iop > best:
            best = iop
    return min(1.0, best)


def iou_against_gold(pred_start: float, pred_end: float,
                     gold_intervals: list) -> float:
    """
    Intersection-over-Union against multi-interval gold.
    Matches official NExT-GQA protocol: compute IoU against each gold interval
    independently, return the maximum.
    IoU_i = overlap(pred, gold_i) / (pred_dur + gold_i_dur - overlap)
    Returns 0 if prediction has zero duration or no gold.
    """
    pred_dur = pred_end - pred_start
    if pred_dur <= 0.0 or not gold_intervals:
        return 0.0
    best = 0.0
    for g in gold_intervals:
        ov       = interval_overlap(pred_start, pred_end, g[0], g[1])
        gold_dur = max(0.0, g[1] - g[0])
        union    = pred_dur + gold_dur - ov
        iou      = ov / union if union > 0.0 else 0.0
        if iou > best:
            best = iou
    return best


# ── §1b  Grounding node filter ───────────────────────────────────────────────
#
# Only TranscriptNode and VisualNode have meaningful fine-grained temporal
# spans.  TopicNode and EntityNode span the whole video or have no temporal
# annotation, so including them in localization metrics would inflate IoP/IoU
# and make flat-vs-graph comparisons unfair.
#
# Pass node_types={"VisualNode"} for the visual-only sensitivity variant.
# This filter is applied ONLY to grounding metrics.  The LLM answering context
# (built from evidence_nodes inside mc_answer.py) is unaffected.

def _filter_grounding_nodes(
    evidence_nodes: list,
    video_duration: Optional[float] = None,
    node_types: Optional[set] = None,
) -> list:
    """
    Keep only nodes of the selected types with valid, non-trivial temporal spans.

    node_types defaults to GROUNDING_NODE_TYPES (TranscriptNode + VisualNode).
    Pass {"VisualNode"} for the visual-only grounding variant.

    Excluded:
      * node_type not in node_types
      * start or end is None
      * end <= start
      * end == inf
      * span covers >= 95% of the video (whole-video placeholder spans)
    """
    allowed = node_types if node_types is not None else GROUNDING_NODE_TYPES
    out = []
    for n in evidence_nodes:
        if n.get("node_type") not in allowed:
            continue
        s = n.get("start")
        e = n.get("end")
        if s is None or e is None:
            continue
        try:
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        if e == float("inf"):
            continue
        if video_duration is not None and video_duration > 0 and (e - s) >= 0.95 * video_duration:
            continue
        out.append(n)
    return out


# ── §2  Per-node and set-level grounding metrics ─────────────────────────────

def _node_iop(node: dict, gold_intervals: list) -> float:
    """IoP of one retrieved node dict against all gold intervals."""
    n_start = float(node.get("start") or 0.0)
    n_end   = float(node.get("end")   or 0.0)
    return iop_against_gold(n_start, n_end, gold_intervals)


def _node_iou(node: dict, gold_intervals: list) -> float:
    """IoU of one retrieved node dict against all gold intervals."""
    n_start = float(node.get("start") or 0.0)
    n_end   = float(node.get("end")   or 0.0)
    return iou_against_gold(n_start, n_end, gold_intervals)


def _hit_at_k_seed(
    evidence_nodes: list,
    gold_intervals: list,
    k: int,
    threshold: float = GROUNDING_THRESHOLD,
) -> bool:
    """
    True if any of the first k *seed* nodes (is_expanded=False) has IoP >= threshold.

    Filters to seed-only nodes before slicing at [:k], so this metric is
    unaffected by expanded nodes regardless of top_k.  Seeds are already
    sorted by score desc, so seed_nodes[:k] gives the top-k seeds.
    """
    if not gold_intervals:
        return False
    seed_nodes = [n for n in evidence_nodes if not n.get("is_expanded", False)]
    for node in seed_nodes[:k]:
        if _node_iop(node, gold_intervals) >= threshold:
            return True
    return False


def _hit_any(
    evidence_nodes: list,
    gold_intervals: list,
    threshold: float = GROUNDING_THRESHOLD,
) -> bool:
    """
    True if ANY retrieved node (seeds + expanded) has IoP >= threshold.
    Graph-hop conditions have more nodes, so this metric is influenced by
    set size — show it alongside hit_at_seed_budget to isolate the expansion gain.
    """
    if not gold_intervals:
        return False
    for node in evidence_nodes:
        if _node_iop(node, gold_intervals) >= threshold:
            return True
    return False


def _iog_coverage(evidence_nodes: list, gold_intervals: list) -> float:
    """
    Fraction of total gold evidence covered by the union of retrieved node spans.

    IoG = overlap(union_of_retrieved_spans, gold) / total_gold_length
    Range [0, 1].  Returns 0 if no gold or no valid node spans.
    More nodes from graph expansion can only increase this metric.
    """
    gold_len = total_gold_length(gold_intervals)
    if gold_len <= 0.0 or not gold_intervals:
        return 0.0

    # Collect valid node spans
    spans = []
    for n in evidence_nodes:
        ns = float(n.get("start") or 0.0)
        ne = float(n.get("end")   or 0.0)
        if ne > ns:
            spans.append((ns, ne))

    if not spans:
        return 0.0

    # Merge overlapping spans
    spans.sort()
    merged: List[List[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Sum overlaps of merged spans with gold (non-overlapping merged => safe to sum)
    total_overlap = sum(
        overlap_with_gold(ms, me, gold_intervals)
        for ms, me in merged
    )
    return min(1.0, total_overlap / gold_len)


def _top1_metrics(evidence_nodes: list, gold_intervals: list) -> Tuple[float, float]:
    """
    IoP and IoU of the single highest-ranked retrieved node (evidence_nodes[0]).

    This is the PRIMARY fair localization metric for flat-vs-graph comparison:
    the top-1 node is the same seed regardless of hop_expansion.
    Returns (0.0, 0.0) when there are no nodes or no gold.
    """
    if not evidence_nodes or not gold_intervals:
        return 0.0, 0.0
    top1 = evidence_nodes[0]
    return _node_iop(top1, gold_intervals), _node_iou(top1, gold_intervals)


def _best_cluster_metrics(
    evidence_nodes: list,
    gold_intervals: list,
    window_s: float = CLUSTER_WINDOW_S,
) -> Tuple[float, float]:
    """
    IoP and IoU of a temporally compact cluster built around the top-1 node.

    Algorithm (no gold labels used):
      1. Anchor = span of evidence_nodes[0] (highest-ranked seed).
      2. Add any other node whose span overlaps or is within window_s of
         the current cluster boundary.
      3. Cluster span = [min(starts), max(ends)] of included nodes.
      4. Compute IoP and IoU of that span against gold.

    This gives a "tight" prediction in the neighbourhood of the best seed.
    It can grow slightly with expanded nodes but is constrained by window_s.
    Returns (0.0, 0.0) when no valid nodes or no gold.
    """
    if not evidence_nodes or not gold_intervals:
        return 0.0, 0.0

    # Anchor: top-1 node
    anchor_start = float(evidence_nodes[0].get("start") or 0.0)
    anchor_end   = float(evidence_nodes[0].get("end")   or 0.0)
    if anchor_end <= anchor_start:
        return 0.0, 0.0

    c_start, c_end = anchor_start, anchor_end

    for node in evidence_nodes[1:]:
        ns = float(node.get("start") or 0.0)
        ne = float(node.get("end")   or 0.0)
        if ne <= ns:
            continue
        # Include if within window_s of current cluster bounds
        if ns <= c_end + window_s and ne >= c_start - window_s:
            c_start = min(c_start, ns)
            c_end   = max(c_end,   ne)

    return (
        iop_against_gold(c_start, c_end, gold_intervals),
        iou_against_gold(c_start, c_end, gold_intervals),
    )


# ── §3  Sanity check — time-base alignment ───────────────────────────────────

def _graph_max_end(graph_path: Path) -> Optional[float]:
    """
    Read graph.json and return the maximum node end time in seconds.
    Returns None on error or if no valid end times are found.
    """
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", {})
        if isinstance(nodes, dict):
            node_vals = nodes.values()
        elif isinstance(nodes, list):
            node_vals = nodes
        else:
            return None
        valid_ends = [
            float(v["end"])
            for v in node_vals
            if isinstance(v, dict)
            and v.get("end") is not None
            and v["end"] != float("inf")
        ]
        return max(valid_ends) if valid_ends else None
    except Exception:
        return None


# ── §4  Data loading helpers ─────────────────────────────────────────────────

def _get_answer_index(row: dict) -> int:
    """
    NExT-GQA val.csv stores 'answer' as answer TEXT (not an integer).
    Matches it against a0–a4 and returns the 0-based index.
    Returns -1 if no match is found.
    """
    answer_text = str(row.get("answer", "")).strip().lower()
    for i in range(5):
        opt = str(row.get(f"a{i}", "")).strip().lower()
        if opt == answer_text:
            return i
    return -1


def _load_nextgqa_split(
    path: Path,
    video_id_filter: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> List[dict]:
    """
    Load a NExT-GQA CSV split (val or test).
    Columns: video_id, frame_count, width, height, question,
             answer (TEXT), qid, type, a0–a4
    """
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if video_id_filter and row["video_id"] != video_id_filter:
                continue
            rows.append(dict(row))
            if max_questions is not None and len(rows) >= max_questions:
                break
    return rows


def _load_gsub(path: Path) -> dict:
    """
    Load gsub_val.json or gsub_test.json.
    Schema: { "<video_id>": { "duration": int, "fps": float,
              "location": { "<qid>": [[s,e], ...] } } }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_by_video(rows: list) -> Dict[str, list]:
    groups: Dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["video_id"], []).append(row)
    return groups


# ── §5  Per-condition runner ──────────────────────────────────────────────────

def run_condition(
    cond_name: str,
    top_k: int,
    hop_expansion: int,
    rows: list,
    graphs_dir: Path,
    gsub: dict,
    output_dir: Path,
    text_model: str,
    hybrid_alpha: float,
    retrieval_only: bool = False,
    split: str = "val",
    # Paper-comparable denominators (passed from main after loading CSV+gsub)
    n_total_split_questions: int = 0,
    n_total_with_gold: int = 0,
    # Grounding node filter mode
    node_types_mode: str = "transcript+visual",
) -> dict:
    """Run one retrieval condition and return aggregate summary dict."""
    from videograph_eval.mc_answer import GraphAnswerSession

    cond_dir = output_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cond_dir / "detailed.jsonl"

    # Resolve active node types for grounding filter
    active_node_types = (
        VISUAL_ONLY_NODE_TYPES if node_types_mode == "visual-only"
        else GROUNDING_NODE_TYPES
    )

    mode_tag = "RETRIEVAL-ONLY" if retrieval_only else "FULL"
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"CONDITION : {cond_name}  (top_k={top_k}, hop_expansion={hop_expansion}, "
        f"mode={mode_tag}, node_types={node_types_mode})"
    )
    logger.info("=" * 60)

    grouped  = _group_by_video(rows)
    n_videos = len(grouped)

    # ── Denominator counters ─────────────────────────────────────────────────
    n_answered              = 0   # questions where we got a valid predicted answer
    n_grounded              = 0   # questions with a gold interval in gsub
    n_missing_graph         = 0   # questions skipped because graph.json absent
    n_missing_gold_interval = 0   # questions answered but with no gsub entry
    n_answer_parse_failures = 0   # questions where answer parsing returned -1
    n_correct               = 0

    # ── Graph coverage counters ──────────────────────────────────────────────
    n_videos_with_graph    = 0
    n_videos_missing_graph = 0

    # ── Grounding accumulators (denominator = n_grounded) ───────────────────
    sum_hit1_seed         = 0
    sum_hit5_seed         = 0
    sum_hit7_seed         = 0
    sum_hit_seed_budget        = 0    # hit_at_seed_budget: seeds only, k=top_k
    sum_hit_beyond_seed_budget = 0    # hit_any AND NOT hit_at_seed_budget (general extra-evidence)
    sum_hit_any                = 0
    sum_hit_expanded_only      = 0    # NOT hit_at_seed_budget AND expanded node hits (true graph gain)
    sum_best_iop          = 0.0
    sum_best_iou          = 0.0
    sum_top1_iop          = 0.0
    sum_top1_iou          = 0.0
    sum_cluster_iop       = 0.0
    sum_cluster_iou       = 0.0
    sum_iog               = 0.0
    # Threshold-based metrics matching official NExT-GQA paper columns
    sum_iop_at_03         = 0
    sum_iop_at_05         = 0
    sum_iou_at_03         = 0
    sum_iou_at_05         = 0

    # ── Acc@GQA accumulators ─────────────────────────────────────────────────
    # answered-subset denominator (n_answered): existing behaviour
    n_acc_gqa_top1   = 0
    n_acc_gqa_oracle = 0

    # ── Node count accumulators (denominator = n_answered) ───────────────────
    sum_n_nodes    = 0
    sum_n_seeds    = 0
    sum_n_expanded = 0

    # ── Per question-type ────────────────────────────────────────────────────
    by_type: Dict[str, dict] = {}

    processed = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for vid_idx, (video_id, video_rows) in enumerate(grouped.items()):
            graph_path = graphs_dir / video_id / "graph.json"

            if not graph_path.exists():
                logger.warning(
                    f"  [{vid_idx+1}/{n_videos}] {video_id} - graph not found, "
                    f"skipping {len(video_rows)} questions"
                )
                n_missing_graph        += len(video_rows)
                n_videos_missing_graph += 1
                continue

            n_videos_with_graph += 1
            logger.info(
                f"  [{vid_idx+1}/{n_videos}] {video_id} ({len(video_rows)} questions)"
            )

            # ── §6 Time-base sanity check ────────────────────────────────────
            vid_gsub      = gsub.get(video_id, {})
            gsub_duration = vid_gsub.get("duration")
            graph_max_end = _graph_max_end(graph_path)
            if (
                gsub_duration is not None
                and graph_max_end is not None
                and abs(graph_max_end - gsub_duration) > TIMEBASE_WARN_S
            ):
                logger.warning(
                    f"    TIME-BASE MISMATCH: gsub duration={gsub_duration}s  "
                    f"graph max_end={graph_max_end:.1f}s  "
                    f"diff={abs(graph_max_end - gsub_duration):.1f}s  "
                    f"grounding metrics may be unreliable for {video_id}"
                )

            vid_location = vid_gsub.get("location", {})
            vid_fps      = vid_gsub.get("fps")

            try:
                session = GraphAnswerSession(
                    graph_path=str(graph_path),
                    top_k=top_k,
                    hop_expansion=hop_expansion,
                    text_model=text_model,
                    hybrid_alpha=hybrid_alpha,
                    dataset=f"nextqa-{split}",
                )
            except Exception as exc:
                logger.error(f"    Failed to load session: {exc}")
                n_missing_graph        += len(video_rows)
                n_videos_with_graph    -= 1     # undo; session load failed
                n_videos_missing_graph += 1
                continue

            for row in video_rows:
                qid    = str(row["qid"])
                q_text = row["question"]
                q_type = row.get("type", "")
                options = [row.get(f"a{i}", "") for i in range(5)]
                gold_idx = _get_answer_index(row)

                # Gold temporal evidence: always [[s,e],...] from raw JSON
                gold_intervals: list = vid_location.get(qid, [])
                if gold_intervals and isinstance(gold_intervals[0], (int, float)):
                    gold_intervals = [gold_intervals]   # defensive normalisation

                has_gold = len(gold_intervals) > 0

                # ── retrieve (and optionally answer) ─────────────────────────
                if retrieval_only:
                    try:
                        evidence_nodes = session.retrieve(q_text)
                    except Exception as exc:
                        logger.error(f"    qid={qid} retrieve failed: {exc}")
                        evidence_nodes = []
                    predicted      = None
                    answer_correct = None
                    result         = {}
                else:
                    try:
                        result = session.answer(q_text, options)
                    except Exception as exc:
                        logger.error(f"    qid={qid} answer failed: {exc}")
                        result = {
                            "predicted":         -1,
                            "raw_response":      f"ERROR: {exc}",
                            "answer_time_s":     0.0,
                            "failure_reason":    "answer_error",
                            "retrieval_context": "",
                            "qa_user_prompt":    "",
                            "evidence_nodes":    [],
                        }
                    predicted      = result["predicted"]
                    evidence_nodes = result.get("evidence_nodes", [])

                    # Track parse failures (gold_idx == -1 means answer text didn't match)
                    if gold_idx < 0:
                        n_answer_parse_failures += 1
                    answer_correct = bool(predicted == gold_idx) if gold_idx >= 0 else False
                    if answer_correct:
                        n_correct += 1

                n_answered += 1
                if not has_gold:
                    n_missing_gold_interval += 1

                # ── Node counts ───────────────────────────────────────────────
                n_nodes_q    = len(evidence_nodes)
                n_seeds_q    = sum(1 for n in evidence_nodes if not n.get("is_expanded", False))
                n_expanded_q = n_nodes_q - n_seeds_q
                sum_n_nodes    += n_nodes_q
                sum_n_seeds    += n_seeds_q
                sum_n_expanded += n_expanded_q

                # ── §3  Per-node grounding scores ────────────────────────────
                # grounding_nodes: only allowed types with valid temporal spans.
                # node_types_mode controls which types are included.
                # evidence_nodes (all types) is only used for LLM context
                # (already consumed inside session.answer() above).
                grounding_nodes = _filter_grounding_nodes(
                    evidence_nodes,
                    video_duration=gsub_duration,
                    node_types=active_node_types,
                )

                per_node = []
                for node in evidence_nodes:
                    niop = _node_iop(node, gold_intervals) if has_gold else 0.0
                    niou = _node_iou(node, gold_intervals) if has_gold else 0.0
                    per_node.append({
                        "node_id":          node["node_id"],
                        "node_type":        node["node_type"],
                        "start":            node.get("start"),
                        "end":              node.get("end"),
                        "score":            node.get("score"),
                        "is_expanded":      node.get("is_expanded", False),
                        "expansion_source": node.get("expansion_source"),
                        "iop":              round(niop, 4),
                        "iou":              round(niou, 4),
                    })

                # Grounding scalars use grounding_nodes only
                grounding_iops = [
                    _node_iop(n, gold_intervals) for n in grounding_nodes
                ] if has_gold else []
                grounding_ious = [
                    _node_iou(n, gold_intervals) for n in grounding_nodes
                ] if has_gold else []

                # Scalar grounding metrics (denominator: n_grounded)
                if has_gold and grounding_nodes:
                    hit1            = _hit_at_k_seed(grounding_nodes, gold_intervals, 1)
                    hit5            = _hit_at_k_seed(grounding_nodes, gold_intervals, 5)
                    hit7            = _hit_at_k_seed(grounding_nodes, gold_intervals, 7)
                    hit_seed_budget = _hit_at_k_seed(grounding_nodes, gold_intervals, top_k)
                    hit_any         = _hit_any(grounding_nodes, gold_intervals)
                    best_iop        = max(grounding_iops)
                    best_iou        = max(grounding_ious)
                    top1_iop, top1_iou        = _top1_metrics(grounding_nodes, gold_intervals)
                    cl_iop,   cl_iou          = _best_cluster_metrics(grounding_nodes, gold_intervals)
                    iog                        = _iog_coverage(grounding_nodes, gold_intervals)
                    # General extra-evidence metric: any node (seed or expanded) hit when seeds alone didn't
                    hit_beyond_seed_budget = bool(hit_any) and not bool(hit_seed_budget)
                    # True expansion gain: seed budget missed AND at least one *expanded* node hits
                    hit_expanded_only = (
                        not bool(hit_seed_budget)
                        and any(
                            _node_iop(n, gold_intervals) >= GROUNDING_THRESHOLD
                            for n in grounding_nodes
                            if n.get("is_expanded", False)
                        )
                    )
                elif has_gold:
                    # Gold exists but no valid grounding node: score 0, keep in denominator
                    hit1 = hit5 = hit7 = hit_seed_budget = hit_any = False
                    hit_beyond_seed_budget = False
                    hit_expanded_only = False
                    best_iop = best_iou = 0.0
                    top1_iop = top1_iou = 0.0
                    cl_iop   = cl_iou   = 0.0
                    iog = 0.0
                else:
                    hit1 = hit5 = hit7 = hit_seed_budget = hit_any = None   # undefined: no gold
                    hit_beyond_seed_budget = None
                    hit_expanded_only = None
                    best_iop = best_iou = None
                    top1_iop = top1_iou = None
                    cl_iop   = cl_iou   = None
                    iog = None

                # ── Acc@GQA variants ─────────────────────────────────────────
                if retrieval_only or answer_correct is None:
                    # QA not run — Acc@GQA is undefined
                    acc_gqa_top1   = None
                    acc_gqa_oracle = None
                else:
                    # PRIMARY: top-1 node only (fair across conditions)
                    acc_gqa_top1 = (
                        answer_correct and top1_iop is not None and top1_iop >= GROUNDING_THRESHOLD
                    )
                    # ORACLE: best achievable node (upper-bound; favours graph-hop)
                    acc_gqa_oracle = (
                        answer_correct and best_iop is not None and best_iop >= GROUNDING_THRESHOLD
                    )

                # ── JSONL record ─────────────────────────────────────────────
                record = {
                    # Identity
                    "condition":          cond_name,
                    "retrieval_only":     retrieval_only,
                    "node_types_mode":    node_types_mode,
                    "video_id":           video_id,
                    "qid":                qid,
                    "question_type":      q_type,
                    # QA (null when retrieval_only)
                    "question":           q_text,
                    "options":            options,
                    "gold_answer_text":   row.get("answer", ""),
                    "gold_answer_index":  gold_idx,
                    "predicted_answer":   predicted,
                    "raw_response":       result.get("raw_response") if not retrieval_only else None,
                    "answer_correct":     answer_correct,
                    "failure_reason":     result.get("failure_reason") if not retrieval_only else None,
                    "answer_time_s":      result.get("answer_time_s", 0.0) if not retrieval_only else None,
                    # Node counts
                    "n_nodes":            n_nodes_q,
                    "n_seed_nodes":       n_seeds_q,
                    "n_expanded_nodes":   n_expanded_q,
                    # Gold grounding
                    "has_gold_grounding": has_gold,
                    "gold_intervals":     gold_intervals,
                    "video_duration_s":   gsub_duration,
                    "video_fps":          vid_fps,
                    # Retrieved nodes with per-node scores
                    "retrieved_nodes":    per_node,
                    # §3 Retrieval recall metrics (denom = n_grounded)
                    "hit_at_1_seed":      hit1,
                    "hit_at_5_seed":      hit5,
                    "hit_at_7_seed":      hit7,
                    "hit_at_seed_budget":       hit_seed_budget,         # k=top_k, seeds only
                    "hit_beyond_seed_budget":  hit_beyond_seed_budget,  # hit_any AND NOT hit_at_seed_budget
                    "hit_any":                 hit_any,
                    "hit_expanded_only":       hit_expanded_only,       # expanded node actually hit
                    "best_node_iop":      round(best_iop,  4) if best_iop  is not None else None,
                    "best_node_iou":      round(best_iou,  4) if best_iou  is not None else None,
                    "iog_coverage":       round(iog,        4) if iog       is not None else None,
                    # §4 Localization metrics (tight predictions)
                    "top1_iop":           round(top1_iop,  4) if top1_iop  is not None else None,
                    "top1_iou":           round(top1_iou,  4) if top1_iou  is not None else None,
                    "best_cluster_iop":   round(cl_iop,    4) if cl_iop    is not None else None,
                    "best_cluster_iou":   round(cl_iou,    4) if cl_iou    is not None else None,
                    # §4 Threshold-based metrics (official NExT-GQA paper columns)
                    "iop_at_0.3":         bool(top1_iop >= 0.3) if top1_iop is not None else None,
                    "iop_at_0.5":         bool(top1_iop >= 0.5) if top1_iop is not None else None,
                    "iou_at_0.3":         bool(top1_iou >= 0.3) if top1_iou is not None else None,
                    "iou_at_0.5":         bool(top1_iou >= 0.5) if top1_iou is not None else None,
                    # §4 Grounding node filter counts
                    "n_grounding_nodes":  len(grounding_nodes),
                    # §4 Acc@GQA variants (null when retrieval_only)
                    "acc_gqa_top1":               acc_gqa_top1,
                    "acc_gqa_oracle_retrieved":   acc_gqa_oracle,
                }

                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_f.flush()

                # ── Running aggregates ───────────────────────────────────────
                if has_gold:
                    n_grounded            += 1
                    sum_hit1_seed         += int(hit1)
                    sum_hit5_seed         += int(hit5)
                    sum_hit7_seed         += int(hit7)
                    sum_hit_seed_budget        += int(hit_seed_budget)
                    sum_hit_beyond_seed_budget += int(hit_beyond_seed_budget)
                    sum_hit_any                += int(hit_any)
                    sum_hit_expanded_only      += int(hit_expanded_only)
                    sum_best_iop          += best_iop
                    sum_best_iou          += best_iou
                    sum_top1_iop          += top1_iop
                    sum_top1_iou          += top1_iou
                    sum_cluster_iop       += cl_iop
                    sum_cluster_iou       += cl_iou
                    sum_iog               += iog
                    # Threshold-based metrics (official paper columns)
                    sum_iop_at_03         += int(top1_iop >= 0.3)
                    sum_iop_at_05         += int(top1_iop >= 0.5)
                    sum_iou_at_03         += int(top1_iou >= 0.3)
                    sum_iou_at_05         += int(top1_iou >= 0.5)
                if acc_gqa_top1:
                    n_acc_gqa_top1   += 1
                if acc_gqa_oracle:
                    n_acc_gqa_oracle += 1

                # Per question-type
                if q_type not in by_type:
                    by_type[q_type] = {
                        "n": 0, "n_correct": 0,
                        "n_grounded": 0,
                        "sum_hit7_seed": 0, "sum_hit_any": 0,
                        "sum_hit_expanded_only": 0,
                        "sum_best_iop": 0.0, "sum_top1_iop": 0.0,
                        "sum_top1_iou": 0.0,
                        "sum_iop_at_03": 0, "sum_iop_at_05": 0,
                        "sum_iou_at_03": 0, "sum_iou_at_05": 0,
                        "sum_iog": 0.0,
                        "n_acc_gqa_top1": 0, "n_acc_gqa_oracle": 0,
                    }
                t = by_type[q_type]
                t["n"] += 1
                if answer_correct:   t["n_correct"]             += 1
                if has_gold:
                    t["n_grounded"]            += 1
                    t["sum_hit7_seed"]         += int(hit7)
                    t["sum_hit_any"]           += int(hit_any)
                    t["sum_hit_expanded_only"] += int(hit_expanded_only)
                    t["sum_best_iop"]          += best_iop
                    t["sum_top1_iop"]          += top1_iop
                    t["sum_top1_iou"]          += top1_iou
                    t["sum_iop_at_03"]         += int(top1_iop >= 0.3)
                    t["sum_iop_at_05"]         += int(top1_iop >= 0.5)
                    t["sum_iou_at_03"]         += int(top1_iou >= 0.3)
                    t["sum_iou_at_05"]         += int(top1_iou >= 0.5)
                    t["sum_iog"]               += iog
                if acc_gqa_top1:     t["n_acc_gqa_top1"]        += 1
                if acc_gqa_oracle:   t["n_acc_gqa_oracle"]      += 1

                processed += 1
                if processed % 100 == 0:
                    logger.info(f"    Progress: {processed}/{len(rows)}")

    # ── Build summary dict ────────────────────────────────────────────────────
    def _r(num, den) -> float:
        return round(num / den, 4) if den > 0 else 0.0

    ng = n_grounded   # shorthand denominator for grounding metrics
    na = n_answered   # shorthand denominator for QA metrics (graph-covered subset)
    nt = n_total_split_questions   # paper-comparable QA denominator
    nw = n_total_with_gold         # paper-comparable Acc@GQA denominator

    summary = {
        "condition":           cond_name,
        "retrieval_only":      retrieval_only,
        "node_types_mode":     node_types_mode,
        "top_k":               top_k,
        "hop_expansion":       hop_expansion,

        # ── Denominator counters ─────────────────────────────────────────────
        # n_total_split_questions and n_total_with_gold are the paper-comparable
        # denominators (include ALL questions in the split, even missing-graph ones).
        # n_answered and n_grounded are the graph-covered-subset denominators.
        "n_total_split_questions":  nt,
        "n_total_with_gold":        nw,
        "n_answered":               na,      # graph-covered subset
        "n_grounded":               ng,      # gold intervals found for graph-covered Qs
        "n_missing_graph":          n_missing_graph,
        "n_missing_gold_interval":  n_missing_gold_interval,
        "n_answer_parse_failures":  n_answer_parse_failures,

        # ── Graph coverage ───────────────────────────────────────────────────
        "n_total_videos":           n_videos,
        "n_videos_with_graph":      n_videos_with_graph,
        "n_videos_missing_graph":   n_videos_missing_graph,
        "graph_coverage_pct":       round(n_videos_with_graph / max(n_videos, 1), 4),

        # ── Node counts (denominator = n_answered) ───────────────────────────
        "avg_nodes":           _r(sum_n_nodes,    na),
        "avg_seed_nodes":      _r(sum_n_seeds,    na),
        "avg_expanded_nodes":  _r(sum_n_expanded, na),

        # ── QA accuracy ──────────────────────────────────────────────────────
        # accuracy_all_split:  PAPER-COMPARABLE  (denom = all split questions;
        #                       missing-graph questions implicitly score 0)
        # accuracy:            graph-covered subset  (denom = n_answered)
        "accuracy_all_split":  None if retrieval_only else _r(n_correct, nt),
        "accuracy":            None if retrieval_only else _r(n_correct, na),

        # ── §3 Retrieval recall (denominator = n_grounded) ───────────────────
        "hit_at_1_seed":       _r(sum_hit1_seed,         ng),
        "hit_at_5_seed":       _r(sum_hit5_seed,         ng),
        "hit_at_7_seed":       _r(sum_hit7_seed,         ng),
        # hit_at_seed_budget: seeds only, k=top_k (fair across any --top-k value)
        "hit_at_seed_budget":       _r(sum_hit_seed_budget,        ng),
        # hit_beyond_seed_budget: general "extra evidence" — hit_any AND NOT hit_at_seed_budget
        # Can be >0 even for flat (when budget < total seeds returned)
        "hit_beyond_seed_budget":   _r(sum_hit_beyond_seed_budget, ng),
        "n_hit_beyond_seed_budget": sum_hit_beyond_seed_budget,
        "hit_any":                  _r(sum_hit_any,                ng),
        # hit_expanded_only: true graph expansion gain — seed budget missed AND expanded node hits
        # Always 0 for flat (no expanded nodes)
        "hit_expanded_only":        _r(sum_hit_expanded_only,      ng),
        "n_hit_expanded_only":      sum_hit_expanded_only,
        "mean_best_iop":       _r(sum_best_iop,           ng),
        "mean_best_iou":       _r(sum_best_iou,           ng),
        "mean_iog_coverage":   _r(sum_iog,                ng),

        # ── §4 Localization (denominator = n_grounded) ───────────────────────
        "mean_top1_iop":       _r(sum_top1_iop,    ng),
        "mean_top1_iou":       _r(sum_top1_iou,    ng),
        "mean_cluster_iop":    _r(sum_cluster_iop, ng),
        "mean_cluster_iou":    _r(sum_cluster_iou, ng),

        # §4 Threshold-based metrics — match official paper columns
        #    (denominator = n_grounded; based on top-1 node IoP/IoU)
        "iop_at_0.3":          _r(sum_iop_at_03, ng),
        "iop_at_0.5":          _r(sum_iop_at_05, ng),
        "iou_at_0.3":          _r(sum_iou_at_03, ng),
        "iou_at_0.5":          _r(sum_iou_at_05, ng),

        # ── Acc@GQA ──────────────────────────────────────────────────────────
        # acc_gqa_all_with_gold:  PAPER-COMPARABLE  (denom = n_total_with_gold;
        #                          all questions with gold, even missing-graph ones)
        # acc_gqa_top1:           graph-covered subset  (denom = n_answered)
        "acc_gqa_all_with_gold":         None if retrieval_only else _r(n_acc_gqa_top1, nw),
        "acc_gqa_top1":                  None if retrieval_only else _r(n_acc_gqa_top1, na),
        "acc_gqa_oracle_retrieved":      None if retrieval_only else _r(n_acc_gqa_oracle, na),

        # ── Per type ─────────────────────────────────────────────────────────
        # All grounding metrics use v["n_grounded"] as denominator.
        # QA metrics use v["n"] (graph-covered questions of that type).
        "by_type": {
            qt: {
                "n":                     v["n"],
                "n_grounded":            v["n_grounded"],
                "accuracy":              _r(v["n_correct"],              v["n"]),
                "acc_gqa_top1":          _r(v["n_acc_gqa_top1"],         v["n"]),
                "acc_gqa_oracle":        _r(v["n_acc_gqa_oracle"],       v["n"]),
                # Localization (denom = n_grounded per type)
                "mean_top1_iop":         _r(v["sum_top1_iop"],           v["n_grounded"]),
                "mean_top1_iou":         _r(v["sum_top1_iou"],           v["n_grounded"]),
                "iop_at_0.3":            _r(v["sum_iop_at_03"],          v["n_grounded"]),
                "iop_at_0.5":            _r(v["sum_iop_at_05"],          v["n_grounded"]),
                "iou_at_0.3":            _r(v["sum_iou_at_03"],          v["n_grounded"]),
                "iou_at_0.5":            _r(v["sum_iou_at_05"],          v["n_grounded"]),
                # Retrieval (denom = n_grounded per type)
                "hit_at_7_seed":         _r(v["sum_hit7_seed"],          v["n_grounded"]),
                "hit_any":               _r(v["sum_hit_any"],             v["n_grounded"]),
                "hit_expanded_only":     _r(v["sum_hit_expanded_only"],   v["n_grounded"]),
                "mean_iog_coverage":     _r(v["sum_iog"],                 v["n_grounded"]),
                "mean_best_iop":         _r(v["sum_best_iop"],            v["n_grounded"]),
            }
            for qt, v in by_type.items()
        },
        "jsonl_path": str(jsonl_path),
    }

    def _pct(v): return f"{v:.2%}" if v is not None else "N/A"
    logger.info(f"  {cond_name} summary ({mode_tag}, node_types={node_types_mode}):")
    logger.info(
        f"    Graph coverage:        {n_videos_with_graph}/{n_videos} videos "
        f"({summary['graph_coverage_pct']:.2%})"
    )
    logger.info(
        f"    Avg nodes (seed/exp):  {summary['avg_nodes']:.1f}  "
        f"({summary['avg_seed_nodes']:.1f} seeds + {summary['avg_expanded_nodes']:.1f} expanded)"
    )
    if not retrieval_only:
        logger.info(
            f"    Accuracy (all split):  {_pct(summary['accuracy_all_split'])}  "
            f"[PAPER denom={nt}]"
        )
        logger.info(
            f"    Accuracy (answered):   {_pct(summary['accuracy'])}  "
            f"[graph-covered denom={na}]"
        )
    logger.info(
        f"    Hit@1/5/7 (seed only): "
        f"{_pct(summary['hit_at_1_seed'])} / "
        f"{_pct(summary['hit_at_5_seed'])} / "
        f"{_pct(summary['hit_at_7_seed'])}"
    )
    logger.info(
        f"    Hit@seed_budget (k={top_k}): {_pct(summary['hit_at_seed_budget'])}"
    )
    logger.info(f"    Hit@any (all nodes):   {_pct(summary['hit_any'])}")
    logger.info(
        f"    Hit beyond budget:     {_pct(summary['hit_beyond_seed_budget'])}  "
        f"({sum_hit_beyond_seed_budget} questions)  [hit_any AND NOT hit@seed_budget]"
    )
    logger.info(
        f"    Hit expanded only:     {_pct(summary['hit_expanded_only'])}  "
        f"({sum_hit_expanded_only} questions)  [expanded node hit AND NOT hit@seed_budget]"
    )
    logger.info(
        f"    Mean top1 IoP/IoU:     "
        f"{summary['mean_top1_iop']:.3f} / {summary['mean_top1_iou']:.3f}"
    )
    logger.info(
        f"    Mean best IoP/IoU:     "
        f"{summary['mean_best_iop']:.3f} / {summary['mean_best_iou']:.3f}"
    )
    logger.info(f"    Mean IoG coverage:     {summary['mean_iog_coverage']:.3f}")
    if not retrieval_only:
        logger.info(
            f"    Acc@GQA (all w/gold):  {_pct(summary['acc_gqa_all_with_gold'])}  "
            f"[PAPER denom={nw}]"
        )
        logger.info(
            f"    Acc@GQA top1:          {_pct(summary['acc_gqa_top1'])}  "
            f"[graph-covered denom={na}]  (PRIMARY fair)"
        )
        logger.info(
            f"    Acc@GQA oracle-retr:   {_pct(summary['acc_gqa_oracle_retrieved'])}  "
            f"[oracle upper-bound]"
        )
    logger.info(
        f"    Denominators: answered={na}  grounded={ng}  "
        f"missing_graph={n_missing_graph}  "
        f"missing_gold={n_missing_gold_interval}  "
        f"parse_fail={n_answer_parse_failures}"
    )

    return summary


# ── §6  Seed-order verification ───────────────────────────────────────────────

def _verify_seed_order(all_results: dict, output_dir: Path) -> dict:
    """
    Verify that seed node IDs and hit_at_seed_budget are identical across all
    conditions for each question.

    GraphRetriever is supposed to return the same top-k seeds in the same order
    regardless of hop_expansion.  If this invariant holds, both the seed node
    ID list and hit_at_seed_budget should be IDENTICAL across flat / graph-1 /
    graph-2 for every question.

    Reads the per-condition JSONL files and compares per-question values.
    Returns a verification summary dict that is added to _meta in summary.json.
    """
    cond_names = [k for k in all_results if not k.startswith("_")]
    if len(cond_names) < 2:
        logger.info("Seed-order verification skipped: fewer than 2 conditions.")
        return {"skipped": True, "reason": "fewer than 2 conditions"}

    # per_cond maps cond -> {qid -> {"hit": bool|None, "seed_ids": list}}
    per_cond: Dict[str, dict] = {}
    for cond in cond_names:
        jsonl_path = Path(all_results[cond].get("jsonl_path", ""))
        if not jsonl_path.exists():
            logger.warning(f"Seed-order verification: JSONL not found for {cond}")
            return {"skipped": True, "reason": f"JSONL not found for {cond}"}
        records: dict = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    qid = rec["qid"]
                    seed_ids = [
                        n["node_id"]
                        for n in rec.get("retrieved_nodes", [])
                        if not n.get("is_expanded", False)
                    ]
                    records[qid] = {
                        "hit":      rec.get("hit_at_seed_budget"),
                        "seed_ids": seed_ids,
                    }
                except Exception:
                    continue
        per_cond[cond] = records
        logger.info(f"  Seed verification: loaded {len(records)} questions from {cond}")

    base_cond = cond_names[0]
    base_records = per_cond[base_cond]
    mismatches: list = []

    for cond in cond_names[1:]:
        for qid, base in base_records.items():
            other = per_cond[cond].get(qid, {})
            hit_mismatch  = base["hit"]      != other.get("hit")
            ids_mismatch  = base["seed_ids"] != other.get("seed_ids", [])
            if hit_mismatch or ids_mismatch:
                mismatches.append({
                    "qid":          qid,
                    "cond_a":       base_cond,
                    "hit_a":        base["hit"],
                    "seed_ids_a":   base["seed_ids"],
                    "cond_b":       cond,
                    "hit_b":        other.get("hit"),
                    "seed_ids_b":   other.get("seed_ids", []),
                    "hit_mismatch": hit_mismatch,
                    "ids_mismatch": ids_mismatch,
                })

    n_mismatches = len(mismatches)
    verified = n_mismatches == 0

    if not verified:
        logger.warning(
            f"SEED ORDER MISMATCH: {n_mismatches} questions have different "
            f"seed node IDs or hit_at_seed_budget across conditions. "
            f"The fair-comparison claim requires identical seed ordering across "
            f"flat / graph-1 / graph-2. First mismatch: {mismatches[0]}"
        )
    else:
        logger.info(
            f"Seed order VERIFIED: seed node IDs and hit_at_seed_budget are "
            f"identical across all {len(cond_names)} conditions for "
            f"{len(base_records)} questions."
        )

    return {
        "verified":             verified,
        "metric_checked":       "hit_at_seed_budget + seed_node_ids",
        "n_questions_checked":  len(base_records),
        "n_mismatches":         n_mismatches,
        "conditions_compared":  cond_names,
        "mismatch_examples":    mismatches[:10],   # first 10 for debug
    }


# ── §7  Evidence-alignment inspection ────────────────────────────────────────

def _inspect_question(
    video_id: str,
    qid: str,
    rows: List[dict],
    gsub: dict,
    graphs_dir: Path,
    split: str,
    top_k: int = 7,
    node_types_mode: str = "transcript+visual",
    hybrid_alpha: float = 0.7,
    text_model: str = "gpt-4o",
) -> None:
    """
    Print a detailed evidence-alignment inspection for one (video_id, qid) pair.

    Runs flat (hop=0) retrieval, filters grounding nodes, computes all metrics,
    and prints a human-readable table so you can manually verify that graph node
    timestamps and NExT-GQA gold intervals are in the same unit (seconds) and
    that the retrieved spans actually cover the gold temporal window.

    Output sections:
      §1  Basic question / video info and time-base check
      §2  Gold grounding intervals from gsub
      §3  Top retrieved grounding nodes with per-node IoP/IoU
      §4  Official predicted span (top-1 node) and cluster diagnostic
      §5  Aggregate diagnostic metrics
    """
    from videograph_eval.mc_answer import GraphAnswerSession

    active_node_types = (
        VISUAL_ONLY_NODE_TYPES if node_types_mode == "visual-only"
        else GROUNDING_NODE_TYPES
    )

    # Find the CSV row
    row = next(
        (r for r in rows
         if r["video_id"] == video_id and str(r["qid"]) == str(qid)),
        None,
    )
    if row is None:
        print(f"\n[INSPECT] No row for video_id={video_id} qid={qid} in loaded split.")
        return

    q_text  = row["question"]
    q_type  = row.get("type", "")
    options = [row.get(f"a{i}", "") for i in range(5)]
    gold_idx = _get_answer_index(row)
    gold_ans = row.get("answer", "")

    vid_gsub       = gsub.get(video_id, {})
    gsub_duration  = vid_gsub.get("duration")
    vid_fps        = vid_gsub.get("fps")
    gold_intervals = vid_gsub.get("location", {}).get(str(qid), [])
    if gold_intervals and isinstance(gold_intervals[0], (int, float)):
        gold_intervals = [gold_intervals]
    has_gold = bool(gold_intervals)

    graph_path    = graphs_dir / video_id / "graph.json"
    graph_exists  = graph_path.exists()
    graph_max_end = _graph_max_end(graph_path) if graph_exists else None
    timebase_ok   = (
        graph_max_end is None
        or gsub_duration is None
        or abs(graph_max_end - gsub_duration) <= TIMEBASE_WARN_S
    )

    sep  = "-" * 74
    sep2 = "- " * 37

    # Ensure UTF-8 output on Windows
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print()
    print(sep)
    print("  EVIDENCE ALIGNMENT INSPECTION")
    print(sep)

    # §1 Basic info
    print(f"  Split          : {split}")
    print(f"  Video ID       : {video_id}")
    print(f"  QID            : {qid}")
    print(f"  Type           : {q_type}")
    print(f"  Question       : {q_text}")
    print(f"  Gold answer    : {gold_ans!r}  (option index {gold_idx})")
    for i, opt in enumerate(options):
        marker = " <-- gold" if i == gold_idx else ""
        print(f"    a{i}: {opt}{marker}")
    print(f"  Node types     : {node_types_mode}")
    print(f"  Top-k          : {top_k}")
    print(f"  gsub duration  : {gsub_duration}s   fps={vid_fps}")
    if graph_max_end is not None:
        print(f"  Graph max_end  : {graph_max_end:.2f}s")
    else:
        print(f"  Graph max_end  : N/A (graph missing or no finite end times)")
    if not timebase_ok:
        diff = abs((graph_max_end or 0) - (gsub_duration or 0))
        print(f"  TIME-BASE WARN : diff={diff:.1f}s > {TIMEBASE_WARN_S}s *** check timestamps ***")
    else:
        print(f"  Time-base      : OK (graph/gsub within {TIMEBASE_WARN_S}s)")

    # §2 Gold grounding intervals
    print()
    print(f"  Gold grounding intervals  [{len(gold_intervals)} segment(s)]")
    if has_gold:
        for idx, g in enumerate(gold_intervals):
            print(f"    [{idx}]  [{g[0]:.2f}s -> {g[1]:.2f}s]  duration={g[1]-g[0]:.2f}s")
        print(f"    Total gold duration : {total_gold_length(gold_intervals):.2f}s")
    else:
        print("    (no gold interval for this question in gsub)")

    # §3 Retrieved grounding nodes
    print()
    if not graph_exists:
        print(f"  [INSPECT] Graph not found: {graph_path}")
        print(sep)
        return

    try:
        session = GraphAnswerSession(
            graph_path=str(graph_path),
            top_k=top_k,
            hop_expansion=0,           # flat for inspection
            text_model=text_model,
            hybrid_alpha=hybrid_alpha,
            dataset=f"nextqa-{split}",
        )
        evidence_nodes = session.retrieve(q_text)
    except Exception as exc:
        print(f"  [INSPECT] Retrieval failed: {exc}")
        print(sep)
        return

    grounding_nodes = _filter_grounding_nodes(
        evidence_nodes, video_duration=gsub_duration, node_types=active_node_types,
    )

    print(f"  Retrieved nodes (all types)  : {len(evidence_nodes)}")
    print(f"  After grounding filter       : {len(grounding_nodes)} valid {node_types_mode} nodes")
    print()
    hdr = (f"  {'Rk':<2}  {'Node ID':<28}  {'Type':<14}  "
           f"{'Start':>7}  {'End':>7}  {'Dur':>5}  "
           f"{'Score':>6}  {'Exp':>3}  "
           f"{'IoP@gold':>8}  {'IoU@gold':>8}")
    print(hdr)
    print(f"  {sep2}")
    for rank, node in enumerate(grounding_nodes, 1):
        ns    = float(node.get("start") or 0.0)
        ne    = float(node.get("end")   or 0.0)
        score = node.get("score") or 0.0
        nid   = node.get("node_id", "?")[:28]
        ntype = node.get("node_type", "?")[:14]
        is_exp   = node.get("is_expanded", False)
        exp_src  = (node.get("expansion_source") or "")[:6]
        exp_tag  = f"Y({exp_src})" if is_exp else "N"
        niop = iop_against_gold(ns, ne, gold_intervals) if has_gold else float("nan")
        niou = iou_against_gold(ns, ne, gold_intervals) if has_gold else float("nan")
        flag = " <-- TOP-1 (official pred)" if rank == 1 else ""
        print(
            f"  {rank:<2}  {nid:<28}  {ntype:<14}  "
            f"{ns:>7.2f}  {ne:>7.2f}  {ne-ns:>5.2f}  "
            f"{score:>6.4f}  {exp_tag:<5}  "
            f"{niop:>8.4f}  {niou:>8.4f}{flag}"
        )
        if rank >= 12:
            print(f"  ... ({len(grounding_nodes) - 12} more nodes not shown)")
            break

    # §4 Official predicted span
    print()
    if grounding_nodes and has_gold:
        top1_iop, top1_iou = _top1_metrics(grounding_nodes, gold_intervals)
        cl_iop, cl_iou     = _best_cluster_metrics(grounding_nodes, gold_intervals)
        t1 = grounding_nodes[0]
        t1s = float(t1.get("start") or 0.0)
        t1e = float(t1.get("end")   or 0.0)
        hit_str = "GROUNDED" if top1_iop >= GROUNDING_THRESHOLD else f"BELOW threshold ({GROUNDING_THRESHOLD})"
        print(f"  Official prediction  [PRIMARY — top-1 node span]")
        print(f"    Span           : [{t1s:.2f}s -> {t1e:.2f}s]  duration={t1e-t1s:.2f}s")
        print(f"    top-1 IoP      : {top1_iop:.4f}   {hit_str}")
        print(f"    top-1 IoU      : {top1_iou:.4f}")
        print(f"  Cluster prediction [DIAGNOSTIC — not the official metric]")
        print(f"    cluster IoP    : {cl_iop:.4f}   cluster IoU: {cl_iou:.4f}")
    elif not has_gold:
        print(f"  Official prediction : N/A (no gold interval)")
    else:
        print(f"  Official prediction : NONE (no valid grounding nodes after filter)")
        print(f"    top-1 IoP = 0.0  (kept in denominator)")

    # §5 Aggregate diagnostics
    print()
    if has_gold:
        hit1            = _hit_at_k_seed(grounding_nodes, gold_intervals, 1)
        hit7            = _hit_at_k_seed(grounding_nodes, gold_intervals, 7)
        hit_seed_budget = _hit_at_k_seed(grounding_nodes, gold_intervals, top_k)
        hit_any         = _hit_any(grounding_nodes, gold_intervals)
        hit_beyond      = bool(hit_any) and not bool(hit_seed_budget)
        hit_exp         = (
            not bool(hit_seed_budget)
            and any(
                iop_against_gold(
                    float(n.get("start") or 0), float(n.get("end") or 0), gold_intervals
                ) >= GROUNDING_THRESHOLD
                for n in grounding_nodes
                if n.get("is_expanded", False)
            )
        )
        best_iop = max(
            (iop_against_gold(float(n.get("start") or 0), float(n.get("end") or 0), gold_intervals)
             for n in grounding_nodes),
            default=0.0,
        ) if grounding_nodes else 0.0
        best_iou = max(
            (iou_against_gold(float(n.get("start") or 0), float(n.get("end") or 0), gold_intervals)
             for n in grounding_nodes),
            default=0.0,
        ) if grounding_nodes else 0.0
        iog = _iog_coverage(grounding_nodes, gold_intervals)
        print(f"  Diagnostics (flat retrieval, top_k={top_k})")
        print(f"    Hit@1 seed           : {hit1}")
        print(f"    Hit@7 seed           : {hit7}")
        print(f"    Hit@seed_budget (k={top_k}): {hit_seed_budget}")
        print(f"    Hit@any              : {hit_any}")
        print(f"    hit_beyond_budget    : {hit_beyond}  (hit_any AND NOT hit@seed_budget)")
        print(f"    hit_expanded_only    : {hit_exp}  (expanded node hit AND NOT hit@seed_budget)")
        print(f"    best_node_IoP        : {best_iop:.4f}")
        print(f"    best_node_IoU        : {best_iou:.4f}")
        print(f"    IoG coverage     : {iog:.4f}")
    print(sep)


# ── §8  Report generation ─────────────────────────────────────────────────────

def _save_summary(all_results: dict, output_dir: Path) -> None:
    """Write summary.json and summary.md."""
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary JSON  -> {json_path}")

    conds = [v for k, v in all_results.items() if not k.startswith("_")]
    meta  = all_results.get("_meta", {})
    q_types = ["CW", "CH", "TN", "TC", "TP"]
    retrieval_only_run = any(s.get("retrieval_only") for s in conds)
    node_types_mode = meta.get("node_types_mode", "transcript+visual")

    def _pct(v):  return f"{v:.2%}"  if v is not None else "N/A"
    def _f3(v):   return f"{v:.3f}"  if v is not None else "N/A"
    def _f1(v):   return f"{v:.1f}"  if v is not None else "N/A"
    def _int(v):  return str(int(v)) if v is not None else "N/A"

    prediction_mode = meta.get("prediction_mode", "top-1-node")
    graph_cov_pcts  = [s.get("graph_coverage_pct", 1.0) for s in conds]
    min_cov         = min(graph_cov_pcts) if graph_cov_pcts else 1.0
    coverage_warn   = f"  WARNING: Graph coverage = {min_cov:.0%} - missing-graph questions score 0 in paper-comparable metrics." if min_cov < 1.0 else ""

    lines: List[str] = [
        "# NExT-GQA Grounding Experiment — Results",
        "",
        f"> Split: **{meta.get('split', '?')}**  "
        f"| IoP threshold = {GROUNDING_THRESHOLD}  "
        f"| Cluster window = {CLUSTER_WINDOW_S}s  "
        f"| Node types = **{node_types_mode}**  "
        f"| Prediction mode = **{prediction_mode}**  "
        + ("| **RETRIEVAL-ONLY** — QA/Acc@GQA N/A"
           if retrieval_only_run
           else ""),
        "",
        "> **Prediction rule**",
        f"> The official grounding prediction is the temporal span of the **top-ranked retrieved node** (`{prediction_mode}`).",
        "> The cluster span is a secondary diagnostic — it is NOT the official predicted interval.",
        "> Global earliest-to-latest union of all retrieved nodes is **never** used as the official prediction.",
        "",
        "> **Comparability with LLoVi-style baselines**",
        "> LLoVi-style methods select a single timestamped evidence window from retrieved evidence.",
        "> This harness uses the same principle: the top-1 ranked node span is the predicted window.",
        "> Perfect comparability requires matching (i) prediction rule, (ii) split (val vs test),",
        "> and (iii) denominator (all questions vs graph-covered subset). See table headers for denominator labels.",
        "",
        "> **Denominator key**",
        "> - `[PAPER]` = denominator is ALL questions in the split (missing-graph -> score 0).",
        "> - `[graph-covered]` = denominator is questions where a graph was built (subset, typically higher).",
        "> - `[denom=n_grounded]` = questions with a gold interval AND a graph.",
        f">{coverage_warn}",
        "",
        "> **Metric roles**",
        "> - Official/paper: `accuracy_all_split` [PAPER], `acc_gqa_all_with_gold` [PAPER], `mean_top1_iop/iou`, `iop/iou_at_0.3/0.5`.",
        "> - Graph-covered subset: `accuracy`, `acc_gqa_top1` — comparable across conditions but not to split-wide baselines.",
        "> - Retrieval diagnostics: `hit_any`, `hit_expanded_only`, `iog_coverage`, `mean_best_iop` — support flat-vs-graph claim.",
        "> - Oracle upper-bounds: `acc_gqa_oracle_retrieved`, `mean_best_iop` — do NOT use for primary comparison.",
        "",
    ]

    # ── Graph coverage ────────────────────────────────────────────────────────
    lines += [
        "## Graph Coverage",
        "",
        "| Condition | Total videos | With graph | Missing graph | Coverage |",
        "|-----------|-------------|------------|---------------|---------|",
    ]
    for s in conds:
        lines.append(
            f"| {s['condition']:<9} "
            f"| {_int(s.get('n_total_videos'))} "
            f"| {_int(s.get('n_videos_with_graph'))} "
            f"| {_int(s.get('n_videos_missing_graph'))} "
            f"| {_pct(s.get('graph_coverage_pct'))} |"
        )

    # ── Denominator counts ────────────────────────────────────────────────────
    lines += [
        "",
        "## Denominator Counts",
        "",
        "| Condition | Total Qs (split) | Total w/gold | Answered (graph-covered) | Grounded | Missing graph (Qs) | Missing gold | Parse fail |",
        "|-----------|-----------------|-------------|--------------------------|----------|--------------------|--------------|------------|",
    ]
    for s in conds:
        lines.append(
            f"| {s['condition']:<9} "
            f"| {_int(s.get('n_total_split_questions'))} "
            f"| {_int(s.get('n_total_with_gold'))} "
            f"| {_int(s.get('n_answered'))} "
            f"| {_int(s.get('n_grounded'))} "
            f"| {_int(s.get('n_missing_graph'))} "
            f"| {_int(s.get('n_missing_gold_interval'))} "
            f"| {_int(s.get('n_answer_parse_failures'))} |"
        )

    # ── Node counts ───────────────────────────────────────────────────────────
    lines += [
        "",
        "## Node Counts per Query  _(denom = answered questions)_",
        "",
        "| Condition | top_k | hop | avg_nodes | avg_seed_nodes | avg_expanded_nodes |",
        "|-----------|-------|-----|-----------|----------------|---------------------|",
    ]
    for s in conds:
        lines.append(
            f"| {s['condition']:<9} "
            f"| {s['top_k']} "
            f"| {s['hop_expansion']} "
            f"| {_f1(s['avg_nodes'])} "
            f"| {_f1(s['avg_seed_nodes'])} "
            f"| {_f1(s['avg_expanded_nodes'])} |"
        )

    # ── QA accuracy ───────────────────────────────────────────────────────────
    if not retrieval_only_run:
        lines += [
            "",
            "## QA Accuracy",
            "",
            "| Condition | hop | Acc (all split) [PAPER] | Acc (answered) [graph-covered] |",
            "|-----------|-----|------------------------|-------------------------------|",
        ]
        for s in conds:
            lines.append(
                f"| {s['condition']:<9} | {s['hop_expansion']} "
                f"| {_pct(s.get('accuracy_all_split'))} "
                f"| {_pct(s.get('accuracy'))} |"
            )
        lines += [
            "",
            "_`Acc (all split)` denominator = all questions in the split CSV "
            "(missing-graph questions score 0). "
            "`Acc (answered)` denominator = graph-covered questions only._",
        ]

        lines += [
            "",
            "## Acc@GQA  _(answer correct AND top-1 IoP >= 0.5)_",
            "",
            "| Condition | hop | Acc@GQA (all w/gold) [PAPER] | Acc@GQA top1 [graph-covered] | Acc@GQA oracle [upper-bound] |",
            "|-----------|-----|------------------------------|------------------------------|------------------------------|",
        ]
        for s in conds:
            lines.append(
                f"| {s['condition']:<9} | {s['hop_expansion']} "
                f"| {_pct(s.get('acc_gqa_all_with_gold'))} "
                f"| {_pct(s.get('acc_gqa_top1'))} "
                f"| {_pct(s.get('acc_gqa_oracle_retrieved'))} |"
            )
        lines += [
            "",
            "_`Acc@GQA (all w/gold)` denominator = all questions with gold intervals "
            "(missing-graph questions score 0) — **use this for paper comparison**. "
            "`Acc@GQA top1` uses graph-covered denominator. "
            "`Acc@GQA oracle` uses the best node per question (upper-bound only)._",
        ]

    # ── Retrieval recall ──────────────────────────────────────────────────────
    lines += [
        "",
        "## Retrieval Recall  _(denom = n_grounded; IoP >= 0.5 to count as hit)_",
        "",
        "| Condition | top_k | hop | Hit@1 (seed) | Hit@5 (seed) | Hit@7 (seed) | Hit@budget (seed, k=top_k) | Hit@any | Hit beyond budget | Hit expanded-only | IoG coverage |",
        "|-----------|-------|-----|-------------|-------------|-------------|---------------------------|---------|------------------|------------------|-------------|",
    ]
    for s in conds:
        lines.append(
            f"| {s['condition']:<9} "
            f"| {s['top_k']} "
            f"| {s['hop_expansion']} "
            f"| {_pct(s['hit_at_1_seed'])} "
            f"| {_pct(s['hit_at_5_seed'])} "
            f"| {_pct(s['hit_at_7_seed'])} "
            f"| {_pct(s.get('hit_at_seed_budget'))} "
            f"| {_pct(s['hit_any'])} "
            f"| {_pct(s.get('hit_beyond_seed_budget'))} ({_int(s.get('n_hit_beyond_seed_budget'))}) "
            f"| {_pct(s.get('hit_expanded_only'))} ({_int(s.get('n_hit_expanded_only'))}) "
            f"| {_f3(s['mean_iog_coverage'])} |"
        )
    lines += [
        "",
        "_All `seed` metrics use only nodes where `is_expanded=False` (true seed-only). "
        "`Hit@1/5/7 (seed)` apply the corresponding fixed K; valid for comparison only "
        "when `top_k >= K`. `Hit@budget` uses `k=top_k` and is always valid. "
        "`Hit@any` includes all nodes. "
        "`Hit beyond budget` = `hit_any AND NOT hit@budget`: any extra evidence beyond the seed budget "
        "(may be >0 for flat when budget < total seeds returned). "
        "`Hit expanded-only` = expanded node hit AND seed budget missed: **true graph expansion gain**; "
        "always 0 for flat (no expanded nodes)._",
    ]

    # ── Localization quality ──────────────────────────────────────────────────
    lines += [
        "",
        "## Localization Quality  _(denom = n_grounded; top-1 node; official max-over-segments IoP/IoU)_",
        "",
        "| Condition | top_k | hop | mIoP | IoP@0.3 | IoP@0.5 | mIoU | IoU@0.3 | IoU@0.5 | Cluster IoP | Best IoP (oracle) |",
        "|-----------|-------|-----|------|---------|---------|------|---------|---------|-------------|------------------|",
    ]
    for s in conds:
        lines.append(
            f"| {s['condition']:<9} "
            f"| {s['top_k']} "
            f"| {s['hop_expansion']} "
            f"| {_f3(s['mean_top1_iop'])} "
            f"| {_pct(s.get('iop_at_0.3'))} "
            f"| {_pct(s.get('iop_at_0.5'))} "
            f"| {_f3(s['mean_top1_iou'])} "
            f"| {_pct(s.get('iou_at_0.3'))} "
            f"| {_pct(s.get('iou_at_0.5'))} "
            f"| {_f3(s['mean_cluster_iop'])} "
            f"| {_f3(s['mean_best_iop'])} |"
        )
    lines += [
        "",
        "_All localization metrics use the **top-1 retrieved node** as the predicted span "
        "(official NExT-GQA protocol: IoP/IoU computed against each gold interval "
        "independently, take max). `Cluster IoP` uses a tight window around top-1. "
        "`Best IoP` is oracle (max over all retrieved nodes). "
        "Denominator = n_grounded (questions with gold intervals, graph-covered)._",
    ]

    # ── Per-type tables ───────────────────────────────────────────────────────
    def _type_table(metric_key: str, title: str, fmt_fn=_pct) -> List[str]:
        block = [
            "",
            f"## Per Question-Type — {title}",
            "",
            "| Condition | top_k | hop | " + " | ".join(q_types) + " |",
            "|-----------|-------|-----|" + "|".join(["------"] * len(q_types)) + "|",
        ]
        for s in conds:
            cells = [
                fmt_fn(s.get("by_type", {}).get(qt, {}).get(metric_key, 0.0))
                for qt in q_types
            ]
            block.append(
                f"| {s['condition']:<9} | {s['top_k']} | {s['hop_expansion']} | "
                + " | ".join(cells) + " |"
            )
        return block

    if not retrieval_only_run:
        lines += _type_table("accuracy",           "Acc@QA  [denom = n per type]")
        lines += _type_table("acc_gqa_top1",       "Acc@GQA top1 (PRIMARY fair)  [denom = n per type]")
        lines += _type_table("acc_gqa_oracle",     "Acc@GQA oracle (upper-bound)  [denom = n per type]")
    lines += _type_table("mean_top1_iop",      "mIoP  [denom = n_grounded per type]",      fmt_fn=_f3)
    lines += _type_table("iop_at_0.3",         "IoP@0.3  [denom = n_grounded per type]")
    lines += _type_table("iop_at_0.5",         "IoP@0.5  [denom = n_grounded per type]")
    lines += _type_table("mean_top1_iou",      "mIoU  [denom = n_grounded per type]",      fmt_fn=_f3)
    lines += _type_table("iou_at_0.3",         "IoU@0.3  [denom = n_grounded per type]")
    lines += _type_table("iou_at_0.5",         "IoU@0.5  [denom = n_grounded per type]")
    lines += _type_table("hit_at_7_seed",      "Hit@7 seed  [denom = n_grounded per type]")
    lines += _type_table("hit_any",            "Hit@any  [denom = n_grounded per type]")
    lines += _type_table("hit_expanded_only",  "Hit expanded-only  [denom = n_grounded per type]")
    lines += _type_table("mean_iog_coverage",  "IoG coverage  [denom = n_grounded per type]", fmt_fn=_f3)
    lines += _type_table("mean_best_iop",      "Mean Best IoP (oracle)  [denom = n_grounded per type]", fmt_fn=_f3)

    # ── Seed-order verification (if available) ────────────────────────────────
    seed_verify = meta.get("seed_order_verification")
    if seed_verify and not seed_verify.get("skipped"):
        status = "PASS" if seed_verify.get("verified") else "FAIL"
        lines += [
            "",
            "## Seed-Order Verification",
            "",
            f"**Status: {status}**  "
            f"| Metric checked: `{seed_verify.get('metric_checked', 'hit_at_seed_budget + seed_node_ids')}`  "
            f"| Questions checked: {seed_verify.get('n_questions_checked', 'N/A')}  "
            f"| Mismatches: {seed_verify.get('n_mismatches', 'N/A')}",
            "",
            "_A PASS confirms that seed node IDs and `hit_at_seed_budget` are identical "
            "across all conditions for every question, validating the fair-comparison claim._",
        ]
        if not seed_verify.get("verified") and seed_verify.get("mismatch_examples"):
            lines += [
                "",
                "First mismatch examples:",
                "```",
            ]
            for ex in seed_verify["mismatch_examples"][:3]:
                lines.append(json.dumps(ex))
            lines.append("```")

    md_path = output_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Summary MD    -> {md_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NExT-GQA grounding experiment: flat vs graph-hop retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",    required=True,
        help="Root data dir (must contain NExT-GQA/<split>.csv and NExT-GQA/gsub_<split>.json)")
    parser.add_argument("--graphs-dir",  required=True,
        help="Per-video graph.json directory, e.g. results/v1/graphs/nextqa-val")
    parser.add_argument("--output-dir",  required=True,
        help="Output directory for JSONL + summary files")
    parser.add_argument("--split",       default="val", choices=["val", "test"],
        help="Dataset split to evaluate (default: val)")
    parser.add_argument("--max-questions", type=int, default=None,
        help="Limit total questions loaded (debug)")
    parser.add_argument("--video-id",    default=None,
        help="Filter to one video ID, e.g. 4882821564")
    parser.add_argument("--conditions",  nargs="+",
        default=list(CONDITIONS.keys()), choices=list(CONDITIONS.keys()),
        help="Conditions to run (default: all three)")
    parser.add_argument("--retrieval-only", action="store_true",
        help="Skip LLM answering; compute grounding metrics only (no API cost for QA)")
    parser.add_argument("--top-k", type=int, default=None,
        help="Override top_k for all conditions (e.g. for k-sweep; default: per-condition value)")
    parser.add_argument("--text-model",  default="gpt-4o",
        help="OpenAI model for MC answering (default: gpt-4o)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7,
        help="Embedding weight 0-1 (default: 0.7)")
    parser.add_argument(
        "--node-types",
        default="transcript+visual",
        choices=["transcript+visual", "visual-only"],
        help=(
            "Node types to include in grounding metrics. "
            "'transcript+visual' (default): TranscriptNode + VisualNode. "
            "'visual-only': VisualNode only — for modality-comparable analysis "
            "against vision/caption baselines. LLM context is unchanged."
        ),
    )
    parser.add_argument("--verify-seed-order", action="store_true",
        help=(
            "After all conditions finish, verify that seed node IDs and "
            "hit_at_seed_budget are identical across conditions for every question. "
            "Confirms the fair-comparison invariant. "
            "Reads JSONL output files; adds result to summary.json."
        ),
    )
    parser.add_argument(
        "--inspect-question",
        default=None,
        metavar="VIDEO_ID:QID",
        help=(
            "Print a detailed evidence-alignment inspection for one question "
            "and exit before running the full evaluation. "
            "Format: VIDEO_ID:QID, e.g. --inspect-question 4882821564:1"
        ),
    )
    parser.add_argument(
        "--inspect-first-grounded",
        action="store_true",
        help=(
            "Find the first question that has both a graph and a gold interval, "
            "print an alignment inspection, then exit before the full evaluation. "
            "Useful for quickly confirming timestamp alignment on real data."
        ),
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split          = args.split
    csv_path       = data_dir / "NExT-GQA" / f"{split}.csv"
    gsub_json_path = data_dir / "NExT-GQA" / f"gsub_{split}.json"

    for p, label in [
        (csv_path,       f"NExT-GQA/{split}.csv"),
        (gsub_json_path, f"NExT-GQA/gsub_{split}.json"),
        (graphs_dir,     "--graphs-dir"),
    ]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    logger.info(f"Loading {csv_path}")
    rows = _load_nextgqa_split(csv_path, args.video_id, args.max_questions)
    n_total_split_questions = len(rows)
    logger.info(f"  {n_total_split_questions} questions")

    logger.info(f"Loading {gsub_json_path}")
    gsub = _load_gsub(gsub_json_path)
    logger.info(f"  {len(gsub)} videos with grounding")

    n_total_with_gold = sum(
        1 for r in rows
        if gsub.get(r["video_id"], {}).get("location", {}).get(str(r["qid"]))
    )
    logger.info(
        f"  {n_total_with_gold}/{n_total_split_questions} questions have gold intervals "
        f"({100 * n_total_with_gold / max(n_total_split_questions, 1):.1f}%)"
    )

    n_total_videos = len(set(r["video_id"] for r in rows))

    logger.info("")
    logger.info("=" * 60)
    logger.info("NExT-GQA GROUNDING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"  Split:              {split}")
    logger.info(f"  Graphs dir:         {graphs_dir}")
    logger.info(f"  Output dir:         {output_dir}")
    logger.info(f"  Questions:          {n_total_split_questions}")
    logger.info(f"  With gold:          {n_total_with_gold}")
    logger.info(f"  Unique videos:      {n_total_videos}")
    logger.info(f"  Video filter:       {args.video_id or 'all'}")
    logger.info(f"  Conditions:         {args.conditions}")
    logger.info(f"  Retrieval only:     {args.retrieval_only}")
    logger.info(f"  Node types:         {args.node_types}")
    logger.info(f"  Top-k override:     {args.top_k or 'per-condition default'}")
    logger.info(f"  Text model:         {args.text_model}")
    logger.info(f"  Hybrid alpha:       {args.hybrid_alpha}")
    logger.info(f"  GQA threshold:      IoP >= {GROUNDING_THRESHOLD}")
    logger.info(f"  Cluster window:     +/- {CLUSTER_WINDOW_S}s")
    logger.info(f"  Verify seed order:  {args.verify_seed_order}")
    logger.info("=" * 60)

    # ── Inspection mode (exits before running full evaluation) ────────────────
    if args.inspect_question or args.inspect_first_grounded:
        if args.inspect_question:
            parts = args.inspect_question.split(":", 1)
            if len(parts) != 2:
                logger.error("--inspect-question must be VIDEO_ID:QID, e.g. 4882821564:1")
                sys.exit(1)
            insp_vid, insp_qid = parts[0].strip(), parts[1].strip()
        else:  # --inspect-first-grounded
            insp_vid, insp_qid = None, None
            for r in rows:
                vid = r["video_id"]
                qid = str(r["qid"])
                if (graphs_dir / vid / "graph.json").exists():
                    if gsub.get(vid, {}).get("location", {}).get(qid):
                        insp_vid, insp_qid = vid, qid
                        break
            if insp_vid is None:
                logger.error("--inspect-first-grounded: no question found with both a graph and gold interval")
                sys.exit(1)
        effective_k_insp = args.top_k if args.top_k is not None else CONDITIONS["flat"]["top_k"]
        _inspect_question(
            video_id=insp_vid, qid=insp_qid, rows=rows, gsub=gsub,
            graphs_dir=graphs_dir, split=split, top_k=effective_k_insp,
            node_types_mode=args.node_types, hybrid_alpha=args.hybrid_alpha,
            text_model=args.text_model,
        )
        sys.exit(0)

    all_results: dict = {}
    t_start = time.time()

    for cond_name in args.conditions:
        cfg = CONDITIONS[cond_name]
        effective_k = args.top_k if args.top_k is not None else cfg["top_k"]
        all_results[cond_name] = run_condition(
            cond_name               = cond_name,
            top_k                   = effective_k,
            hop_expansion           = cfg["hop_expansion"],
            rows                    = rows,
            graphs_dir              = graphs_dir,
            gsub                    = gsub,
            output_dir              = output_dir,
            text_model              = args.text_model,
            hybrid_alpha            = args.hybrid_alpha,
            retrieval_only          = args.retrieval_only,
            split                   = split,
            n_total_split_questions = n_total_split_questions,
            n_total_with_gold       = n_total_with_gold,
            node_types_mode         = args.node_types,
        )

    # ── Seed-order verification ───────────────────────────────────────────────
    seed_verification: Optional[dict] = None
    if args.verify_seed_order:
        logger.info("")
        logger.info("Running seed-order verification ...")
        seed_verification = _verify_seed_order(all_results, output_dir)

    all_results["_meta"] = {
        "version":                  "v4",
        "split":                    split,
        "total_time_s":             round(time.time() - t_start, 1),
        "n_total_split_questions":  n_total_split_questions,
        "n_total_with_gold":        n_total_with_gold,
        "n_total_videos":           n_total_videos,
        "video_id_filter":          args.video_id,
        "max_questions":            args.max_questions,
        "conditions":               args.conditions,
        "retrieval_only":           args.retrieval_only,
        "node_types_mode":          args.node_types,
        "top_k_override":           args.top_k,
        "text_model":               args.text_model,
        "hybrid_alpha":             args.hybrid_alpha,
        "grounding_threshold":      GROUNDING_THRESHOLD,
        "cluster_window_s":         CLUSTER_WINDOW_S,
        "graphs_dir":               str(graphs_dir),
        "seed_order_verification":  seed_verification,
        "prediction_mode":          "top-1-node",
    }

    _save_summary(all_results, output_dir)

    # ── Final console table ───────────────────────────────────────────────────
    def _pct(v): return f"{v:>6.2%}" if v is not None else f"{'N/A':>6}"
    logger.info("")
    logger.info("=" * 100)
    logger.info("FINAL RESULTS")
    logger.info("=" * 100)
    hdr = (
        f"{'Cond':<9}  {'k':>2}  {'hop':>3}  "
        f"{'Seeds':>5}  {'Exp':>4}  "
        f"{'H@sb':>6}  {'H@any':>6}  {'H@bsb':>6}  {'H@exp':>6}  "
        f"{'Top1IoP':>8}  {'BestIoP':>8}  "
        f"{'IoG':>6}  {'Acc(all)':>9}  {'GQA(all)':>9}"
    )
    logger.info(hdr)
    logger.info(
        "  H@sb  = hit_at_seed_budget (seed-only, k=top_k)  |  "
        "H@bsb = hit_beyond_seed_budget (general extra)  |  "
        "H@exp = hit_expanded_only (true graph gain, 0 for flat)"
    )
    logger.info("-" * 100)
    for cn in args.conditions:
        s = all_results[cn]
        logger.info(
            f"{s['condition']:<9}  {s['top_k']:>2}  {s['hop_expansion']:>3}  "
            f"{s['avg_seed_nodes']:>5.1f}  {s['avg_expanded_nodes']:>4.1f}  "
            f"{_pct(s.get('hit_at_seed_budget'))}  {_pct(s['hit_any'])}  "
            f"{_pct(s.get('hit_beyond_seed_budget'))}  {_pct(s.get('hit_expanded_only'))}  "
            f"{s['mean_top1_iop']:>8.3f}  {s['mean_best_iop']:>8.3f}  "
            f"{s['mean_iog_coverage']:>6.3f}  "
            f"{_pct(s.get('accuracy_all_split'))}  "
            f"{_pct(s.get('acc_gqa_all_with_gold'))}"
        )
    logger.info("=" * 100)
    logger.info(f"Total time: {all_results['_meta']['total_time_s']:.0f}s")
    logger.info(f"Output:     {output_dir}")
    if seed_verification:
        status = "PASS" if seed_verification.get("verified") else "FAIL"
        logger.info(
            f"Seed order: {status} "
            f"({seed_verification.get('n_mismatches', '?')} mismatches)"
        )


if __name__ == "__main__":
    main()
