"""
Multiple-choice answering using graph-based retrieval.

Retrieves context from the video's knowledge graph and asks the LLM
to select the best option from a variable number of choices.
"""

import logging
import re
import time
import json
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

EGOSCHEMA_CONTEXT = """Note: In this video, 'C' refers to the camera wearer (first-person view) and 'O' refers to another person."""
NO_CONTEXT_MESSAGE = "No relevant context was retrieved from the video graph."


def _build_system_prompt(num_options: int) -> str:
    valid_indices = ", ".join(str(i) for i in range(num_options))
    return (
        "You are answering a multiple-choice question about a video.\n"
        "Based on the provided context from the video's knowledge graph, select the best answer.\n"
        f"Respond with ONLY the option number ({valid_indices}). Nothing else."
    )


class GraphAnswerSession:
    """Reusable per-video QA session that keeps graph/retriever state in memory."""

    def __init__(
        self,
        graph_path: str,
        text_model: str = "gpt-4o",
        top_k: int = 10,
        hop_expansion: int = 2,
        hybrid_alpha: float = 0.7,
        allowed_node_types=None,
        use_state_change_channel: bool = True,
        expansion_edge_types=None,
        persist_visual_channel_embeddings: bool = True,
        temperature: float = 0.0,
        dataset: str = "",
    ):
        from videograph.graph.serialization import load_graph_json
        from videograph.retrieval.graph_retrieval import GraphRetriever

        self.graph_path = Path(graph_path)
        self.video_dir = self.graph_path.parent
        self.embeddings_path = self.video_dir / "embeddings.json"
        self.graph = load_graph_json(self.graph_path)
        self.state_change_by_clip = _load_state_change_by_clip(self.video_dir)
        self.retriever = GraphRetriever(
            hybrid_alpha=hybrid_alpha,
            persist_visual_channel_embeddings=persist_visual_channel_embeddings,
        )
        self.client = OpenAI()
        self.text_model = text_model
        self.top_k = top_k
        self.hop_expansion = hop_expansion
        self.allowed_node_types = allowed_node_types
        self.use_state_change_channel = use_state_change_channel
        self.expansion_edge_types = expansion_edge_types
        self.temperature = temperature
        self.dataset = dataset

    def retrieve(self, question: str) -> list:
        """
        Run retrieval only — no LLM call.

        Returns the same evidence_nodes list that answer() would include,
        but skips the LLM.  Use this for grounding-metric sweeps where QA
        accuracy is not needed (saves ~$0.01/question on gpt-4o).
        """
        results, _ = self.retriever.retrieve(
            question,
            self.graph,
            top_k=self.top_k,
            hop_expansion=self.hop_expansion,
            include_visual=True,
            embeddings_path=self.embeddings_path if self.embeddings_path.exists() else None,
            allowed_node_types=self.allowed_node_types,
            use_state_change_channel=self.use_state_change_channel,
            expansion_edge_types=self.expansion_edge_types,
        )
        return [
            {
                "node_id":          r.node_id,
                "node_type":        r.node_type,
                "start":            r.start,
                "end":              r.end,
                "score":            r.score,
                "is_expanded":      r.is_expanded,
                "expansion_source": r.expansion_source,
            }
            for r in results
        ]

    def answer(
        self,
        question: str,
        options: list,
        start_time: float = None,
    ) -> dict:
        """Answer one multiple-choice question against the loaded video graph."""
        if start_time is None:
            start_time = time.time()

        if not options:
            return {
                "predicted": -1,
                "raw_response": "No answer options provided.",
                "answer_time_s": time.time() - start_time,
                "failure_reason": "invalid_options",
                "retrieval_context": "",
                "qa_user_prompt": "",
                "evidence_nodes": [],
            }

        results, subgraph = self.retriever.retrieve(
            question,
            self.graph,
            top_k=self.top_k,
            hop_expansion=self.hop_expansion,
            include_visual=True,
            embeddings_path=self.embeddings_path if self.embeddings_path.exists() else None,
            allowed_node_types=self.allowed_node_types,
            use_state_change_channel=self.use_state_change_channel,
            expansion_edge_types=self.expansion_edge_types,
        )

        context = _build_context(results, subgraph, self.state_change_by_clip)

        if _is_empty_context(context):
            return {
                "predicted": -1,
                "raw_response": NO_CONTEXT_MESSAGE,
                "answer_time_s": time.time() - start_time,
                "failure_reason": "no_context",
                "retrieval_context": context,
                "qa_user_prompt": "",
                "evidence_nodes": [
                    {
                        "node_id": r.node_id,
                        "node_type": r.node_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score,
                        "is_expanded": r.is_expanded,
                        "expansion_source": r.expansion_source,
                    }
                    for r in results
                ],
            }

        options_text = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
        answer_range = f"0-{len(options) - 1}"

        extra_context = ""
        if self.dataset == "egoschema":
            extra_context = f"\n{EGOSCHEMA_CONTEXT}\n"

        user_prompt = f"""Context from video:
{context}
{extra_context}
Question: {question}

Options:
{options_text}

Answer ({answer_range}):"""

        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": _build_system_prompt(len(options))},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=8,
        )

        raw = response.choices[0].message.content.strip()
        predicted = _parse_option(raw, len(options))
        answer_time = time.time() - start_time

        return {
            "predicted": predicted,
            "raw_response": raw,
            "answer_time_s": answer_time,
            "failure_reason": None,
            "retrieval_context": context,
            "qa_user_prompt": user_prompt,
            "evidence_nodes": [
                {
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                    "is_expanded": r.is_expanded,
                    "expansion_source": r.expansion_source,
                }
                for r in results
            ],
        }


def answer_mc(
    question: str,
    options: list,
    graph_path: str,
    text_model: str = "gpt-4o",
    top_k: int = 10,
    hop_expansion: int = 2,
    hybrid_alpha: float = 0.7,
    allowed_node_types=None,
    use_state_change_channel: bool = True,
    expansion_edge_types=None,
    persist_visual_channel_embeddings: bool = True,
    temperature: float = 0.0,
    dataset: str = "",
) -> dict:
    """
    Answer a multiple-choice question using graph-based retrieval.

    Args:
        question: The question text
        options: List of option strings
        graph_path: Path to graph.json
        text_model: Model for answer generation
        top_k: Number of nodes to retrieve
        hop_expansion: Hops for subgraph expansion
        hybrid_alpha: Weight for semantic retrieval vs lexical retrieval
        allowed_node_types: Optional node types to include in retrieval/context
        use_state_change_channel: Whether to add visual state-change retrieval seeds
        expansion_edge_types: Optional edge-type override for expansion
        persist_visual_channel_embeddings: Whether to write missing visual sidecar embeddings
        temperature: Sampling temperature (0 for deterministic)
        dataset: Dataset name (adds dataset-specific prompt hints)

    Returns:
        Dict with 'predicted' (0-based option index), 'raw_response' (str), 'answer_time_s' (float)
    """
    start_time = time.time()
    if not options:
        return {
            "predicted": -1,
            "raw_response": "No answer options provided.",
            "answer_time_s": time.time() - start_time,
            "failure_reason": "invalid_options",
            "retrieval_context": "",
            "qa_user_prompt": "",
        }
    session = GraphAnswerSession(
        graph_path=graph_path,
        text_model=text_model,
        top_k=top_k,
        hop_expansion=hop_expansion,
        hybrid_alpha=hybrid_alpha,
        allowed_node_types=allowed_node_types,
        use_state_change_channel=use_state_change_channel,
        expansion_edge_types=expansion_edge_types,
        persist_visual_channel_embeddings=persist_visual_channel_embeddings,
        temperature=temperature,
        dataset=dataset,
    )
    return session.answer(question, options, start_time=start_time)


def _build_context(results, subgraph, state_change_by_clip: dict) -> str:
    """Build the QA context with chronological primary and expanded evidence."""
    def sort_key(result) -> tuple:
        return (
            float(getattr(result, "start", 0.0) or 0.0),
            float(getattr(result, "end", 0.0) or 0.0),
            str(getattr(result, "node_type", "")),
            str(getattr(result, "node_id", "")),
        )

    def build_snippet(node) -> str:
        node_type = getattr(node, "node_type", "")
        if hasattr(node_type, "value"):
            node_type = node_type.value

        if node_type == "TranscriptNode":
            return (getattr(node, "text", "") or "")[:1000]

        if node_type == "VisualNode":
            visual_description = getattr(node, "visual_description", "") or ""
            ocr_text = getattr(node, "ocr_text", "") or ""
            detected_entities = getattr(node, "detected_entities", []) or []
            clip_id = getattr(node, "clip_id", "") or ""
            state_change = getattr(node, "state_change_from_previous", "") or ""
            if clip_id and state_change_by_clip:
                state_change = state_change or (state_change_by_clip.get(clip_id, "") or "")

            snippet_parts = []
            if visual_description:
                snippet_parts.append(visual_description)
            if ocr_text:
                snippet_parts.append(f"OCR: {ocr_text}")
            if detected_entities:
                snippet_parts.append(f"Entities: {', '.join(detected_entities[:10])}")
            if state_change.strip():
                snippet_parts.append(
                    f"State change from previous clip: {state_change.strip()}"
                )
            return " ".join(snippet_parts)[:1000]

        if node_type == "TopicNode":
            title = getattr(node, "title", "") or ""
            description = getattr(node, "description", "") or ""
            keywords = getattr(node, "keywords", []) or []

            snippet_parts = []
            if title:
                snippet_parts.append(title)
            if description:
                snippet_parts.append(description)
            if keywords:
                snippet_parts.append(f"Keywords: {', '.join(keywords[:10])}")
            return " ".join(snippet_parts)[:1000]

        if node_type == "EntityNode":
            name = getattr(node, "name", "") or ""
            entity_type = getattr(node, "entity_type", "") or ""
            aliases = getattr(node, "aliases", []) or []

            snippet_parts = []
            if name:
                snippet_parts.append(name)
            if entity_type:
                snippet_parts.append(f"({entity_type})")
            if aliases:
                snippet_parts.append(f"Aliases: {', '.join(aliases[:10])}")
            return " ".join(snippet_parts)[:1000]

        return ""

    def format_entry(result, node) -> str:
        snippet = build_snippet(node)
        if not snippet:
            return ""
        node_type = getattr(node, "node_type", "")
        if hasattr(node_type, "value"):
            node_type = node_type.value
        start = getattr(node, "start", 0) or 0
        end = getattr(node, "end", 0) or 0
        ts = f" [{start:.1f}s-{end:.1f}s]" if start or end else ""
        return f"[{result.node_id}] ({node_type}{ts}): {snippet}"

    if not subgraph:
        return NO_CONTEXT_MESSAGE

    primary_results = sorted(
        [r for r in results if not getattr(r, "is_expanded", False)],
        key=sort_key,
    )
    expanded_results = sorted(
        [r for r in results if getattr(r, "is_expanded", False)],
        key=sort_key,
    )

    primary_entries = []
    for result in primary_results:
        node = subgraph.nodes.get(result.node_id)
        if node is None:
            continue
        entry = format_entry(result, node)
        if entry:
            primary_entries.append(entry)

    expanded_entries = []
    for result in expanded_results:
        node = subgraph.nodes.get(result.node_id)
        if node is None:
            continue
        entry = format_entry(result, node)
        if not entry:
            continue
        if getattr(result, "expansion_source", None):
            entry = f"{entry}, expanded via {result.expansion_source}"
        expanded_entries.append(entry)

    if not primary_entries and not expanded_entries:
        return NO_CONTEXT_MESSAGE

    sections = []
    if primary_entries:
        sections.append("Primary retrieved evidence:\n" + "\n\n".join(primary_entries))
    if expanded_entries:
        sections.append("Expanded 1-hop evidence:\n" + "\n\n".join(expanded_entries))

    return "\n\n".join(sections)


def _load_state_change_by_clip(video_dir: Path) -> dict:
    """
    Load clip-level state change annotations from visual.json if available.
    """
    state_index_path = Path(video_dir) / "state_changes.json"
    if state_index_path.exists():
        try:
            with open(state_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("state_change_by_clip", {})
            if isinstance(mapping, dict):
                return {
                    str(k).strip(): str(v).strip()
                    for k, v in mapping.items()
                    if str(k).strip() and str(v).strip()
                }
        except Exception:
            pass

    visual_path = Path(video_dir) / "visual.json"
    if not visual_path.exists():
        return {}

    try:
        with open(visual_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    mapping = {}
    for row in data.get("analyses", []):
        clip_id = str(row.get("clip_id", "") or "").strip()
        state_change = str(row.get("state_change_from_previous", "") or "").strip()
        if clip_id and state_change:
            mapping[clip_id] = state_change
    return mapping


def _is_empty_context(context: str) -> bool:
    """Return True when retrieval failed to produce usable evidence."""
    return not context.strip() or context.strip() == NO_CONTEXT_MESSAGE


def _parse_option(raw: str, num_options: int) -> int:
    """Parse the LLM response to extract a 0-based option index."""
    raw = raw.strip()

    # Try direct integer answer
    direct_int = re.fullmatch(r"\d+", raw)
    if direct_int:
        val = int(direct_int.group(0))
        if 0 <= val < num_options:
            return val

    # Try to find any valid integer token in the response text
    for match in re.finditer(r"\b(\d+)\b", raw):
        val = int(match.group(1))
        if 0 <= val < num_options:
            return val

    # Try letter answers (A, B, C, ...)
    for match in re.finditer(r"\b([A-Za-z])\b", raw):
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < num_options:
            return idx

    # Fallback: return 0 (will be wrong, but avoids crash)
    logger.warning(
        f"Could not parse MC answer from '{raw}' for {num_options} options, defaulting to 0"
    )
    return 0


