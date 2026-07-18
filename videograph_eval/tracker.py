"""
API call tracker for measuring LLM usage during graph building.

Uses monkey-patching to intercept OpenAI API calls and record
call counts, token usage, estimated cost, and wall-clock time.
"""

import logging
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    # GPT-4o
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Whisper
    "whisper-1": {"per_minute": 0.006},
    "whisper-large-v3": {"per_minute": 0.0015},
    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


@dataclass
class APICall:
    """Record of a single API call."""
    call_type: str  # "chat", "transcription", "embedding"
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float
    timestamp: float
    audio_duration_s: float = 0.0


@dataclass
class TrackerStats:
    """Aggregated tracking statistics."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    calls_by_type: Dict[str, int] = field(default_factory=dict)
    cost_by_type: Dict[str, float] = field(default_factory=dict)
    duration_by_type: Dict[str, float] = field(default_factory=dict)
    input_tokens_by_type: Dict[str, int] = field(default_factory=dict)
    output_tokens_by_type: Dict[str, int] = field(default_factory=dict)
    audio_duration_by_type: Dict[str, float] = field(default_factory=dict)
    calls_by_stage: Dict[str, int] = field(default_factory=dict)
    cost_by_stage: Dict[str, float] = field(default_factory=dict)
    duration_by_stage: Dict[str, float] = field(default_factory=dict)
    input_tokens_by_stage: Dict[str, int] = field(default_factory=dict)
    output_tokens_by_stage: Dict[str, int] = field(default_factory=dict)
    audio_duration_by_stage: Dict[str, float] = field(default_factory=dict)
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    calls_by_stage_and_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    cost_by_stage_and_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_duration_s": round(self.total_duration_s, 2),
            "total_audio_duration_s": round(self.total_audio_duration_s, 2),
            "calls_by_type": self.calls_by_type,
            "cost_by_type": {k: round(v, 6) for k, v in self.cost_by_type.items()},
            "duration_by_type": {k: round(v, 2) for k, v in self.duration_by_type.items()},
            "input_tokens_by_type": self.input_tokens_by_type,
            "output_tokens_by_type": self.output_tokens_by_type,
            "audio_duration_by_type": {
                k: round(v, 2) for k, v in self.audio_duration_by_type.items()
            },
            "calls_by_stage": self.calls_by_stage,
            "cost_by_stage": {k: round(v, 6) for k, v in self.cost_by_stage.items()},
            "duration_by_stage": {
                k: round(v, 2) for k, v in self.duration_by_stage.items()
            },
            "input_tokens_by_stage": self.input_tokens_by_stage,
            "output_tokens_by_stage": self.output_tokens_by_stage,
            "audio_duration_by_stage": {
                k: round(v, 2) for k, v in self.audio_duration_by_stage.items()
            },
            "calls_by_model": self.calls_by_model,
            "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
            "calls_by_stage_and_type": self.calls_by_stage_and_type,
            "cost_by_stage_and_type": {
                stage: {call_type: round(cost, 6) for call_type, cost in by_type.items()}
                for stage, by_type in self.cost_by_stage_and_type.items()
            },
        }


def _pricing_for_model(model: str) -> Optional[dict]:
    model = model.split("/", 1)[-1]
    pricing = MODEL_PRICING.get(model)
    if pricing:
        return pricing

    # Match date-stamped snapshots such as gpt-4o-2024-08-06 without
    # accidentally treating gpt-4o-mini as gpt-4o.
    for key in MODEL_PRICING:
        if model.startswith(f"{key}-20"):
            return MODEL_PRICING[key]
    return None


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    audio_duration_s: float = 0.0,
) -> float:
    """Estimate cost in USD for a given API call."""
    pricing = _pricing_for_model(model)
    if not pricing:
        logger.warning(f"No pricing data for model '{model}', cost will be 0")
        return 0.0

    if "per_minute" in pricing:
        return (audio_duration_s / 60.0) * pricing["per_minute"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def _nested_increment(target: Dict[str, Dict[str, Any]], outer: str, inner: str, value: Any):
    target.setdefault(outer, {})
    target[outer][inner] = target[outer].get(inner, 0) + value


def _audio_file_path(file_arg: Any) -> Optional[Path]:
    if file_arg is None:
        return None

    if isinstance(file_arg, (str, Path)):
        return Path(file_arg)

    name = getattr(file_arg, "name", None)
    if name:
        return Path(name)

    if isinstance(file_arg, tuple):
        for item in file_arg:
            path = _audio_file_path(item)
            if path is not None:
                return path

    return None


def _probe_audio_duration_s(file_arg: Any) -> float:
    path = _audio_file_path(file_arg)
    if path is None or not path.exists():
        return 0.0

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration if duration > 0 else 0.0
    except Exception as exc:
        logger.warning(f"Could not probe audio duration for {path}: {exc}")

    return 0.0


class APITracker:
    """
    Context manager that tracks OpenAI API calls via monkey-patching.

    Usage:
        tracker = APITracker()
        with tracker:
            # All OpenAI calls in this block are tracked
            ...
        stats = tracker.get_stats()
    """

    def __init__(self):
        self.calls: List[APICall] = []
        self._originals: Dict[str, Any] = {}
        self._active = False
        self._start_time: float = 0.0
        self._stage = "unknown"
        self._lock = Lock()

    def __enter__(self):
        self._active = True
        self._start_time = time.time()
        self._patch()
        return self

    def __exit__(self, *args):
        self._unpatch()
        self._active = False

    @contextmanager
    def stage(self, name: str):
        """Temporarily label tracked API calls with a pipeline stage."""
        previous = self._stage
        self._stage = name
        try:
            yield
        finally:
            self._stage = previous

    def _record_call(self, call: APICall):
        with self._lock:
            self.calls.append(call)

    def _patch(self):
        """Monkey-patch OpenAI client methods to intercept calls."""
        try:
            import openai.resources.chat.completions
            import openai.resources.embeddings
            import openai.resources.audio.transcriptions

            # Patch chat completions
            orig_chat = openai.resources.chat.completions.Completions.create
            self._originals["chat_create"] = orig_chat

            tracker_ref = self

            def tracked_chat_create(self_inner, *args, **kwargs):
                start = time.time()
                response = orig_chat(self_inner, *args, **kwargs)
                duration = time.time() - start

                model = kwargs.get("model", getattr(response, "model", "unknown"))
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                cost = _estimate_cost(model, input_tokens, output_tokens)

                tracker_ref._record_call(APICall(
                    call_type="chat",
                    stage=tracker_ref._stage,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    duration_s=duration,
                    timestamp=time.time(),
                ))
                return response

            openai.resources.chat.completions.Completions.create = tracked_chat_create

            # Patch embeddings
            orig_embed = openai.resources.embeddings.Embeddings.create
            self._originals["embed_create"] = orig_embed

            def tracked_embed_create(self_inner, *args, **kwargs):
                start = time.time()
                response = orig_embed(self_inner, *args, **kwargs)
                duration = time.time() - start

                model = kwargs.get("model", getattr(response, "model", "unknown"))
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = 0
                cost = _estimate_cost(model, input_tokens, output_tokens)

                tracker_ref._record_call(APICall(
                    call_type="embedding",
                    stage=tracker_ref._stage,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    duration_s=duration,
                    timestamp=time.time(),
                ))
                return response

            openai.resources.embeddings.Embeddings.create = tracked_embed_create

            # Patch audio transcription
            orig_audio = openai.resources.audio.transcriptions.Transcriptions.create
            self._originals["audio_create"] = orig_audio

            def tracked_audio_create(self_inner, *args, **kwargs):
                start = time.time()
                response = orig_audio(self_inner, *args, **kwargs)
                duration = time.time() - start

                model = kwargs.get("model", "whisper-1")
                audio_duration_s = _probe_audio_duration_s(kwargs.get("file"))
                cost = _estimate_cost(
                    model,
                    input_tokens=0,
                    output_tokens=0,
                    audio_duration_s=audio_duration_s,
                )

                tracker_ref._record_call(APICall(
                    call_type="transcription",
                    stage=tracker_ref._stage,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=cost,
                    duration_s=duration,
                    timestamp=time.time(),
                    audio_duration_s=audio_duration_s,
                ))
                return response

            openai.resources.audio.transcriptions.Transcriptions.create = tracked_audio_create

            logger.info("APITracker: patched OpenAI methods for tracking")

        except ImportError:
            logger.warning("APITracker: could not import openai, tracking disabled")

    def _unpatch(self):
        """Restore original OpenAI methods."""
        try:
            import openai.resources.chat.completions
            import openai.resources.embeddings
            import openai.resources.audio.transcriptions

            if "chat_create" in self._originals:
                openai.resources.chat.completions.Completions.create = self._originals["chat_create"]
            if "embed_create" in self._originals:
                openai.resources.embeddings.Embeddings.create = self._originals["embed_create"]
            if "audio_create" in self._originals:
                openai.resources.audio.transcriptions.Transcriptions.create = self._originals["audio_create"]

            logger.info("APITracker: restored original OpenAI methods")
        except ImportError:
            pass

    def get_stats(self) -> TrackerStats:
        """Compute aggregated statistics from all tracked calls."""
        stats = TrackerStats()
        for call in self.calls:
            stats.total_calls += 1
            stats.total_input_tokens += call.input_tokens
            stats.total_output_tokens += call.output_tokens
            stats.total_cost_usd += call.cost_usd
            stats.total_duration_s += call.duration_s
            stats.total_audio_duration_s += call.audio_duration_s
            stats.calls_by_type[call.call_type] = stats.calls_by_type.get(call.call_type, 0) + 1
            stats.cost_by_type[call.call_type] = stats.cost_by_type.get(call.call_type, 0.0) + call.cost_usd
            stats.duration_by_type[call.call_type] = stats.duration_by_type.get(call.call_type, 0.0) + call.duration_s
            stats.input_tokens_by_type[call.call_type] = stats.input_tokens_by_type.get(call.call_type, 0) + call.input_tokens
            stats.output_tokens_by_type[call.call_type] = stats.output_tokens_by_type.get(call.call_type, 0) + call.output_tokens
            stats.audio_duration_by_type[call.call_type] = stats.audio_duration_by_type.get(call.call_type, 0.0) + call.audio_duration_s
            stats.calls_by_stage[call.stage] = stats.calls_by_stage.get(call.stage, 0) + 1
            stats.cost_by_stage[call.stage] = stats.cost_by_stage.get(call.stage, 0.0) + call.cost_usd
            stats.duration_by_stage[call.stage] = stats.duration_by_stage.get(call.stage, 0.0) + call.duration_s
            stats.input_tokens_by_stage[call.stage] = stats.input_tokens_by_stage.get(call.stage, 0) + call.input_tokens
            stats.output_tokens_by_stage[call.stage] = stats.output_tokens_by_stage.get(call.stage, 0) + call.output_tokens
            stats.audio_duration_by_stage[call.stage] = stats.audio_duration_by_stage.get(call.stage, 0.0) + call.audio_duration_s
            stats.calls_by_model[call.model] = stats.calls_by_model.get(call.model, 0) + 1
            stats.cost_by_model[call.model] = stats.cost_by_model.get(call.model, 0.0) + call.cost_usd
            _nested_increment(stats.calls_by_stage_and_type, call.stage, call.call_type, 1)
            _nested_increment(stats.cost_by_stage_and_type, call.stage, call.call_type, call.cost_usd)
        return stats

    def get_wall_time(self) -> float:
        """Get total wall-clock time since tracker was activated."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0


