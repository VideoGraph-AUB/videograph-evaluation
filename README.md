# VideoGraph Evaluation

Benchmark evaluation pipeline for VideoGraph.

Runs end-to-end evaluation on EgoSchema, NExT-QA, and Video-MME: processes
videos, builds knowledge graphs, answers multiple-choice questions, and produces
accuracy reports with optional API cost tracking.

Depends on `videograph-core` for graph construction, retrieval, and answering.

## Prerequisites

- **Python >= 3.10**
- **FFmpeg** and **FFprobe** on `PATH` (required by `videograph-core` for video processing)
- **OpenAI API key**

## Install

For local development with `videograph-core` as a sibling directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../videograph-core
pip install -e .
cp .env.example .env         # then set OPENAI_API_KEY
```

On a remote machine, install both packages from their paths or git URLs:

```bash
pip install -e /path/to/videograph-core
pip install -e /path/to/videograph-evaluation
```

## Expected Data Layout

Place videos and QA files under a single `--data-dir`:

```text
<data-dir>/
  EgoSchema Subset Videos/       # .mp4 files
  EgoSchema Subset QAs.json
  NExT-QA Val Videos/
  NExT-QA Val QAs.csv
  NExT-QA Test Videos/
  NExT-QA Test QAs.csv
  VIDEO-MME Short Videos/
  VIDEO-MME Short QAs.csv
  VIDEO-MME Medium Videos/
  VIDEO-MME Medium QAs.csv
  VIDEO-MME Long Videos/
  VIDEO-MME Long QAs.csv
```

## QA File Schemas

The evaluator expects multiple-choice questions. Internally, answers are stored
as **0-based option indices**.

### EgoSchema

`EgoSchema Subset QAs.json` must be a JSON list. Each item must contain:

```json
{
  "q_uid": "0074f737-11cb-497d-8d07-77c3a8127391",
  "question": "Taking into account all the actions...",
  "option 0": "C is cooking.",
  "option 1": "C is doing laundry.",
  "option 2": "C is cleaning the kitchen.",
  "option 3": "C is cleaning dishes.",
  "option 4": "C is cleaning the bathroom.",
  "_answer": 3
}
```

Required fields:

```text
q_uid, question, option 0, option 1, option 2, option 3, option 4, _answer
```

Video files are matched by `q_uid`, so the corresponding video must be:

```text
EgoSchema Subset Videos/<q_uid>.mp4
```

`_answer` must be a 0-based integer from `0` to `4`.

### NExT-QA

`NExT-QA Val QAs.csv` and `NExT-QA Test QAs.csv` must contain:

```csv
video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
4010069381,369,640,480,how do the two man play the instrument,0,6,CH,roll the handle,tap their feet,strum the string,hit with sticks,...
```

Required fields:

```text
video, question, answer, qid, type, a0, a1, a2, a3, a4
```

`frame_count`, `width`, and `height` may be present but are not used by the
loader. Video files are matched by `video`, so the corresponding video must be:

```text
NExT-QA Val Videos/<video>.mp4
NExT-QA Test Videos/<video>.mp4
```

`answer` must be a 0-based integer from `0` to `4`. `type` is used for the
NExT-QA causal/temporal/descriptive breakdown.

### VIDEO-MME

`VIDEO-MME Short QAs.csv`, `VIDEO-MME Medium QAs.csv`, and
`VIDEO-MME Long QAs.csv` must contain:

```csv
video_id,duration,domain,sub_category,url,videoID,question_id,task_type,question,optionA,optionB,optionC,optionD,answer
001,short,Knowledge,Humanity & History,https://www.youtube.com/watch?v=fFjv93ACGo8,fFjv93ACGo8,001-1,Counting Problem,When demonstrating...,Apples.,Candles.,Berries.,...,A
```

Required fields:

```text
question, answer
```

And at least one video-id field:

```text
videoID, video_id, or video
```

And option columns named like either:

```text
optionA, optionB, optionC, optionD
```

or:

```text
option0, option1, option2, option3
```

The loader uses `videoID` first when present, then falls back to `video_id` or
`video`. Video files are matched by that resolved ID:

```text
VIDEO-MME Short Videos/<videoID>.mp4
VIDEO-MME Medium Videos/<videoID>.mp4
VIDEO-MME Long Videos/<videoID>.mp4
```

`answer` may be a letter label (`A`, `B`, `C`, `D`), a numeric index, or the full
answer text. Numeric answers support both 0-based and 1-based values when they
can be resolved unambiguously.

## Run

```bash
python -m videograph_eval.run \
  --data-dir /data \
  --output-dir /results/v1.0 \
  --version v1.0 \
  --datasets nextqa-val
```

Options:

| Flag | Description |
|------|-------------|
| `--datasets` | Subset of datasets to run (default: all) |
| `--max-videos N` | Limit videos per dataset (for debugging) |
| `--skip-processing` | Skip graph building, only run QA on existing graphs |
| `--cleanup` | Delete intermediate files (frames, clips, audio) after each video |
| `--track-performance` | Disable cache and track API calls, cost, and timing |
| `--max-parallel-vision N` | Override parallel workers for vision captioning/OCR |

## Outputs

Results are written under `--output-dir`:

```text
<output-dir>/
  graphs/<dataset>/<video_id>/   # graph.json + embeddings.json per video
  predictions/<dataset>.json     # per-question predictions
  predictions/<dataset>_qa_trace.md  # detailed QA trace with retrieval context
  results.json                   # combined accuracy and performance metrics
  report.md                      # human-readable summary report
```

## NExT-GQA Grounding Evaluation

`run_nextgqa_grounding.py` evaluates temporal grounding on NExT-GQA by comparing
flat retrieval against graph-hop retrieval.

Three conditions are evaluated:

- **flat** — seed-only retrieval, no graph traversal (`hop_expansion=0`)
- **graph-1** — seeds + 1-hop neighbours (`hop_expansion=1`)
- **graph-2** — seeds + 2-hop neighbourhood (`hop_expansion=2`)

### Data layout

Place the NExT-GQA annotation files under `--data-dir`:

```text
data/
  NExT-GQA/
    val.csv
    test.csv
    gsub_val.json
    gsub_test.json
```

`val.csv` / `test.csv` are the NExT-GQA question splits. The `answer` column is
answer **text** (not a 0-based index); the script matches it against `a0`–`a4` to
resolve the correct option.
`gsub_val.json` / `gsub_test.json` contain the gold temporal grounding intervals.

### Graph layout

`--graphs-dir` must point to a directory with one sub-folder per video ID, each
containing `graph.json`. Graphs are produced by the main pipeline
(`videograph_eval.run --datasets nextqa-val`):

```text
# val split
results/v1/graphs/nextqa-val/
  <video_id>/
    graph.json

# test split
results/v1_test/graphs/nextqa-test/
  <video_id>/
    graph.json
```

### Example command

Run from the `videograph-evaluation/` directory using the project venv:

```bash
python run_nextgqa_grounding.py \
  --split val \
  --graphs-dir results/v1/graphs/nextqa-val \
  --data-dir data \
  --output-dir results/nextgqa_val \
  --retrieval-only \
  --verify-seed-order
```

Smoke-test with a small subset and `--top-k` override:

```bash
python run_nextgqa_grounding.py \
  --split val \
  --graphs-dir results/v1/graphs/nextqa-val \
  --data-dir data \
  --output-dir results/smoke \
  --max-questions 20 \
  --retrieval-only \
  --top-k 2 \
  --verify-seed-order
```

### Options

| Flag | Description |
|------|-------------|
| `--split val\|test` | Dataset split to evaluate |
| `--graphs-dir PATH` | Directory containing per-video `graph.json` files |
| `--data-dir PATH` | Root data directory (must contain `NExT-GQA/`) |
| `--output-dir PATH` | Output directory for JSONL and summary files |
| `--retrieval-only` | Skip LLM answering; compute grounding metrics only |
| `--top-k N` | Override seed retrieval budget for all conditions |
| `--verify-seed-order` | Verify seed node IDs and `hit_at_seed_budget` are identical across all conditions |
| `--node-types transcript+visual\|visual-only` | Node types included in grounding metrics (default: both) |
| `--inspect-first-grounded` | Print alignment inspection for the first question with a graph and gold interval, then exit |
| `--inspect-question VIDEO_ID:QID` | Print alignment inspection for a specific question, then exit |

### Outputs

All outputs are written under `--output-dir` and should not be committed:

```text
<output-dir>/
  flat/detailed.jsonl      # per-question records for each condition
  graph-1/detailed.jsonl
  graph-2/detailed.jsonl
  summary.json             # aggregate metrics across all conditions
  summary.md               # human-readable report
```

### Metrics

Grounding is evaluated on `TranscriptNode` and `VisualNode` only. `TopicNode` and
`EntityNode` are excluded from localization metrics (they span the whole video or
carry no fine-grained temporal annotation) but may still appear in the LLM
answering context. The predicted interval is the span of the top-1 retrieved node;
no global union span is used.

Key retrieval recall metrics:

| Metric | Definition |
|--------|------------|
| `hit_at_seed_budget` | Any of the top-`k` seed-only nodes hits gold (`k = --top-k`) |
| `hit_beyond_seed_budget` | A hit exists beyond the seed budget (`hit_any AND NOT hit_at_seed_budget`) |
| `hit_expanded_only` | A graph-expanded node hits gold after the seed budget misses; always 0 for flat |

`--verify-seed-order` confirms that seed node IDs and `hit_at_seed_budget` are
identical across flat / graph-1 / graph-2 for every question, validating the
fair flat-vs-graph comparison.

## Repository Boundary

This repo contains only the public QA evaluation pipeline. It depends on
`videograph-core` for all graph construction and retrieval logic.

