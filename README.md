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

## Repository Boundary

This repo contains only the public QA evaluation pipeline. It depends on
`videograph-core` for all graph construction and retrieval logic.

