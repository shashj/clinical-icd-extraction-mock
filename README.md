# clinical-icd-extraction-mock

A mock clinical ICD-10 coding service that writes realistic traces to [Langfuse](https://langfuse.com) **without calling any LLM**.

It behaves like a production extraction pipeline: it fetches six prompts from Langfuse prompt management, runs them over clinical notes, and records every step with its input, output, model, token usage and a link to the prompt version. The model outputs come from a hand-written dataset of 30 cases, so you get a repeatable, production-shaped trace set for testing dashboards, prompt management, filters and evaluation.

> **All data is synthetic.** The notes are invented for this repository. They contain no real patient data and no names. Do not use this repository with real clinical records.

## The pipeline

One trace per clinical note, shaped like this:

```
icd-extraction                      chain
├─ clean-note                       generation   icd-note-cleaning
├─ extract-paragraphs               generation   icd-paragraph-extraction
├─ extract-diseases-treatments      generation   icd-disease-treatment-extraction
├─ build-query                      generation   icd-query-builder
├─ vector-search                    retriever    ICD-10-CM candidates (mock vector index)
├─ select-code                      generation   icd-code-selection
├─ validate-code                    generation   icd-code-validation
└─ select-code, validate-code       again when validation rejects the first choice
```

| Prompt | Mock model | What it does |
| --- | --- | --- |
| `icd-note-cleaning` | `gpt-4o-mini` | Removes the header and identifiers, expands abbreviations |
| `icd-paragraph-extraction` | `gemini-2.5-flash` | Keeps the paragraphs that matter for coding |
| `icd-disease-treatment-extraction` | `claude-sonnet-4-5-20250929` | Extracts disease names and what is being treated |
| `icd-query-builder` | `gpt-4o-mini` | Writes the vector search query |
| `icd-code-selection` | `gpt-4.1` | Chooses the final code from all the context |
| `icd-code-validation` | `o3-mini` | Checks the chosen code against the note and candidates |

The model names match Langfuse's built-in model definitions, so Langfuse infers cost from the token usage the service reports.

## What the traces contain

- **30 cases**, each with a five-paragraph note, the recorded output of every step, five vector candidates with similarity scores (the correct code plus real near-miss siblings), and the validation verdict. Cases cover 30 distinct, common ICD-10-CM codes. Every code and description is checked against the official [CDC/NCHS ICD-10-CM FY2027 code file](https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2027/); `data/icd10cm_subset.json` holds the codes used.
- **Interesting paths**, so the traces are not all identical:
  - 6 cases where the first selection is wrong and validation rejects it, so the pipeline selects and validates again;
  - 3 cases where the first vector search comes back empty (`WARNING`) and the query is rewritten;
  - 3 cases where the model picks a near-miss code and validation wrongly accepts it, so `icd_correct` is `false`;
  - about 3 percent of traces get an injected vector search timeout (`ERROR`) and a retry.
- **Scores** on each trace: `icd_correct` (boolean), `validation_passed` (boolean) and `selection_attempts` (numeric).
- **Metadata and identity**: `case_id`, specialty, a clinician user ID, an encounter session ID, and tags.
- Every request is one trace with a real root: the `icd-extraction` span has no parent, and every step is a child of it. Each run creates new traces, because the SDK generates trace and span IDs. The run ID (`--run-id`, random by default) is stored in the metadata and session IDs of that run's traces, so you can filter a run.

## Setup

Requires Python 3.11 or newer and a running Langfuse instance (Langfuse Cloud or self-hosted).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in your project keys
```

`.env`:

```
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Usage

```bash
icd-mock validate-dataset            # check the dataset and its ICD codes
icd-mock prompts sync                # create or update the six prompts in Langfuse (production label)
icd-mock run --traces 1000 --seed 42 # write 1000 traces
icd-mock run --dry-run               # run the pipeline without sending anything
```

`run` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--traces` | 1000 | Number of traces. Case `i % 30` is used for trace `i`. |
| `--seed` | 42 | Seeds users, token counts and which traces get a timeout. |
| `--run-id` | random | Labels one run: it is in the session IDs and metadata of that run's traces. |
| `--flush-every` | 25 | Flush to Langfuse every N traces. The SDK drops spans when its export queue is full, so a fast loop must flush. |
| `--start-index` | 0 | Index of the first trace, to append more traces with the same seed. |
| `--latency-scale` | 0.01 | Sleep this fraction of each step's recorded latency. `1` is real time, `0` writes as fast as possible. |
| `--dry-run` | off | Use local prompts and send nothing. |

The Langfuse SDK takes timestamps from the clock, so traces are written at the time you run the command, and step durations are the recorded latency scaled by `--latency-scale`. Every step also starts a few milliseconds after the previous one, so Langfuse shows the steps in order.

## How it works

- `src/icd_mock/pipeline.py` runs the steps and records them with the Langfuse Python SDK v3. Each generation passes the prompt client from `get_prompt`, which is what links the generation to its prompt version.
- `src/icd_mock/mock_llm.py` returns the recorded text with usage derived from its length. Nothing contacts a model provider.
- `src/icd_mock/vector_db.py` returns the recorded candidates, with official descriptions.
- `data/cases/*.json` is the hand-written dataset. `scripts/build_code_subset.py` rebuilds `data/icd10cm_subset.json` from the official code file.

## Development

```bash
pytest
ruff check .
```

## License

MIT. The ICD-10-CM code descriptions come from the U.S. Centers for Disease Control and Prevention (NCHS) and are in the public domain.
