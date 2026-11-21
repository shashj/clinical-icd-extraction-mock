"""Command line: sync the prompts, run the traces, validate the dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from collections.abc import Sequence

from langfuse import Langfuse

from .config import load_settings, make_client, make_offline_client
from .dataset import ICD_RELEASE, load_cases, load_code_descriptions, used_codes
from .pipeline import IcdPipeline
from .prompts import SPECS, LangfusePromptStore, LocalPromptStore, sync_prompts


def _cmd_validate(_: argparse.Namespace) -> int:
    cases = load_cases()
    descriptions = load_code_descriptions()
    missing = sorted(used_codes(cases) - descriptions.keys())
    if missing:
        print(f"Codes missing from data/icd10cm_subset.json: {', '.join(missing)}")
        return 1
    print(
        json.dumps(
            {
                "cases": len(cases),
                "release": ICD_RELEASE,
                "distinct_expected_codes": len({c.expected_code for c in cases}),
                "codes_checked": len(used_codes(cases)),
                "validation_loops": sum(c.is_loop for c in cases),
                "retrieval_misses": sum(c.is_retrieval_miss for c in cases),
                "wrong_final_selections": sum(c.is_wrong_final for c in cases),
            },
            indent=2,
        )
    )
    return 0


def _authenticate(langfuse: Langfuse, host: str) -> bool:
    try:
        if langfuse.auth_check():
            return True
        print(f"Langfuse rejected the keys at {host}.")
    except Exception as exc:  # noqa: BLE001 - any transport failure means we cannot reach it
        print(f"Cannot reach Langfuse at {host}: {exc.__class__.__name__}. Is it running?")
    return False


def _cmd_prompts_sync(_: argparse.Namespace) -> int:
    settings = load_settings()
    langfuse = make_client(settings)
    if not _authenticate(langfuse, settings.host):
        return 1
    result = sync_prompts(langfuse)
    langfuse.flush()
    for name, status in result.items():
        print(f"{status:10} {name}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases()
    if args.dry_run:
        langfuse = make_offline_client()
        prompts: LangfusePromptStore | LocalPromptStore = LocalPromptStore()
        target = "dry run (nothing is sent)"
    else:
        settings = load_settings()
        langfuse = make_client(settings)
        if not _authenticate(langfuse, settings.host):
            return 1
        prompts = LangfusePromptStore(langfuse)
        for spec in SPECS:
            try:
                prompts.get(spec.name)
            except Exception:  # noqa: BLE001
                print(f"Prompt {spec.name!r} not found. Run `icd-mock prompts sync` first.")
                return 1
        target = settings.host

    run_id = args.run_id or uuid.uuid4().hex[:8]
    pipeline = IcdPipeline(
        langfuse, prompts, seed=args.seed, run_id=run_id, latency_scale=args.latency_scale
    )
    tally: Counter[str] = Counter()
    started = time.time()
    for offset in range(args.traces):
        index = args.start_index + offset
        case = cases[index % len(cases)]
        result = pipeline.run_case(case, index)
        tally["traces"] += 1
        tally["observations"] += len(result.steps)
        for kind in ("generation", "retriever", "chain"):
            tally[f"{kind}s"] += result.count(kind)
        tally["selection_loops"] += result.selection_attempts > 1
        tally["retrieval_misses"] += case.is_retrieval_miss
        tally["injected_timeouts"] += result.timeout_injected
        tally["icd_correct"] += result.correct
        tally["icd_incorrect"] += not result.correct
        # The SDK exports spans in the background through a bounded queue and drops spans
        # when the queue is full. Flushing regularly keeps a fast producer from losing data.
        if args.flush_every and (offset + 1) % args.flush_every == 0:
            langfuse.flush()
        if (offset + 1) % 100 == 0:
            print(f"  {offset + 1}/{args.traces} traces", file=sys.stderr)

    langfuse.flush()
    summary = {
        "target": target,
        "seed": args.seed,
        "run_id": run_id,
        "elapsed_seconds": round(time.time() - started, 1),
        **tally,
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="icd-mock",
        description="A mock clinical ICD-10 extraction service that writes traces to Langfuse.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-dataset", help="check the dataset and its ICD codes").set_defaults(
        func=_cmd_validate
    )

    prompts = sub.add_parser("prompts", help="manage the six prompts in Langfuse")
    prompts_sub = prompts.add_subparsers(dest="prompts_command", required=True)
    prompts_sub.add_parser("sync", help="create or update the prompts").set_defaults(
        func=_cmd_prompts_sync
    )

    run = sub.add_parser("run", help="run the pipeline and write traces")
    run.add_argument("--traces", type=int, default=1000, help="number of traces (default 1000)")
    run.add_argument("--seed", type=int, default=42, help="seed for users, tokens and timeouts")
    run.add_argument(
        "--run-id",
        default=None,
        help="label for this run, stored in session ids and trace metadata (default: random)",
    )
    run.add_argument("--start-index", type=int, default=0, help="index of the first trace")
    run.add_argument(
        "--latency-scale",
        type=float,
        default=0.01,
        help="sleep this fraction of each step's recorded latency (default 0.01; 1 is real time)",
    )
    run.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="flush to Langfuse every N traces so no spans are dropped (0 disables)",
    )
    run.add_argument("--dry-run", action="store_true", help="run without sending anything")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
