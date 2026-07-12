"""Pnyx command-line entry point.

Invoked as a module (no console-script install needed):

    python -m pnyx.cli run    --config pnyx/configs/mock.yaml
    python -m pnyx.cli status --config pnyx/configs/mock.yaml

    # P2 dataset pipeline (deterministic, no network):
    python -m pnyx.cli build-questions --out datasets/staging/
    python -m pnyx.cli merge-render    --staging datasets/staging/
    python -m pnyx.cli verify-report   --staging datasets/staging/ [--assemble]

``run`` executes or resumes the experiment (resume is the default — there is no
resume flag; rerunning the same command after any crash continues from the last
completed turn). ``status`` prints turns done/remaining per condition and
parse-failure counts (the cost ledger arrives in P3, so a $0.00 placeholder is
printed for now).

The three dataset subcommands drive the P2 build -> render -> verify pipeline;
the rendering/verification themselves are done by Claude subagents OUTSIDE this
codebase writing files into the staging dir. ``merge-render`` and
``verify-report`` exit nonzero when anything is missing/invalid/failing so they
compose as CI-style gates.

``--data-dir`` overrides the config's ``data_dir`` (used by the kill-resume
test to point a run at a temp directory without editing the YAML).

Entry mechanism: ``python -m pnyx.cli`` runs this file as ``__main__`` via the
``main()`` guard below. No ``__main__.py`` and no ``[project.scripts]`` console
script are defined — module invocation keeps the package import-only and avoids
an install step for tests.
"""

import argparse
import sys

from pnyx import dataset
from pnyx.runner import load_config, run_experiment, status_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pnyx", description="Pnyx experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run or resume the experiment")
    run_p.add_argument("--config", required=True, help="path to a run config YAML")
    run_p.add_argument("--data-dir", default=None,
                       help="override the config's data_dir (log/output directory)")
    run_p.add_argument("--override-budget", action="store_true",
                       help="keep making LLM calls past the config budget_usd hard-stop")

    status_p = sub.add_parser("status", help="print turns done/remaining")
    status_p.add_argument("--config", required=True, help="path to a run config YAML")
    status_p.add_argument("--data-dir", default=None,
                          help="override the config's data_dir")

    build_p = sub.add_parser("build-questions",
                             help="generate the 50-question dataset + render specs")
    build_p.add_argument("--out", required=True, help="staging output directory")

    merge_p = sub.add_parser("merge-render",
                             help="validate + merge rendered/*.json into questions_rendered.jsonl")
    merge_p.add_argument("--staging", required=True, help="staging directory")

    verify_p = sub.add_parser("verify-report",
                              help="aggregate verdicts/*.json; --assemble writes the final datasets")
    verify_p.add_argument("--staging", required=True, help="staging directory")
    verify_p.add_argument("--assemble", action="store_true",
                          help="if all questions pass, split into datasets/questions[_pilot]_v1.jsonl")
    verify_p.add_argument("--datasets-dir", default="datasets",
                          help="output dir for the assembled dataset files")
    verify_p.add_argument("--progress", default="PROGRESS.md",
                          help="PROGRESS.md path to append the summary to on --assemble")

    return parser


def _cmd_run(args) -> int:
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir})
    run_experiment(config, override_budget=args.override_budget)
    return 0


def _cmd_status(args) -> int:
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir})
    print(status_report(config))
    return 0


def _cmd_build_questions(args) -> int:
    stats = dataset.build_questions(args.out)
    print(f"built {stats.n_questions} questions "
          f"({stats.n_easy} easy + {stats.n_hard} hard) -> {stats.out_dir}")
    return 0


def _cmd_merge_render(args) -> int:
    stats = dataset.merge_render(args.staging)
    print(f"merged {stats.n_merged} rendered questions")
    if stats.missing:
        print(f"MISSING ({len(stats.missing)}): {', '.join(stats.missing)}")
    if stats.invalid:
        for qid, reason in stats.invalid.items():
            print(f"INVALID {qid}: {reason}")
    return 0 if stats.ok else 1


def _cmd_verify_report(args) -> int:
    stats = dataset.verify_report(
        args.staging,
        assemble=args.assemble,
        datasets_dir=args.datasets_dir,
        progress_path=args.progress if args.assemble else None,
    )
    print(dataset.summary_table(stats))
    if args.assemble and not stats.assembled:
        return 1
    return 0 if stats.all_pass else 1


_DISPATCH = {
    "run": _cmd_run,
    "status": _cmd_status,
    "build-questions": _cmd_build_questions,
    "merge-render": _cmd_merge_render,
    "verify-report": _cmd_verify_report,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces a valid subcommand
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
