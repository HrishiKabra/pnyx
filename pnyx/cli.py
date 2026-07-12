"""Pnyx command-line entry point.

Invoked as a module (no console-script install needed):

    python -m pnyx.cli run    --config pnyx/configs/mock.yaml
    python -m pnyx.cli status --config pnyx/configs/mock.yaml

``run`` executes or resumes the experiment (resume is the default — there is no
resume flag; rerunning the same command after any crash continues from the last
completed turn). ``status`` prints turns done/remaining per condition and
parse-failure counts (the cost ledger arrives in P3, so a $0.00 placeholder is
printed for now).

``--data-dir`` overrides the config's ``data_dir`` (used by the kill-resume
test to point a run at a temp directory without editing the YAML).

Entry mechanism: ``python -m pnyx.cli`` runs this file as ``__main__`` via the
``main()`` guard below. No ``__main__.py`` and no ``[project.scripts]`` console
script are defined — module invocation keeps the package import-only and avoids
an install step for tests.
"""

import argparse
import sys

from pnyx.runner import load_config, run_experiment, status_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pnyx", description="Pnyx experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run or resume the experiment")
    run_p.add_argument("--config", required=True, help="path to a run config YAML")
    run_p.add_argument("--data-dir", default=None,
                       help="override the config's data_dir (log/output directory)")

    status_p = sub.add_parser("status", help="print turns done/remaining")
    status_p.add_argument("--config", required=True, help="path to a run config YAML")
    status_p.add_argument("--data-dir", default=None,
                          help="override the config's data_dir")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.data_dir is not None:
        config = config.model_copy(update={"data_dir": args.data_dir})

    if args.command == "run":
        run_experiment(config)
    elif args.command == "status":
        print(status_report(config))
    else:  # pragma: no cover - argparse enforces a valid subcommand
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
