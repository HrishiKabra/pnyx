"""Pnyx command-line entry point.

Invoked as a module (no console-script install needed):

    python -m pnyx.cli run    --config pnyx/configs/mock.yaml
    python -m pnyx.cli status --config pnyx/configs/mock.yaml

    # P2 dataset pipeline (deterministic, no network):
    python -m pnyx.cli build-questions --out datasets/staging/
    python -m pnyx.cli merge-render    --staging datasets/staging/
    python -m pnyx.cli verify-report   --staging datasets/staging/ [--assemble]

    # P3 pilot analysis (offline, no network — reads logs a run already wrote):
    python -m pnyx.cli analyze-pilot --runs data/pilot_b20 data/pilot_b40 [--progress]

    # P5 main-run analysis (offline, no network — reads the P4 grid's logs,
    # cost ledgers, and config YAMLs; writes figures + tables + stats):
    python -m pnyx.cli analyze-main --data-root data/main --configs pnyx/configs/main \
        --out analysis_out [--bootstrap-seed 0]

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

``analyze-pilot`` aggregates one or more pilot run directories (already
produced by ``run`` — this subcommand makes no LLM calls itself) into
parse-fail / price-degeneracy / posterior-gap / Brier / cost stats; see
``pnyx.analysis.pilot``. ``--progress`` writes/replaces a "## P3 pilot"
section in PROGRESS.md.

``analyze-main`` is the P5 finale: it loads every condition of the P4 main-run
grid (``pnyx.analysis.mainrun.load_condition``), computes each one's
per-question metrics + team-rho (``mainrun.question_metrics`` /
``team_rho_summary``), the H3 adversary summary and H-wealth order effects
(``pnyx.analysis.manipulation``), then writes ``fig1.png/pdf``,
``fig2.png/pdf`` (``pnyx.analysis.figures``), ``table1.md``, ``table2.md``,
and ``stats.md`` (``pnyx.analysis.tables``) into ``--out``. This is the ONLY
place in the codebase that reads a condition's cost ledger directly and loads
its config YAML (Table 2's agent→model mapping) — every analysis module it
calls is a pure function over already-loaded data, per Global Constraints
("I/O only in the CLI layer"). Config YAML filenames are matched
case-insensitively to the condition name (the repo's real configs are
lowercase, e.g. ``D_k1.yaml``, for condition ``"D_K1"``).

Entry mechanism: ``python -m pnyx.cli`` runs this file as ``__main__`` via the
``main()`` guard below. No ``__main__.py`` and no ``[project.scripts]`` console
script are defined — module invocation keeps the package import-only and avoids
an install step for tests.
"""

import argparse
import sys
from pathlib import Path

from pnyx import dataset
from pnyx.analysis import capability, figures, manipulation
from pnyx.analysis import mainrun as mainrun_analysis
from pnyx.analysis import pilot as pilot_analysis
from pnyx.analysis import tables as tables_analysis
from pnyx.analysis.pilot import _read_costs
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

    analyze_p = sub.add_parser(
        "analyze-pilot",
        help="aggregate pilot run(s) into parse-fail/degeneracy/gap/Brier/cost stats",
    )
    analyze_p.add_argument("--runs", required=True, nargs="+",
                           help="one or more pilot run data directories, e.g. data/pilot_b20 data/pilot_b40")
    analyze_p.add_argument("--progress", action="store_true",
                           help="write/replace the '## P3 pilot' section in PROGRESS.md")
    analyze_p.add_argument("--progress-path", default="PROGRESS.md",
                           help="PROGRESS.md path (only used with --progress)")

    analyze_main_p = sub.add_parser(
        "analyze-main",
        help="aggregate the P4 main-run grid into fig1/2 + table1/2 + stats.md",
    )
    analyze_main_p.add_argument("--data-root", required=True,
                                help="P4 main-run data directory, e.g. data/main")
    analyze_main_p.add_argument("--configs", required=True,
                                help="P4 main-run config directory, e.g. pnyx/configs/main")
    analyze_main_p.add_argument("--out", required=True,
                                help="output directory for figures/tables/stats.md")
    analyze_main_p.add_argument("--bootstrap-seed", type=int, default=0,
                                help="seed threaded into every paired bootstrap (determinism)")

    analyze_v2_p = sub.add_parser(
        "analyze-v2",
        help="capability-tier + herding analysis -> tiers.md, herding.md, fig2_v2, fig3",
    )
    analyze_v2_p.add_argument("--flash", required=True,
                              help="flash tier condition dir (condition A), e.g. data/main/A")
    analyze_v2_p.add_argument("--pro", required=True,
                              help="pro tier condition dir, e.g. data/v2/A_PRO")
    analyze_v2_p.add_argument("--luna", required=True,
                              help="luna tier condition dir, e.g. data/v2/A_LUNA")
    analyze_v2_p.add_argument("--out", required=True,
                              help="output directory for tiers.md/herding.md/figures")
    analyze_v2_p.add_argument("--bootstrap-seed", type=int, default=0,
                              help="seed threaded into every paired/cluster bootstrap")

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


def _cmd_analyze_pilot(args) -> int:
    analyses = [pilot_analysis.analyze_run(Path(run_dir)) for run_dir in args.runs]
    print(pilot_analysis.format_report(analyses))
    if args.progress:
        pilot_analysis.write_progress(Path(args.progress_path), analyses)
    return 0


# ---------------------------------------------------------------------------
# analyze-main (P5)
# ---------------------------------------------------------------------------

# D_K* condition name -> its adversary bankroll multiplier (CLAUDE.md §7).
_D_K_MULTIPLIER: dict[str, int] = {"D_K1": 1, "D_K3": 3, "D_K10": 10}


def _config_path_for_condition(configs_dir: Path, condition: str) -> Path:
    """Case-insensitive match of a condition name (e.g. ``"D_K1"``,
    ``"W_FIXED"``) to its config YAML (the repo's real files are lowercase:
    ``D_k1.yaml``, ``W_fixed.yaml``)."""
    target = condition.lower()
    for p in sorted(configs_dir.glob("*.yaml")):
        if p.stem.lower() == target:
            return p
    raise FileNotFoundError(
        f"no config YAML for condition {condition!r} found in {configs_dir}"
    )


def _model_map_for_condition(configs_dir: Path, condition: str) -> dict[str, str]:
    """That condition's ``agent_id -> model_id`` mapping, read from its
    config YAML (Table 2's per-model parse-fail breakdown needs this; a
    condition's event log carries only agent_ids, never model identity)."""
    config = load_config(str(_config_path_for_condition(configs_dir, condition)))
    return {
        a.agent_id: config.models[a.model].model_id
        for a in config.agents
        if a.kind == "llm"
    }


def _condition_total_cost(data_root: Path, condition: str) -> float:
    """Sum of ``{condition}.costs.jsonl`` under ``data_root/{condition}/``
    (0.0 if the ledger doesn't exist, e.g. an all-free-tier condition that
    never billed)."""
    ledger = data_root / condition / f"{condition}.costs.jsonl"
    if not ledger.exists():
        return 0.0
    return sum(entry.cost for entry in _read_costs(ledger))


def _grand_mean_std(df, col: str) -> tuple[float, float]:
    per_seed = df.groupby("seed")[col].mean().to_numpy(dtype=float)
    return float(per_seed.mean()), float(per_seed.std())


def _cmd_analyze_main(args) -> int:
    data_root = Path(args.data_root)
    configs_dir = Path(args.configs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_seed = args.bootstrap_seed
    conditions = tables_analysis.MAIN_CONDITIONS

    loaded = {
        cond: mainrun_analysis.load_condition(data_root / cond) for cond in conditions
    }

    metrics_by_cond = {}
    for cond in conditions:
        source_name = tables_analysis.PASS1_SOURCE.get(cond)
        source = loaded[source_name] if source_name is not None else None
        metrics_by_cond[cond] = mainrun_analysis.question_metrics(loaded[cond], source)

    rho_by_cond = {
        cond: mainrun_analysis.team_rho_summary(loaded[cond])
        for cond in ("A", "B1", "B3", "C")
    }

    costs_by_cond = {
        cond: _condition_total_cost(data_root, cond) for cond in conditions
    }
    model_by_cond = {
        cond: _model_map_for_condition(configs_dir, cond) for cond in conditions
    }

    wealth_effects = manipulation.wealth_order_effects(
        metrics_by_cond["A"], metrics_by_cond["W_FIXED"], metrics_by_cond["W_SHUFFLED"],
        n_boot=10_000, seed=bootstrap_seed,
    )
    adv_dfs = {
        k: manipulation.adversary_metrics(loaded[cond], metrics_by_cond["C"])
        for cond, k in _D_K_MULTIPLIER.items()
    }
    adv_summary = manipulation.summarize_adversary(adv_dfs)

    figures.fig1(metrics_by_cond["A"], out_dir)
    figures.fig2(
        rho_by_cond,
        {c: metrics_by_cond[c] for c in ("A", "B1", "B3", "C")},
        adv_summary,
        out_dir,
    )

    (out_dir / "table1.md").write_text(
        tables_analysis.table1(metrics_by_cond, costs_by_cond)
    )
    (out_dir / "table2.md").write_text(
        tables_analysis.table2(metrics_by_cond, loaded, model_by_cond)
    )
    (out_dir / "stats.md").write_text(
        tables_analysis.stats_report(
            metrics_by_cond, rho_by_cond, adv_summary, wealth_effects,
            n_boot=10_000, bootstrap_seed=bootstrap_seed,
        )
    )

    gap_mean, gap_std = _grand_mean_std(metrics_by_cond["A"], "market_gap")
    brier_mean, brier_std = _grand_mean_std(metrics_by_cond["A"], "market_brier")
    h1_vs_mean = mainrun_analysis.compare(
        metrics_by_cond["A"], "market_gap", "mean_gap", n_boot=10_000, seed=bootstrap_seed
    )
    total_spend = sum(costs_by_cond.values())

    print("=== Pnyx P5 analysis summary ===")
    print(
        f"H1 (condition A): market gap = {gap_mean:.3f} +/- {gap_std:.3f}, "
        f"market Brier = {brier_mean:.3f} +/- {brier_std:.3f}"
    )
    print(
        f"H1 market vs. mean pool: Delta gap = {h1_vs_mean.mean_diff:.3f} "
        f"[{h1_vs_mean.ci[0]:.3f}, {h1_vs_mean.ci[1]:.3f}], "
        f"Wilcoxon p = {h1_vs_mean.wilcoxon_p:.3f}"
    )
    flip_rates = ", ".join(
        f"k={int(row.k)}: {row.flip_rate:.3f}"
        for row in adv_summary.sort_values("k").itertuples()
    )
    print(f"H3 flip rates: {flip_rates}")
    print(f"Total spend across ledgers: ${total_spend:.4f}")
    print(
        f"Wrote fig1.png/pdf, fig2.png/pdf, table1.md, table2.md, stats.md -> {out_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# analyze-v2 (capability axis + herding decomposition)
# ---------------------------------------------------------------------------

# v2 phase-map conditions: the P5 arms (loaded from the flash dir's parent when
# present) plus the two new capability tiers, mapped to their figure keys.
_V2_PHASE_MAP_SIBLINGS: tuple[str, ...] = ("B1", "B3", "C")
_V2_HERDING_N_BOOT = 2_000
_V2_TIER_N_BOOT = 10_000


def _cmd_analyze_v2(args) -> int:
    flash_dir = Path(args.flash)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.bootstrap_seed

    conds = {
        "flash": mainrun_analysis.load_condition(flash_dir),
        "pro": mainrun_analysis.load_condition(Path(args.pro)),
        "luna": mainrun_analysis.load_condition(Path(args.luna)),
    }

    tiers_df = capability.tier_metrics(
        conds, n_boot=_V2_TIER_N_BOOT, bootstrap_seed=seed
    )
    (out_dir / "tiers.md").write_text(
        capability.render_tiers_md(conds, n_boot=_V2_TIER_N_BOOT, bootstrap_seed=seed)
    )

    herding_by_tier = {
        tier: capability.herding_weights(cond, n_boot=_V2_HERDING_N_BOOT, seed=seed)
        for tier, cond in conds.items()
    }
    (out_dir / "herding.md").write_text(
        capability.render_herding_md(herding_by_tier)
    )

    figures.fig3(tiers_df, herding_by_tier, out_dir)

    # fig2_v2: the P5 phase map (condition A + any B1/B3/C siblings of the
    # flash dir) regenerated with the two new capability tiers added.
    phase_conds = {flash_dir.name: conds["flash"]}
    for name in _V2_PHASE_MAP_SIBLINGS:
        sibling = flash_dir.parent / name
        if sibling.exists():
            phase_conds[name] = mainrun_analysis.load_condition(sibling)
    phase_conds["A_PRO"] = conds["pro"]
    phase_conds["A_LUNA"] = conds["luna"]

    rho_by_cond = {
        name: mainrun_analysis.team_rho_summary(cond)
        for name, cond in phase_conds.items()
    }
    gapdiff_by_cond = {
        name: mainrun_analysis.question_metrics(cond, None)
        for name, cond in phase_conds.items()
    }
    figures.fig2_v2(rho_by_cond, gapdiff_by_cond, out_dir)

    print("=== Pnyx v2 capability-tier summary ===")
    for _i, row in tiers_df.iterrows():
        print(
            f"{row['tier']:>5}: market gap {row['market_gap']:.3f}, "
            f"pool gap {row['pool_gap']:.3f}, deficit {row['deficit']:+.3f} "
            f"(Wilcoxon p = {row['deficit_p']:.3f}), ρ = {row['rho_mean']:.3f}"
        )
    for tier, hw in herding_by_tier.items():
        pooled = hw[hw["scope"] == "pooled"].iloc[0]
        print(
            f"{tier:>5}: herding b2 (price weight) pooled = {pooled['b2']:.3f} "
            f"[{pooled['b2_lo']:.3f}, {pooled['b2_hi']:.3f}]"
        )
    print(
        f"Wrote tiers.md, herding.md, fig3.png/pdf, fig2_v2.png/pdf -> {out_dir}"
    )
    return 0


_DISPATCH = {
    "run": _cmd_run,
    "status": _cmd_status,
    "build-questions": _cmd_build_questions,
    "merge-render": _cmd_merge_render,
    "verify-report": _cmd_verify_report,
    "analyze-pilot": _cmd_analyze_pilot,
    "analyze-main": _cmd_analyze_main,
    "analyze-v2": _cmd_analyze_v2,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces a valid subcommand
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
