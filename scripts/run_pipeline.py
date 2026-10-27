#!/usr/bin/env python
"""Command-line entry point for the whole pipeline.

Steps (see docs/METHODOLOGY_MAPPING.md for the step -> module mapping):

  data       Step 1  download InfoDengue + NASA POWER + IBGE metadata, build the
                     processed tables, EDA and smoke-test baselines, El Nino index
                     (ONI) (30-45 min); --with-chirps adds CHIRPS rainfall (1-2 h)
  reproduce  Step 2  fetch the Lim et al. (2026) CV mask (data only) and run the
                     MAP reproduction (seconds); add --nuts for NUTS (~20 min/fold)
  baselines  Step 3  B0, B1, B1p and the no-climate DM models B4, B4s (LOYO)
  brazil     Steps 3-7  full LOYO experiment: baselines, B4, B4s, M1, ablations,
                     shuffles, checks, scoring (weekly 60: ~22 min on 4 CPUs)
  m2         Step 5  optional boosted softmax M2 (needs the brazil run first)
  extension  Step 8  pre-registered rain-lag extension (M1-R, M1-RTH)
  report     Step 9  tables, sanity-check table and figures from saved results
  official   official experiment: design A (expanding window, primary), B (learning
                     curve) or C (LOYO), with ONI features and M2 early stopping;
                     checkpoints per fold (rerun resumes). Not part of 'all'.
  official-report    pre-registered statistics, tables and figures of the official
                     runs (results/official/{tables,figures}); needs the local
                     metrics_detail files (and fold checkpoints for coefficients)
  all        everything above in order (except 'official')

Options:
  --subset {60,300}   municipalities (pilot subset or all)
  --res {weekly,monthly}
  --quick             main models only (B4, B4s, M1) at weekly resolution, written
                      under a '_quick' tag so saved full results are not overwritten
  --years 2016,2019   restrict LOYO folds / official test years (smoke tests only)
  --design {A,B,C}    official: validation design
  --climate {power,chirps}  official: weather table (NASA POWER primary, CHIRPS robustness)
  --exclude-year 2020 official: sensitivity run without that year
  --models B1,M1      official: subset of models (default: the design's full set)
  --out DIR           official: output folder (default RESULTS_DIR/official)
  --with-chirps       data: also download CHIRPS rainfall (needed for --climate chirps)

Paths come from environment variables (DATA_DIR, RESULTS_DIR, REPORTS_DIR,
LIM_MASK_PATH, DENGUE_ROOT); see src/dengue_recon/config.py.

Examples:
  python scripts/run_pipeline.py --step data
  python scripts/run_pipeline.py --step reproduce
  python scripts/run_pipeline.py --step brazil --subset 60 --quick
  python scripts/run_pipeline.py --step brazil --res monthly --subset 60
  python scripts/run_pipeline.py --step extension --subset 60
  python scripts/run_pipeline.py --step report
  python scripts/run_pipeline.py --step official --design A --climate power --subset 300
  python scripts/run_pipeline.py --step official-report
  python scripts/official_postcheck.py results/official/designA_power_300
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:   # works without `pip install -e .`
    sys.path.insert(0, str(_SRC))

from dengue_recon import config as C  # noqa: E402

STEPS = ["data", "reproduce", "baselines", "brazil", "m2", "extension", "report", "official", "official-report", "all"]


def step_data(args):
    from dengue_recon.data import build_dataset, download, eda
    download.main(["infodengue", "--top-pop", "700"])
    download.main(["select", "--n", "300"])
    download.main(["power"])
    build_dataset.main()
    eda.main()
    from dengue_recon.data import oni
    oni.main()   # El Nino index (NOAA CPC, seconds): needed by --step official
    if args.with_chirps:   # CHIRPS rainfall for the robustness run (about 1-2 h for 300 municipalities)
        download.main(["weather", "--source", "chirps"])
        build_dataset.build_climate("chirps")
    if args.with_cox:
        from dengue_recon.data import validate_sources
        download.main(["cox"])
        validate_sources.main()


def step_reproduce(args):
    from dengue_recon.base_paper.fetch import fetch_lim_mask
    from dengue_recon.base_paper.reproduce import main
    if not C.LIM_MASK_PATH.exists():
        fetch_lim_mask()
    main(nuts=args.nuts, force=args.force)


def _years(args):
    return [int(y) for y in args.years.split(",")] if args.years else None


def step_baselines(args):
    from dengue_recon.experiments.brazil_loyo import main
    main(args.res, args.subset, models=["B4", "B4s"], tag=f"{args.res}_{args.subset}_noclimate", years=_years(args))


def step_brazil(args):
    from dengue_recon.experiments.brazil_loyo import main
    if args.quick:
        main("weekly", args.subset, main_only=True, tag=f"weekly_{args.subset}_quick", years=_years(args))
    else:
        main(args.res, args.subset, main_only=(args.subset == 300), tag=args.tag, years=_years(args))


def step_m2(args):
    from dengue_recon.models.xgb_softmax import main
    main(args.res, args.subset)


def step_extension(args):
    from dengue_recon.extensions.rain_lags import main
    if args.subset == 60:
        main(60, ("M1", "M1-R", "M1-RTH"), "weekly_60")
        if not args.quick:
            main(60, ("M1-R-shufTT-s0",), "weekly_60_shuf")
    else:
        main(300, ("M1-R", "M1-RTH"), "weekly_300")


def step_official(args):
    from dengue_recon.experiments.official import main
    main(design=args.design, climate=args.climate, subset=args.subset,
         exclude_years=[int(y) for y in args.exclude_year],
         models=args.models.split(",") if args.models else None,
         out_dir=args.out, test_years=_years(args))


def step_official_report(args):
    from dengue_recon.reporting.official_report import main
    main(base=args.out)


def step_report(args):
    from dengue_recon.reporting import rain_report, report
    report.main()
    try:
        rain_report.main()
    except FileNotFoundError as exc:
        print("[report] rain extension skipped:", exc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--subset", type=int, default=60, choices=[60, 300])
    p.add_argument("--res", default="weekly", choices=["weekly", "monthly"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--years", default=None, help="comma-separated LOYO years (smoke tests)")
    p.add_argument("--tag", default=None)
    p.add_argument("--nuts", action="store_true", help="reproduce: also run NUTS (slow)")
    p.add_argument("--force", action="store_true", help="reproduce: refit cached folds")
    p.add_argument("--design", default="A", choices=["A", "B", "C"], help="official: validation design")
    p.add_argument("--climate", default="power", choices=["power", "chirps"], help="official: weather table")
    p.add_argument("--exclude-year", type=int, action="append", default=[], help="official: drop a year (e.g. 2020)")
    p.add_argument("--models", default=None, help="official: comma-separated model subset")
    p.add_argument("--out", default=None, help="official: output folder (default RESULTS_DIR/official)")
    p.add_argument("--with-cox", action="store_true", help="data: also fetch Cox et al. and validate sources")
    p.add_argument("--with-chirps", action="store_true",
                   help="data: also fetch CHIRPS rainfall (robustness run; about 1-2 h)")
    args = p.parse_args(argv)
    C.ensure_dirs()
    t0 = time.time()
    if args.step == "all":
        step_data(args)
        step_reproduce(args)
        if args.quick:
            step_brazil(args)
        else:
            for res, sub in [("monthly", 60), ("weekly", 60), ("weekly", 300)]:
                args.res, args.subset = res, sub
                step_brazil(args)
            args.res, args.subset = "weekly", 60
            step_m2(args)
            step_extension(args)
            args.subset = 300
            step_extension(args)
        step_report(args)
    else:
        globals()[f"step_{args.step.replace('-', '_')}"](args)
    print(f"[run_pipeline] step '{args.step}' finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
