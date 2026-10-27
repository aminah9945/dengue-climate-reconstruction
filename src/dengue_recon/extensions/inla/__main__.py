"""CLI for the EXPLORATORY INLA extension.

python -m dengue_recon.extensions.inla run --subset 300 [--variants base,clim20,clim15]
       [--samples 500] [--threads 4] [--years 2012,2013] [--max-folds 2]
python -m dengue_recon.extensions.inla report [--subset 300] [--official-dir DIR]
"""
from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m dengue_recon.extensions.inla")
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--subset", type=int, default=300, choices=[60, 300])
    p.add_argument("--climate", default="power", choices=["power", "chirps"])
    p.add_argument("--variants", default="base,clim20,clim15")
    p.add_argument("--samples", type=int, default=None, help="posterior draws per fit (default 500)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--years", default=None, help="restrict design-A test years (smoke tests)")
    p.add_argument("--max-folds", type=int, default=None, help="stop after this many new folds (time-boxed runs)")
    p.add_argument("--out", default=None, help="output folder (default RESULTS_DIR/extensions/inla)")
    p.add_argument("--work", default=None, help="folder for the per-fold R data and draws")
    p.add_argument("--official-dir", default=None, help="report: folder of the official runs (metrics_detail)")
    a = p.parse_args(argv)
    if a.step == "run":
        from .driver import N_SAMPLES, main as run
        years = [int(y) for y in a.years.split(",")] if a.years else None
        run(subset=a.subset, climate=a.climate, variants=tuple(a.variants.split(",")),
            n_samples=N_SAMPLES if a.samples is None else a.samples, n_threads=a.threads, test_years=years,
            out_dir=a.out, work_dir=a.work, max_folds=a.max_folds)
    else:
        from .report import main as report
        report(subset=a.subset, climate=a.climate, out_dir=a.out, official_dir=a.official_dir)


if __name__ == "__main__":
    main()
