#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight CLI Entry for Cluster Factory

Supported flags align with emoles.build.cluster_factory.entry()
"""

import argparse
from emoles.build.cluster_factory import entry, DEFAULT_REF_DB_PATH, DEFAULT_FIXED_EPS

DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"


def _parse_states(raw: list[str]) -> list[tuple[int, int]]:
    """Parse '3:0 2:2 1:3' into [(3,0), (2,2), (1,3)]."""
    states = []
    for s in raw:
        solv, ani = s.split(":")
        states.append((int(solv), int(ani)))
    return states


def _parse_int_csv(text: str) -> tuple[int, ...]:
    """Parse '1,2,3' into (1, 2, 3)."""
    return tuple(int(x) for x in text.split(",") if x.strip())


def _resolve_source_arg(items: list[str] | None, default: str):
    """
    single .db/.json path  → string
    SMILES list            → list
    nothing                → [default]
    """
    if not items:
        return [default]
    if len(items) == 1 and (items[0].endswith(".db") or items[0].endswith(".json")):
        return items[0]
    return items


def main():
    p = argparse.ArgumentParser(
        description="Build Multi-Component Solvation Clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
# Fixed ε, pure‑anion + mixed states:
  python factory_api.py \\
      --solvents 'COCCOC:DME' \\
      --anions tdi_fsi.db \\
      --states 3:0 1:2 0:3 4:0 2:2 1:3 0:4 \\
      --solvent-mix-sizes 1 --anion-mix-sizes 1,2 \\
      --fixed-eps 7.2 --repeats 10 --workers 32

# Weighted ε from ref DB (omit --fixed-eps):
  python factory_api.py \\
      --solvents solvents.db \\
      --anions 'F[S](=O)(=O)[N-][S](=O)(=O)F:FSI' \\
      --states 3:1 4:1 \\
      --repeats 5
""",
    )

    # ── Input sources ────────────────────────────────────────────────────
    g_in = p.add_argument_group("Input Sources")
    g_in.add_argument(
        "--solvents", nargs="*", default=None,
        help=f"SMILES (:Name) or .db/.json path [default: {DEFAULT_DME_SMILES}]",
    )
    g_in.add_argument(
        "--anions", nargs="*", default=None,
        help=f"SMILES (:Name) or .db/.json path [default: {DEFAULT_FSI_SMILES}]",
    )
    g_in.add_argument(
        "--ref-db-path", default=DEFAULT_REF_DB_PATH,
        help="Solvent ref DB for InChI lookup & ε annotation "
             "(only used when --fixed-eps is not set)",
    )

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--out", default="out_mixture", dest="out_dir",
                   help="Output directory [default: out_mixture]")

    # ── Ion & States ─────────────────────────────────────────────────────
    g_state = p.add_argument_group("Ion & Coordination States")
    g_state.add_argument("--ion", default="Li",
                         help="Cation identifier [default: Li]")
    g_state.add_argument(
        "--states", nargs="+", default=None,
        help="Solvent:Anion count pairs, e.g.  3:1 4:0 0:4  [default: 3:1 4:1]",
    )
    # Legacy (hidden)
    g_state.add_argument("--target-totals", default="4,5", help=argparse.SUPPRESS)
    g_state.add_argument("--anion-counts", default="1", help=argparse.SUPPRESS)

    # ── Mixture logic ────────────────────────────────────────────────────
    g_mix = p.add_argument_group("Mixture Combinatorics")
    g_mix.add_argument(
        "--solvent-mix-sizes", default="1",
        help="Distinct solvent species per cluster, csv [default: 1]",
    )
    g_mix.add_argument(
        "--anion-mix-sizes", default="1",
        help="Distinct anion species per cluster, csv [default: 1]",
    )
    g_mix.add_argument("--repeats", type=int, default=1,
                       help="Structures per unique combination [default: 1]")

    # ── Dielectric constant ──────────────────────────────────────────────
    g_eps = p.add_argument_group("Dielectric Constant (ε)")
    g_eps.add_argument(
        "--fixed-eps", type=float, default=DEFAULT_FIXED_EPS,
        help=f"Use a fixed ε for ALL clusters [default: {DEFAULT_FIXED_EPS}]. "
             f"Set to 0 or 'none' to use weighted ref‑DB lookup instead.",
    )

    # ── Runtime ──────────────────────────────────────────────────────────
    g_run = p.add_argument_group("Runtime")
    g_run.add_argument("--workers", type=int, default=32, dest="n_jobs",
                       help="Parallel build workers [default: 32]")
    g_run.add_argument("--device", default="cuda",
                       help="Torch device for UMA [default: cuda]")
    g_run.add_argument("--no-uma", action="store_false", dest="use_uma",
                       help="Disable UMA pre/post-optimization")
    g_run.add_argument("--no-progress", action="store_false", dest="show_progress",
                       help="Disable progress bars")
    g_run.add_argument("--verbose", action="store_true",
                       help="Verbose UMA output")
    p.set_defaults(use_uma=True, show_progress=True, verbose=False)

    args = p.parse_args()

    # ── Resolve inputs ───────────────────────────────────────────────────
    solv_arg = _resolve_source_arg(args.solvents, DEFAULT_DME_SMILES)
    anion_arg = _resolve_source_arg(args.anions, DEFAULT_FSI_SMILES)

    # ── Resolve states ───────────────────────────────────────────────────
    if args.states:
        parsed_states = _parse_states(args.states)
    else:
        t_totals = _parse_int_csv(args.target_totals)
        a_counts = _parse_int_csv(args.anion_counts)
        parsed_states = [
            (t - a, a) for t in t_totals for a in a_counts if t >= a
        ]

    # ── Resolve fixed_eps ────────────────────────────────────────────────
    #    0 or negative → None → weighted mode
    resolved_eps = args.fixed_eps if (args.fixed_eps and args.fixed_eps > 0) else None

    # ── Call factory entry ───────────────────────────────────────────────
    entry(
        solvents=solv_arg,
        anions=anion_arg,
        ref_db_path=args.ref_db_path,
        out_dir=args.out_dir,
        ion=args.ion,
        states=parsed_states,
        solvent_mix_sizes=_parse_int_csv(args.solvent_mix_sizes),
        anion_mix_sizes=_parse_int_csv(args.anion_mix_sizes),
        repeats=args.repeats,
        fixed_eps=resolved_eps,
        use_uma=args.use_uma,
        device=args.device,
        verbose=args.verbose,
        show_progress=args.show_progress,
        n_jobs=args.n_jobs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default test command:

# nohup bash -c "
# python factory_api.py \
#   --solvents sol.db \
#   --anions '[F][B-]1([F])OC(=O)C(=O)O1:DFOB' '[F][B-]1([F])OC(C(F)(F)F)(C(F)(F)F)C(C(F)(F)F)(C(F)(F)F)O1:DFTFB' \
#   --target-totals 4 \
#   --anion-counts 1 \
#   --repeats 7 \
#   --anion-mix-sizes 1 \
#   --solvent-mix-sizes 2 \
#   --fixed-eps 28.29 \
#   --device cuda && \
# python dm_infer_pipeline.py
# " > run_tables.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()