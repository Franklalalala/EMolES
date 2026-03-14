#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight API Entry for Cluster Factory
"""

import argparse
from emoles.build.cluster_factory import entry

DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"


def main():
    parser = argparse.ArgumentParser(description="Build Multi-Component Solvation Clusters")

    # Inputs
    parser.add_argument('--solvents', nargs='*', default=None, help="SMILES or DB path for solvents")
    parser.add_argument('--anions', nargs='*', default=None, help="SMILES or DB path for anions")
    parser.add_argument('--out', default='out_mixture', help="Output directory")

    # Configuration
    parser.add_argument('--ion', default='Li', help="Ion identifier")

    # [NEW] States definition
    parser.add_argument('--states', nargs='+', default=None,
                        help="List of 'SolventCount:AnionCount' pairs, e.g., '3:0 2:2 1:3 0:4'")

    # Old legacy params (kept for backward compatibility)
    parser.add_argument('--target-totals', default='4,5', help="Legacy: Allowed total coordination numbers")
    parser.add_argument('--anion-counts', default='1', help="Legacy: Allowed anion counts")

    # Mixture Logic
    parser.add_argument('--mix-n', default='1', help="List of solvent mix sizes (e.g. '1,2')")
    parser.add_argument('--anion-mix-n', default='1', help="List of anion mix sizes (e.g. '1,2')")
    parser.add_argument('--num-mixtures', type=int, default=10, help="Max random solvent combinations")
    parser.add_argument('--repeats', type=int, default=1, help="Number of structures to generate per valid combination")

    # Parallelism
    parser.add_argument('--workers', type=int, default=32, dest='n_jobs', help="Number of parallel build processes")

    # Flags
    parser.add_argument('--no-uma', action='store_false', dest='use_uma', help="Disable UMA pre/post-optimization")
    parser.set_defaults(use_uma=True)

    parser.add_argument('--device', default='cuda', help="Device for optimization")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Parse and fallback defaults
    solv_arg = args.solvents if args.solvents else [DEFAULT_DME_SMILES]
    if len(solv_arg) == 1 and (solv_arg[0].endswith('.db') or solv_arg[0].endswith('.json')):
        solv_arg = solv_arg[0]

    anion_arg = args.anions if args.anions else [DEFAULT_FSI_SMILES]
    if len(anion_arg) == 1 and (anion_arg[0].endswith('.db') or anion_arg[0].endswith('.json')):
        anion_arg = anion_arg[0]

    # Parse Macro States (Solvent : Anion)
    parsed_states = []
    if args.states:
        for s in args.states:
            solv, ani = s.split(':')
            parsed_states.append((int(solv), int(ani)))
    else:
        # Fallback to legacy Cartesian product
        t_totals = [int(x) for x in args.target_totals.split(',') if x.strip()]
        a_counts = [int(x) for x in args.anion_counts.split(',') if x.strip()]
        for t in t_totals:
            for a in a_counts:
                if t >= a:  # Total >= anion count means solvent = Total - anion
                    parsed_states.append((t - a, a))

    m_n_list = tuple(int(x) for x in args.mix_n.split(',') if x.strip())
    a_m_n_list = tuple(int(x) for x in args.anion_mix_n.split(',') if x.strip())

    entry(
        solvents=solv_arg,
        anions=anion_arg,
        out_dir=args.out,
        ion=args.ion,
        states=parsed_states,
        mix_n_list=m_n_list,
        anion_mix_n_list=a_m_n_list,
        num_mixtures=args.num_mixtures,
        repeats=args.repeats,
        use_uma=args.use_uma,
        device=args.device,
        verbose=args.verbose,
        n_jobs=args.n_jobs
    )


# nohup bash -c "
# python factory_api.py \
#   --solvents 'COCCOC:DME' \
#   --anions tdi_fsi.db \
#   --states 3:0 2:2 1:3 0:4 \
#   --mix-n 1 \
#   --anion-mix-n 1,2 \
#   --repeats 1 \
#   --workers 16 \
#   --device cuda && \
#  python dm_infer_pipeline.py
# " > run_step_diagram.log 2>&1 &

if __name__ == "__main__":
    main()