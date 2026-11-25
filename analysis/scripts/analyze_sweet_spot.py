"""
LEVEL 1 — SWEET-SPOT STATISTICAL ANALYSIS

No QPU usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

INPUT_FILE = (
    RESULTS_DIR
    / "sweet_spot_level1_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "sweet_spot_level1_analysis.json"
)

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]


def main():

    print("=" * 72)
    print("SWEET-SPOT LEVEL 1 ANALYSIS")
    print("=" * 72)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}\n\n"
            "Run run_sweet_spot_level1.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    reports = []

    for backend in BACKENDS:

        sub = df[
            df["backend"]
            == backend
        ]

        sweet = sub[
            sub["group"]
            == "sweet_spot"
        ]["fidelity"].to_numpy(
            dtype=float
        )

        random = sub[
            sub["group"]
            == "random_control"
        ]["fidelity"].to_numpy(
            dtype=float
        )

        if len(sweet) != 1:

            raise RuntimeError(
                f"{backend}: expected exactly "
                "one Sweet-Spot result."
            )

        if len(random) < 2:

            raise RuntimeError(
                f"{backend}: not enough random controls."
            )

        sweet_value = float(
            sweet[0]
        )

        random_mean = float(
            random.mean()
        )

        random_median = float(
            np.median(random)
        )

        random_std = float(
            random.std(
                ddof=1
            )
        )

        random_min = float(
            random.min()
        )

        random_max = float(
            random.max()
        )

        difference = (
            sweet_value
            - random_mean
        )

        relative = (
            difference
            / random_mean
            * 100.0
        )

        below = int(
            np.sum(
                random < sweet_value
            )
        )

        equal = int(
            np.sum(
                random == sweet_value
            )
        )

        percentile = (
            (
                below
                + 0.5 * equal
            )
            / len(random)
        )

        standardized = None

        if random_std > 0:

            standardized = (
                difference
                / random_std
            )

        if difference > 0:

            direction = (
                "Sweet-Spot exceeded "
                "the random-control mean."
            )

        elif difference < 0:

            direction = (
                "Sweet-Spot was below "
                "the random-control mean."
            )

        else:

            direction = (
                "Sweet-Spot matched "
                "the random-control mean."
            )

        report = {
            "backend":
                backend,
            "sweet_spot_fidelity":
                sweet_value,
            "random_control_count":
                len(random),
            "random_mean_fidelity":
                random_mean,
            "random_median_fidelity":
                random_median,
            "random_std":
                random_std,
            "random_min":
                random_min,
            "random_max":
                random_max,
            "absolute_difference":
                difference,
            "relative_improvement_percent":
                relative,
            "sweet_spot_empirical_percentile":
                percentile,
            "standardized_difference":
                standardized,
            "direction":
                direction,
        }

        reports.append(
            report
        )

        print()
        print(
            f"{backend}"
        )
        print(
            f"  Sweet-Spot : {sweet_value:.4f}"
        )
        print(
            f"  Random     : {random_mean:.4f}"
        )
        print(
            f"  Difference : {difference:+.4f}"
        )
        print(
            f"  Relative   : {relative:+.2f}%"
        )
        print(
            f"  Percentile : {percentile:.3f}"
        )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    global_sweet = float(
        np.mean(
            [
                x["sweet_spot_fidelity"]
                for x in reports
            ]
        )
    )

    global_random = float(
        np.mean(
            [
                x["random_mean_fidelity"]
                for x in reports
            ]
        )
    )

    global_difference = (
        global_sweet
        - global_random
    )

    global_relative = (
        global_difference
        / global_random
        * 100.0
    )

    output = {
        "experiment":
            "sweet_spot_vs_random",
        "level":
            1,
        "interpretation":
            (
                "Local calibration-driven Aer simulation. "
                "Random controls are topology-valid. "
                "The comparison is descriptive and does not "
                "constitute a formal hardware significance test."
            ),
        "backend_results":
            reports,
        "global_summary":
            {
                "mean_sweet_spot_fidelity":
                    global_sweet,
                "mean_random_fidelity":
                    global_random,
                "mean_difference":
                    global_difference,
                "relative_improvement_percent":
                    global_relative,
            },
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)

    print(
        f"Global Sweet-Spot : "
        f"{global_sweet:.4f}"
    )

    print(
        f"Global Random     : "
        f"{global_random:.4f}"
    )

    print(
        f"Global difference : "
        f"{global_difference:+.4f}"
    )

    print(
        f"Relative change   : "
        f"{global_relative:+.2f}%"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()