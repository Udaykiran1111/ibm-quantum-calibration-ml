import json
from pathlib import Path

import numpy as np


RANDOM_FILE = Path("results/random_baseline_fez.json")
MODEL_FILE = Path("results/level2_model_a.json")
OUTPUT_FILE = Path("results/level2_statistical_analysis.json")

BOOTSTRAP_ITERATIONS = 100_000
RANDOM_SEED = 20260823


def load_results():
    if not RANDOM_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {RANDOM_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {MODEL_FILE}"
        )

    with open(RANDOM_FILE, "r", encoding="utf-8") as f:
        random_data = json.load(f)

    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    random_results = random_data["results"]

    random_fidelities = np.array(
        [x["fidelity"] for x in random_results],
        dtype=float,
    )

    # model_result = model_data["result"]

    model_fidelity = float(
    model_data["fidelity"]
)

    model_depth = int(
    model_data["transpiled_depth"]
)

    return (
        random_results,
        random_fidelities,
        model_fidelity,
        model_depth,
    )


def bootstrap_mean_ci(values):
    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap_means = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    n = len(values)

    for i in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = sample.mean()

    low, high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(low), float(high)


def main():

    print("=" * 70)
    print("LEVEL 2 — STATISTICAL ANALYSIS")
    print("=" * 70)

    (
        random_results,
        random_fidelities,
        model_fidelity,
        model_depth,
    ) = load_results()

    random_mean = float(
        random_fidelities.mean()
    )

    random_median = float(
        np.median(random_fidelities)
    )

    random_std = float(
        random_fidelities.std(ddof=1)
    )

    random_min = float(
        random_fidelities.min()
    )

    random_max = float(
        random_fidelities.max()
    )

    improvement = (
        model_fidelity - random_mean
    )

    relative_improvement = (
        improvement / random_mean
    ) * 100

    # ------------------------------------------------------------
    # Bootstrap confidence interval for the RANDOM mean
    # ------------------------------------------------------------

    ci_low, ci_high = bootstrap_mean_ci(
        random_fidelities
    )

    # ------------------------------------------------------------
    # Empirical position of Model A within random distribution
    # ------------------------------------------------------------

    below_model = int(
        np.sum(
            random_fidelities < model_fidelity
        )
    )

    equal_model = int(
        np.sum(
            random_fidelities == model_fidelity
        )
    )

    empirical_percentile = (
        (below_model + 0.5 * equal_model)
        / len(random_fidelities)
    )

    # ------------------------------------------------------------
    # Depth-matched comparison
    # ------------------------------------------------------------

    depth_matched = np.array(
        [
            x["fidelity"]
            for x in random_results
            if x.get("transpiled_depth")
            == model_depth
        ],
        dtype=float,
    )

    if len(depth_matched) > 1:

        depth_matched_mean = float(
            depth_matched.mean()
        )

        depth_matched_std = float(
            depth_matched.std(ddof=1)
        )

        depth_matched_difference = (
            model_fidelity
            - depth_matched_mean
        )

    else:

        depth_matched_mean = None
        depth_matched_std = None
        depth_matched_difference = None

    # ------------------------------------------------------------
    # Effect size against random baseline
    #
    # Treating the random sample as the reference distribution.
    # This is descriptive, NOT a formal hypothesis test because
    # there is only one Model A hardware observation.
    # ------------------------------------------------------------

    if random_std > 0:

        standardized_difference = (
            model_fidelity - random_mean
        ) / random_std

    else:

        standardized_difference = None

    # ------------------------------------------------------------
    # Conservative interpretation
    # ------------------------------------------------------------

    if improvement > 0:

        pilot_direction = (
            "Model A exceeded the random-control mean."
        )

    elif improvement < 0:

        pilot_direction = (
            "Model A was below the random-control mean."
        )

    else:

        pilot_direction = (
            "Model A matched the random-control mean."
        )

    interpretation = (
        "This is a pilot hardware comparison. "
        "There is one Model A hardware observation and "
        f"{len(random_fidelities)} random controls. "
        "Therefore the result should not be described as "
        "statistically significant evidence of model superiority. "
        "It supports a preliminary directional comparison."
    )

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    report = {

        "experiment": {
            "backend": "ibm_fez",
            "circuit": "3-qubit GHZ",
            "shots": 256,
            "model": "model_a",
            "model_qubits": [21, 22, 23],
            "model_transpiled_depth": model_depth,
            "random_control_count": len(
                random_fidelities
            ),
        },

        "model_a": {
            "fidelity": model_fidelity,
        },

        "random_controls": {
            "mean": random_mean,
            "median": random_median,
            "std": random_std,
            "min": random_min,
            "max": random_max,
            "bootstrap_mean_95ci": [
                ci_low,
                ci_high,
            ],
        },

        "comparison": {
            "absolute_improvement": improvement,
            "relative_improvement_percent":
                relative_improvement,
            "model_empirical_percentile":
                empirical_percentile,
            "random_controls_below_model":
                below_model,
            "random_controls_equal_model":
                equal_model,
            "standardized_difference":
                standardized_difference,
        },

        "depth_matched_analysis": {
            "model_depth": model_depth,
            "number_of_matching_random_controls":
                len(depth_matched),
            "random_mean":
                depth_matched_mean,
            "random_std":
                depth_matched_std,
            "model_minus_random_mean":
                depth_matched_difference,
        },

        "interpretation": {
            "pilot_direction":
                pilot_direction,
            "conclusion":
                interpretation,
        },

        "limitations": [
            "Only one Model A hardware execution was performed.",
            "Random controls were executed once each.",
            "The experiment used one backend.",
            "The experiment used one circuit family.",
            "The current model predicts individual-qubit viability rather than explicit two-qubit gate quality.",
            "The result is a pilot comparison rather than a statistically powered validation."
        ],
    }

    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------

    print()
    print("MODEL A")
    print("-" * 70)
    print(
        f"Fidelity                 : "
        f"{model_fidelity:.4f}"
    )

    print()
    print("RANDOM BASELINE")
    print("-" * 70)
    print(
        f"Trials                   : "
        f"{len(random_fidelities)}"
    )
    print(
        f"Mean                     : "
        f"{random_mean:.4f}"
    )
    print(
        f"Median                   : "
        f"{random_median:.4f}"
    )
    print(
        f"Std                      : "
        f"{random_std:.4f}"
    )
    print(
        f"Min                      : "
        f"{random_min:.4f}"
    )
    print(
        f"Max                      : "
        f"{random_max:.4f}"
    )

    print()
    print("95% BOOTSTRAP CI FOR RANDOM MEAN")
    print("-" * 70)
    print(
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

    print()
    print("MODEL VS RANDOM")
    print("-" * 70)
    print(
        f"Absolute improvement      : "
        f"{improvement:+.4f}"
    )
    print(
        f"Relative improvement     : "
        f"{relative_improvement:+.2f}%"
    )
    print(
        f"Model empirical percentile: "
        f"{empirical_percentile:.3f}"
    )
    print(
        f"Random controls below     : "
        f"{below_model}/{len(random_fidelities)}"
    )

    print()
    print("DEPTH-MATCHED CONTROL")
    print("-" * 70)
    print(
        f"Model depth              : "
        f"{model_depth}"
    )
    print(
        f"Matching random controls : "
        f"{len(depth_matched)}"
    )

    if depth_matched_mean is not None:

        print(
            f"Random mean              : "
            f"{depth_matched_mean:.4f}"
        )

        print(
            f"Random std               : "
            f"{depth_matched_std:.4f}"
        )

        print(
            f"Model - random           : "
            f"{depth_matched_difference:+.4f}"
        )

    print()
    print("INTERPRETATION")
    print("-" * 70)
    print(pilot_direction)
    print()
    print(interpretation)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()