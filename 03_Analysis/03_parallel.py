from __future__ import annotations

import pandas as pd

from analysis_utils import (
    ITEM_LABELS,
    READY_DIR,
    RESULTS_DIR,
    ensure_dirs,
    paired_condition_effect,
    paired_summary,
    pearson_summary,
    wilson_ci,
    write_csv,
)


def main() -> None:
    ensure_dirs()
    data = pd.read_csv(READY_DIR / "judgment_long.csv")
    parallel = data[data["condition_id"].eq("A10")].pivot(
        index="participant_id", columns="item", values="value"
    )
    required = ["j_ab", "j_ac", "j_bc", "j_excl"]
    parallel = parallel.dropna(subset=required).copy()

    # Predeclared descriptive thresholds for the exploratory anchor.
    parallel["ab_high"] = parallel["j_ab"].ge(5)
    parallel["ac_high"] = parallel["j_ac"].ge(5)
    parallel["bc_high"] = parallel["j_bc"].ge(5)
    parallel["both_predecessor_links_high"] = parallel["ab_high"] & parallel["ac_high"]
    parallel["transitivity_like_discordance"] = (
        parallel["ab_high"] & parallel["ac_high"] & ~parallel["bc_high"]
    )
    parallel["pattern"] = "other"
    parallel.loc[parallel["both_predecessor_links_high"], "pattern"] = "AB and AC high"
    parallel.loc[parallel["transitivity_like_discordance"], "pattern"] = "AB and AC high; BC not high"
    write_csv(parallel.reset_index(), RESULTS_DIR / "parallel_patterns_deidentified.csv")

    summary_rows = []
    n = len(parallel)
    for name, mask, definition in [
        (
            "both_predecessor_links_high",
            parallel["both_predecessor_links_high"],
            "j_ab >= 5 and j_ac >= 5",
        ),
        (
            "transitivity_like_discordance",
            parallel["transitivity_like_discordance"],
            "j_ab >= 5, j_ac >= 5, and j_bc <= 4",
        ),
    ]:
        count = int(mask.sum())
        low, high = wilson_ci(count, n)
        summary_rows.append(
            {
                "classification": name,
                "definition": definition,
                "count": count,
                "n": n,
                "proportion": count / n,
                "wilson_ci95_low": low,
                "wilson_ci95_high": high,
            }
        )
    write_csv(pd.DataFrame(summary_rows), RESULTS_DIR / "parallel_classification.csv")

    ratings = []
    for item in required:
        desc = paired_summary(parallel[item])
        ratings.append(
            {
                "item": item,
                "item_label": ITEM_LABELS[item],
                "n": desc["n"],
                "mean": desc["mean_difference"],
                "sd": desc["sd_difference"],
                "ci95_low": desc["ci95_low"],
                "ci95_high": desc["ci95_high"],
                "median": float(parallel[item].median()),
            }
        )
    write_csv(pd.DataFrame(ratings), RESULTS_DIR / "parallel_rating_descriptives.csv")

    deny = data[data["condition_id"].isin(["C01", "A09"])].pivot(
        index="participant_id", columns=["condition_id", "item"], values="value"
    )
    veto_rows = []
    for item in ["v_epistemic", "v_normative"]:
        values = deny[("A09", item)].dropna()
        desc = paired_summary(values)
        veto_rows.append(
            {
                "item": item,
                "item_label": ITEM_LABELS[item],
                "n": desc["n"],
                "mean": desc["mean_difference"],
                "sd": desc["sd_difference"],
                "ci95_low": desc["ci95_low"],
                "ci95_high": desc["ci95_high"],
                "median": float(values.median()),
            }
        )
    write_csv(pd.DataFrame(veto_rows), RESULTS_DIR / "deny_veto_descriptives.csv")

    identity_change = deny[("A09", "j_identity")] - deny[("C01", "j_identity")]
    treatment_change = deny[("A09", "j_treatment")] - deny[("C01", "j_treatment")]
    associations = []
    for label, x, y in [
        ("epistemic veto vs identity change", deny[("A09", "v_epistemic")], identity_change),
        ("normative veto vs treatment change", deny[("A09", "v_normative")], treatment_change),
    ]:
        row = pearson_summary(x, y)
        row["association"] = label
        associations.append(row)
    write_csv(pd.DataFrame(associations), RESULTS_DIR / "deny_veto_associations_exploratory.csv")


if __name__ == "__main__":
    main()

