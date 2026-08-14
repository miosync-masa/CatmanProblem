from __future__ import annotations

import pandas as pd

from analysis_utils import (
    JUDGMENT_ITEMS,
    RAW_DIR,
    READY_DIR,
    RESULTS_DIR,
    bh_fdr,
    ensure_dirs,
    paired_condition_effect,
    pearson_summary,
    permutation_profile_test,
    profile_difference_matrix,
    write_csv,
)


CONTRASTS = {
    "C02": "Body",
    "C03": "Memory",
    "C04": "Function",
    "C05": "Self-ID uncertain",
    "C06": "Transition",
    "A09": "Self-denial anchor",
}


def main() -> None:
    ensure_dirs()
    included = pd.read_csv(READY_DIR / "judgment_long.csv")
    raw = pd.read_csv(RAW_DIR / "catman_responses_deidentified.csv")
    all_judgment = raw[raw["arm"].eq("judgment")].copy()

    # Participant-level mean tendencies across the nine non-parallel conditions.
    means = (
        included[
            included["item"].isin(JUDGMENT_ITEMS)
            & ~included["condition_id"].eq("A10")
        ]
        .groupby(["participant_id", "item"])["value"]
        .mean()
        .unstack("item")
    )
    corr_rows = []
    for i, item_x in enumerate(JUDGMENT_ITEMS):
        for item_y in JUDGMENT_ITEMS[i + 1 :]:
            row = pearson_summary(means[item_x], means[item_y])
            row.update({"item_x": item_x, "item_y": item_y})
            corr_rows.append(row)
    correlations = pd.DataFrame(corr_rows)
    correlations["p_fdr_10"] = bh_fdr(correlations["p"])
    write_csv(correlations, RESULTS_DIR / "participant_mean_judgment_correlations.csv")

    sensitivity_rows = []
    for sample_label, data in [("attention_pass", included), ("all_judgment", all_judgment)]:
        for index, (condition, contrast_label) in enumerate(CONTRASTS.items()):
            differences = profile_difference_matrix(data, condition, "C01", JUDGMENT_ITEMS)
            test = permutation_profile_test(
                differences,
                permutations=100_000,
                seed=20260900 + index + (100 if sample_label == "all_judgment" else 0),
            )
            test.update(
                {
                    "sample": sample_label,
                    "condition_id": condition,
                    "contrast": contrast_label,
                }
            )
            sensitivity_rows.append(test)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["p_fdr_within_sample"] = sensitivity.groupby("sample")["p_permutation"].transform(
        lambda x: bh_fdr(x)
    )
    write_csv(sensitivity, RESULTS_DIR / "judgment_profile_sensitivity.csv")

    effect_sensitivity = []
    for sample_label, data in [("attention_pass", included), ("all_judgment", all_judgment)]:
        for condition, contrast_label in CONTRASTS.items():
            for item in JUDGMENT_ITEMS:
                row = paired_condition_effect(data, condition, "C01", item)
                row.update({"sample": sample_label, "contrast": contrast_label})
                effect_sensitivity.append(row)
    write_csv(pd.DataFrame(effect_sensitivity), RESULTS_DIR / "judgment_effect_sensitivity.csv")


if __name__ == "__main__":
    main()

