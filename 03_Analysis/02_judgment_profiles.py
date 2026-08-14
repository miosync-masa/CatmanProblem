from __future__ import annotations

import pandas as pd

from analysis_utils import (
    JUDGMENT_ITEMS,
    READY_DIR,
    RESULTS_DIR,
    bh_fdr,
    ensure_dirs,
    linear_contrast_effect,
    paired_condition_effect,
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
    data = pd.read_csv(READY_DIR / "judgment_long.csv")

    effect_rows = []
    profile_rows = []
    for index, (condition, contrast_label) in enumerate(CONTRASTS.items()):
        for item in JUDGMENT_ITEMS:
            row = paired_condition_effect(data, condition, "C01", item)
            row["contrast"] = contrast_label
            effect_rows.append(row)
        differences = profile_difference_matrix(data, condition, "C01", JUDGMENT_ITEMS)
        test = permutation_profile_test(
            differences, permutations=100_000, seed=20260814 + index
        )
        test.update({"condition_id": condition, "contrast": contrast_label})
        profile_rows.append(test)

    effects = pd.DataFrame(effect_rows)
    effects["p_fdr_all_30"] = bh_fdr(effects["p_ttest"])
    write_csv(effects, RESULTS_DIR / "judgment_effects.csv")

    profiles = pd.DataFrame(profile_rows)
    profiles["p_fdr_six_profiles"] = bh_fdr(profiles["p_permutation"])
    profiles["interpretation"] = profiles["p_fdr_six_profiles"].map(
        lambda p: "query-dependent profile" if p < 0.05 else "no detected profile heterogeneity"
    )
    write_csv(profiles, RESULTS_DIR / "judgment_profile_tests.csv")

    interaction_specs = {
        "Body × transition": {"C07": 1.0, "C02": -1.0, "C06": -1.0, "C01": 1.0},
        "Memory × transition": {"C08": 1.0, "C03": -1.0, "C06": -1.0, "C01": 1.0},
    }
    interaction_rows = []
    for label, coefficients in interaction_specs.items():
        for item in JUDGMENT_ITEMS:
            interaction_rows.append(linear_contrast_effect(data, coefficients, item, label))
    interactions = pd.DataFrame(interaction_rows)
    interactions["p_fdr_all_10"] = bh_fdr(interactions["p_ttest"])
    write_csv(interactions, RESULTS_DIR / "judgment_interactions_exploratory.csv")


if __name__ == "__main__":
    main()

