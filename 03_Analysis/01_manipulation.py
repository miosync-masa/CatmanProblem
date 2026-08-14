from __future__ import annotations

import pandas as pd

from analysis_utils import (
    MANIPULATION_ITEMS,
    READY_DIR,
    RESULTS_DIR,
    bh_fdr,
    ensure_dirs,
    paired_condition_effect,
    write_csv,
)


CONTRASTS = {
    "C02": ("Body", "pc_sub"),
    "C03": ("Memory", "pc_info"),
    "C04": ("Function", "pc_func"),
    "C05": ("Self-ID uncertain", "pc_self"),
    "C06": ("Transition", "pc_tran"),
    "A09": ("Self-denial anchor", "pc_self"),
}


def main() -> None:
    ensure_dirs()
    data = pd.read_csv(READY_DIR / "manipulation_long.csv")

    rows = []
    for condition, (contrast_label, target_item) in CONTRASTS.items():
        for item in MANIPULATION_ITEMS:
            row = paired_condition_effect(data, condition, "C01", item)
            row.update(
                {
                    "contrast": contrast_label,
                    "target_item": target_item,
                    "is_target": item == target_item,
                }
            )
            rows.append(row)
    selectivity = pd.DataFrame(rows)
    selectivity["p_fdr_all_30"] = bh_fdr(selectivity["p_ttest"])
    write_csv(selectivity, RESULTS_DIR / "manipulation_selectivity.csv")

    target_rows = []
    for contrast, group in selectivity.groupby("contrast", sort=False):
        target = group[group["is_target"]].iloc[0].to_dict()
        spillover = group[~group["is_target"]].copy()
        largest_index = spillover["mean_difference"].abs().idxmax()
        largest = spillover.loc[largest_index]
        target.update(
            {
                "largest_spillover_item": largest["item"],
                "largest_spillover_mean_difference": largest["mean_difference"],
                "selectivity_gap_absolute": abs(target["mean_difference"]) - abs(largest["mean_difference"]),
                "selectivity_ratio_absolute": abs(target["mean_difference"]) / max(abs(largest["mean_difference"]), 1e-12),
            }
        )
        target_rows.append(target)
    write_csv(pd.DataFrame(target_rows), RESULTS_DIR / "manipulation_target_effects.csv")

    stress_pairs = [
        ("C06", "C01", "Human / retained memory"),
        ("C07", "C02", "Cat body"),
        ("C08", "C03", "Memory lost"),
    ]
    stress_rows = []
    for condition, baseline, context in stress_pairs:
        for item in MANIPULATION_ITEMS:
            row = paired_condition_effect(data, condition, baseline, item)
            row["context"] = context
            stress_rows.append(row)
    stress = pd.DataFrame(stress_rows)
    stress["p_fdr_all_15"] = bh_fdr(stress["p_ttest"])
    write_csv(stress, RESULTS_DIR / "transition_stress_test.csv")

    extras = (
        data[data["item"].isin(["k_copy", "k_term", "k_coex"])]
        .groupby(["condition_id", "condition_label", "item", "item_label"], as_index=False)
        .agg(n=("value", "size"), mean=("value", "mean"), sd=("value", "std"), median=("value", "median"))
    )
    write_csv(extras, RESULTS_DIR / "manipulation_auxiliary_descriptives.csv")


if __name__ == "__main__":
    main()

