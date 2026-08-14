from __future__ import annotations

import pandas as pd

from analysis_utils import CONDITION_LABELS, ITEM_LABELS, RAW_DIR, READY_DIR, RESULTS_DIR, ensure_dirs, write_csv


def main() -> None:
    ensure_dirs()
    responses = pd.read_csv(RAW_DIR / "catman_responses_deidentified.csv")
    sessions = pd.read_csv(RAW_DIR / "catman_sessions_deidentified.csv")

    required = {
        "study",
        "participant_id",
        "arm",
        "condition_id",
        "item",
        "value",
        "attention_pass",
    }
    missing = required - set(responses.columns)
    if missing:
        raise ValueError(f"Missing required response columns: {sorted(missing)}")
    if not responses["value"].between(1, 7).all():
        raise ValueError("All response values must lie on the 1–7 scale.")

    duplicate_keys = ["participant_id", "arm", "condition_id", "item"]
    duplicate_count = int(responses.duplicated(duplicate_keys).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate response keys.")

    responses["condition_label"] = responses["condition_id"].map(CONDITION_LABELS)
    responses["item_label"] = responses["item"].map(ITEM_LABELS)
    included = responses[responses["attention_pass"].eq(1)].copy()
    judgment = included[included["arm"].eq("judgment")].copy()
    manipulation = included[included["arm"].eq("manipulation")].copy()
    sessions_included = sessions[sessions["attention_pass"].eq(1)].copy()

    write_csv(included, READY_DIR / "catman_analysis_ready_long.csv")
    write_csv(judgment, READY_DIR / "judgment_long.csv")
    write_csv(manipulation, READY_DIR / "manipulation_long.csv")
    write_csv(sessions_included, READY_DIR / "sessions_included.csv")

    sample_rows = []
    for arm in ["judgment", "manipulation"]:
        arm_sessions = sessions[sessions["arm"].eq(arm)]
        n_total = int(arm_sessions["participant_id"].nunique())
        n_pass = int(arm_sessions.loc[arm_sessions["attention_pass"].eq(1), "participant_id"].nunique())
        sample_rows.append(
            {
                "arm": arm,
                "n_total": n_total,
                "n_attention_pass": n_pass,
                "n_attention_excluded": n_total - n_pass,
            }
        )
    write_csv(pd.DataFrame(sample_rows), RESULTS_DIR / "sample_flow.csv")

    data_checks = pd.DataFrame(
        [
            {"check": "response_rows_raw", "value": len(responses)},
            {"check": "unique_participants_raw", "value": responses["participant_id"].nunique()},
            {"check": "arm_sessions_raw", "value": len(sessions)},
            {"check": "participants_in_both_arms", "value": sessions.groupby("participant_id")["arm"].nunique().eq(2).sum()},
            {"check": "duplicate_response_keys", "value": duplicate_count},
            {"check": "missing_response_values", "value": int(responses[sorted(required)].isna().sum().sum())},
            {"check": "out_of_range_values", "value": int((~responses["value"].between(1, 7)).sum())},
            {"check": "conditions", "value": responses["condition_id"].nunique()},
            {"check": "judgment_included_n", "value": judgment["participant_id"].nunique()},
            {"check": "manipulation_included_n", "value": manipulation["participant_id"].nunique()},
        ]
    )
    write_csv(data_checks, RESULTS_DIR / "data_quality_checks.csv")


if __name__ == "__main__":
    main()
