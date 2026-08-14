from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis_utils import FIGURES_DIR, RESULTS_DIR


def close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, observed {actual}")


def main() -> None:
    flow = pd.read_csv(RESULTS_DIR / "sample_flow.csv").set_index("arm")
    assert int(flow.loc["judgment", "n_attention_pass"]) == 49
    assert int(flow.loc["manipulation", "n_attention_pass"]) == 50

    checks = pd.read_csv(RESULTS_DIR / "data_quality_checks.csv").set_index("check")["value"]
    assert int(checks["unique_participants_raw"]) == 99
    assert int(checks["arm_sessions_raw"]) == 101
    assert int(checks["participants_in_both_arms"]) == 2
    assert int(checks["duplicate_response_keys"]) == 0
    assert int(checks["missing_response_values"]) == 0

    manipulation = pd.read_csv(RESULTS_DIR / "manipulation_target_effects.csv").set_index("contrast")
    close(manipulation.loc["Body", "mean_difference"], -5.36)
    close(manipulation.loc["Memory", "mean_difference"], -5.28)
    close(manipulation.loc["Function", "mean_difference"], -4.96)
    close(manipulation.loc["Self-ID uncertain", "mean_difference"], -3.50)
    close(manipulation.loc["Transition", "mean_difference"], -2.42)
    close(manipulation.loc["Self-denial anchor", "mean_difference"], -3.54)

    profiles = pd.read_csv(RESULTS_DIR / "judgment_profile_tests.csv").set_index("contrast")
    assert profiles.loc["Body", "p_fdr_six_profiles"] < 0.001
    assert profiles.loc["Memory", "p_fdr_six_profiles"] < 0.01
    assert profiles.loc["Self-ID uncertain", "p_fdr_six_profiles"] < 0.05
    assert profiles.loc["Self-denial anchor", "p_fdr_six_profiles"] < 0.05
    assert profiles.loc["Function", "p_fdr_six_profiles"] > 0.05
    assert profiles.loc["Transition", "p_fdr_six_profiles"] > 0.05

    parallel = pd.read_csv(RESULTS_DIR / "parallel_classification.csv").set_index("classification")
    assert int(parallel.loc["both_predecessor_links_high", "count"]) == 6
    assert int(parallel.loc["transitivity_like_discordance", "count"]) == 2

    for stem in [
        "fig0_catman_framework",
        "fig1_manipulation_selectivity",
        "fig2_judgment_profiles",
        "fig3_deny_and_parallel",
    ]:
        for suffix in ["png", "pdf"]:
            path = FIGURES_DIR / f"{stem}.{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty figure: {path}")

    print("[Cat-Man] verification checks passed")


if __name__ == "__main__":
    main()

