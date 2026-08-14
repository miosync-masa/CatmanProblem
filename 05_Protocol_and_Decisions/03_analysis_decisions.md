# Analysis decisions for the public package

These decisions were fixed while preparing the exploratory manuscript package after data collection unless explicitly marked otherwise.

## Primary sample

- Primary judgment analysis: 49 participants with `attention_pass == 1`.
- Primary manipulation analysis: all 50 participants, all of whom passed.
- Sensitivity analysis: repeat judgment profile tests with all 51 judgment participants.

## Reference condition

Single-cue and anchor effects are computed as within-participant condition-minus-`C01` differences. Transition stress tests use `C06-C01`, `C07-C02`, and `C08-C03` to compare discontinuous versus continuous processing within three cue contexts.

## Effect sizes and intervals

For each paired difference:

- report the mean difference;
- report a two-sided 95% Student-t confidence interval;
- report paired-samples `d_z = mean(diff) / SD(diff)`;
- report the paired t statistic and two-sided p value.

Benjamini-Hochberg FDR correction is applied within each clearly labeled family of tests.

## Query-profile permutation test

For each contrast, form an `N × 5` matrix of participant-level condition-minus-baseline differences. The observed statistic is the sum of squared deviations of the five query means from their grand mean. Under the null of an exchangeable/flat query profile, query labels are randomly permuted within each participant. Each test uses 100,000 Monte Carlo permutations and a fixed seed. The six profile p values are FDR-corrected together.

This permutation procedure was not preregistered and is reported as a direct exploratory test of profile heterogeneity.

## Parallel classification

- “High” is operationalized as ratings 5–7.
- `both_predecessor_links_high`: `j_ab >= 5` and `j_ac >= 5`.
- `transitivity_like_discordance`: both predecessor links high while `j_bc <= 4` (not high).

These thresholds are transparent exploratory operationalizations, not natural-kind cutoffs. Raw ratings and Wilson intervals are provided so alternative thresholds can be evaluated.

## Correlations

Judgment correlations are descriptive participant-level correlations among each participant's mean outcome ratings across the nine non-Parallel conditions. They describe response tendencies and are not used as independent evidence for the causal claims.

