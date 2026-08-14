# Analysis

The analysis is implemented in Python 3 and is deterministic. Run the complete pipeline from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_all.py
```

Scripts run in numeric order. They clean and validate the public raw data, reproduce all statistical tables, run 100,000-permutation query-profile tests, perform sensitivity analyses, generate publication figures in PNG and PDF, and verify the key sample counts and reported results.

The permutation test shuffles query labels within each participant's vector of condition-minus-baseline differences. Its statistic is the sum of squared deviations of the five query-specific mean effects from their grand mean. This directly tests whether an intervention has a flat effect across queries.

All random procedures use fixed seeds beginning with `20260814`.
