# Result Summary

Model: `nvidia/nemotron-3-super-120b-a12b:free`

| slice | correct | takeaway |
| --- | ---: | --- |
| All labeled rows | 14/19 yes, 1 partial | Mixed aggregate; useful sanity check, not the thesis result. |
| Primary rollout-backed verdict probes | 4/8 | Behavior-level prediction is unreliable. |
| Q1 outcome verdicts | 5/7 all; 2/4 primary | Fails both working aligned policies. |
| Q2 descriptions | 4/5 yes, 1 partial | Mechanical code tracing mostly works. |
| Q3 intent verdicts | 4/5 all; 1/2 primary | Still turns retreat-heavy structure into false alignment. |
| Q4 counterfactual verdicts | 1/2 | Misses one dynamics-dependence failure. |
| Diagnostic controls | 8/9 yes, 1 partial | Obvious hand controls are mostly handled. |

Read this as a sliced result, not a total-accuracy benchmark: the model can trace simple policy code, but the primary rollout-backed probes expose where explanation stops matching behavior.
