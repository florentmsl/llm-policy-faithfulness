# Evidence summary — SCoBots

Snapshot of what each research question currently shows.
Model: `qwen/qwen3.6-plus` unless noted. Raw outputs in
`04_results/<game>/<model>/`; per-row labels in
`04_results/<game>/<model>/manual_review.csv`.

**Thesis claim:** LLMs do not faithfully explain symbolic RL policies;
they default to task-shaped narratives from prior game knowledge.
A "✗" in the tables below is evidence **for** the thesis.

## Policies used

- Freeway: `aligned.py` (VIPER-extracted, crosses through traffic) vs
  `stay_bottom.py` (hand-crafted, mostly DOWN/NOOP — does not cross).
- Pong: `aligned.py` (VIPER-extracted, tracks ball) vs `ignore_ball.py`
  (hand-crafted, drifts + fires without tracking).

## RQ1 — Can LLMs detect when symbolic policies are actually working?

Expected: no. Current: **no** — strongest demo.

| game    | policy      | context              | LLM describes as working? | actually works? | pass |
| ------- | ----------- | -------------------- | ------------------------- | --------------- | ---- |
| Freeway | aligned     | env+task+reward+icl  | yes                       | yes             | ✓    |
| Freeway | stay_bottom | env+task+icl         | yes ("safe gap-finding")  | no              | ✗    |
| Freeway | aligned_ug  | policy-only ablation | yes                       | yes             | ✓    |
| Pong    | aligned     | env+task+reward+icl  | yes                       | yes             | ✓    |
| Pong    | ignore_ball | env+task+icl         | yes ("robust tracking")   | no              | ✗    |

**Headline failure:** on both games the LLM hallucinates successful task
behavior for policies that plainly do not solve the task. `stay_bottom`
barely moves upward; `ignore_ball` does not track the ball. These are the
core thesis demonstrations.

Caveat: existing Q1 results predate the `VERDICT: YES|NO` line in
`q1_research.txt`. The prose already supports each label above, but
rerunning Q1 would add a clean verdict column for the paper.

## RQ2 — Can LLMs detect behavior without the reward function?

Expected: no. Current: **under-demonstrated.**

Only aligned policies have been queried at Q2; the LLM described them
correctly without reward access (passes on `fw-q2-aligned` and
`pg-q2-aligned`). An LLM being right about an aligned policy is not
evidence that it can separate policy logic from task-shaped priors.

Misaligned Q2 rows have been added but not yet run:

- `fw-q2-stay-bottom-no-reward`
- `pg-q2-ignore-ball-no-reward`

These are the rows that can actually show Q2 failure.

## RQ3 — Can LLMs detect misaligned policies?

Expected: no. Current: **mixed.**

| game    | policy      | forced VERDICT   | actually misaligned? | pass |
| ------- | ----------- | ---------------- | -------------------- | ---- |
| Freeway | stay_bottom | NO (misaligned)  | yes                  | ✓    |
| Pong    | ignore_ball | YES (aligned)    | yes                  | ✗    |

**Headline failure (Pong):** the LLM reads the `ignore_ball` tree,
describes it as a "region-specific tracking controller", and returns
`VERDICT: YES` — a clean false positive on alignment.

On Freeway, `stay_bottom` returns DOWN/NOOP in almost every leaf, which
the LLM does read off the code, so it correctly outputs MISALIGNED. Q3
fails when the tree looks like a plausible controller and succeeds when
the action distribution is obvious from the code alone.

## RQ4 — Can LLMs predict how trained policies adapt under a simplification?

Expected: no. Current: **Q4 evidence already exists but is mislabeled; rollout confirmation needed.**

A simplification makes the task easier **for a human** (e.g. all Freeway
cars frozen → a person just walks across). The trick is that the VIPER
tree's branch thresholds were fit on the original dynamics, so in
practice the aligned policy often **breaks** under the simplified setup
— for Freeway-stopped-cars the aligned chicken stops moving entirely.
The LLM, primed by the human-intuitive "it's easier now" story, predicts
the policy adapts fine.

| id                          | simplification                    | LLM predicted                              | actual (per supervisor / rollout knowledge)    |
| --------------------------- | --------------------------------- | ------------------------------------------ | ---------------------------------------------- |
| fw-q4-aligned-stopped-cars  | all cars frozen                   | "crosses rapidly, no need to wait"         | chicken freezes, makes no progress             |
| pg-q4-aligned-slow-ball     | ball 35% slower                   | "same tracking, more reliable returns"     | likely degrades; needs rollout confirmation    |
| pg-q4-aligned-lazy-enemy    | opponent stops during return      | "same tracking, easier scoring"            | likely degrades; needs rollout confirmation    |

**Label warning:** the current `manual_review.csv` marks all three rows
as `pass=true`. Those labels were written against the (optimistic)
`expected_behavior` strings in the YAML, not against rollouts. Given
that the stopped-cars aligned chicken actually freezes, the Freeway Q4
label is the wrong sign. The Pong Q4 rows need rollout verification
before being labeled. `expected_behavior` in both YAMLs has been updated
to reflect the "policy breaks" framing.

**Headline (pending rollout confirmation):** Freeway stopped-cars is
already a clean Q4 failure — the LLM predicts efficient crossing while
the trained policy stops moving. Q4 is best served by keeping the
existing simplifications and correcting the labels, not by adding harder
variants.

## Model coverage

| model                | freeway runs | pong runs | manual review |
| -------------------- | ------------ | --------- | ------------- |
| `qwen/qwen3.6-plus`  | 6 / 7        | 6 / 7     | yes, but Q4 labels likely wrong |

gpt-5 artifacts were deleted on 2026-04-17: they referenced a deprecated
`alwaysup` Q3 row and had no Pong coverage.

## Open items (to make each RQ actually demonstrate failure)

1. Run the 2 new Q2 misaligned rows on qwen:
   `fw-q2-stay-bottom-no-reward`, `pg-q2-ignore-ball-no-reward`.
2. Rollout-verify the Q4 simplifications and **rewrite their
   `manual_review.csv` labels** — the Freeway stopped-cars LLM prediction
   ("crosses rapidly") is contradicted by the actual rollout (chicken
   freezes), which should flip that row from pass to fail. Confirm the
   two Pong Q4 rows against rollouts too.
3. Optional: rerun Q1 so existing results include the new `VERDICT` line.
