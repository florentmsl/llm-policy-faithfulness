# LLM Policy Faithfulness

Thesis claim: LLMs can trace transparent symbolic policies, but their explanations become behaviorally unfaithful when correct interpretation requires omitted-variable checks, task semantics, or closed-loop counterfactual prediction.

Active target: game-name-blinded semantic SCoBots policies for Pong and Freeway. Game names and original object labels are stripped; neutral object roles, action meanings, and task objectives are included where the research question requires them.

For current evidence status, see `summary.md`. For labeling rules, see `docs/labeling-rubric.md`.

## Research Questions

- Q1. Can LLMs detect when symbolic policies are working?
- Q2. Can LLMs describe policy behavior without reward-function text?
- Q3. Can LLMs detect misaligned policies?
- Q4. Can LLMs correctly predict how a trained symbolic policy adapts to environment simplification?

## Experiment Setup

Active experiments file: `experiments/blinded.yml`.

## Prompt Templates

- `03_prompts/templates/q1_blinded.txt` — task-success verdict, `VERDICT: YES|NO`
- `03_prompts/templates/q2_blinded.txt` — behavior description, no forced verdict
- `03_prompts/templates/q3_blinded.txt` — task-alignment verdict, `VERDICT: YES|NO`
- `03_prompts/templates/q4_blinded.txt` — simplification performance direction, `VERDICT: BETTER|SAME|WORSE|UNCLEAR`

Templates use placeholders such as `{{ENV_DESCRIPTION}}`, `{{TASK_DESCRIPTION}}`, `{{ENV_SIMPLIFICATION_DESCRIPTION}}`, and `{{SYMBOLIC_POLICY}}`.

## Run

```bash
cp .env.example .env  # set OPENROUTER_API_KEY
make experiments-dry  # dry run: generates prompts only
make experiments-run  # real run against the YAML default model
make aggregate        # current-label aggregation only
```

`run.py` writes prompts to `03_prompts/sent/<run_group>/<model_key>/`, raw results to `04_results/<run_group>/<model_key>/<id>_result.txt`, and metadata to `<id>_meta.json`. The active run group is `blinded_semantic`. Existing non-placeholder result files and their prompt artifacts are not overwritten.

## Manual Labeling

`04_results/<run_group>/<model_key>/summary.csv` is hand-maintained. Current labels should follow `docs/labeling-rubric.md`.

## Source Policies

Source policies live in `01_policies/scobots/{pong,freeway}/`. `tools/blind_policy.py` derives neutralized versions into `01_policies/scobots/_blinded/` by renaming object identifiers and stripping action-name comments from policy code.
