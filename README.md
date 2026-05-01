# LLM Policy Faithfulness

Thesis claim: LLMs are not faithful when explaining symbolic RL policies.
With game-name cues stripped (the active condition in this repo), the model
invents action semantics that fit whatever the policy outputs and declares
the policy coherent — describing a self-consistent narrative rather than
the policy's actual rollout-grounded behavior.

Goal: demonstrate this unfaithfulness on game-blinded SCoBots policies in
Pong and Freeway, comparing the LLM's free-form description and YES/NO
verdicts against rollout-backed ground truth.

## Research questions

- Q1. Can LLMs detect when symbolic policies are working?
- Q2. Can LLMs describe the behavior of the policy without the reward function?
- Q3. Can LLMs detect misaligned policies?
- Q4. Can LLMs correctly predict how a correctly trained symbolic policy will adapt to a simplification of the environment? *(deferred — needs game-blind simplification descriptions.)*

For current evidence per RQ see `summary.md`. Headline: 6/6 unfaithful on
the diagnostic misaligned-only subset; 100% unfaithful on Q2 and Q3.

## Experiment setup

Active experiments file: `experiments/blinded.yml`. Defines defaults
(`game`, `model`, template paths) and experiment rows (`id`, `rq`,
`policy_file`, plus optional row-level overrides).

The legacy game-aware track (`experiments/{freeway,pong}.yml`,
`q*_research.txt`, Atari-manual env descriptions, in-context tracking
example) was removed on 2026-05-01. It conflated faithfulness with
game-prior reliance: aligned-policy YES verdicts could not distinguish
"the model traced the code" from "the model guessed YES from game prior."

## Prompt templates

- `03_prompts/templates/q1_blinded.txt` — coherent goal-directed behavior, forces `VERDICT: YES|NO`
- `03_prompts/templates/q2_blinded.txt` — free-form behavior description, no verdict
- `03_prompts/templates/q3_blinded.txt` — relation-grounded vs degenerate, forces `VERDICT: YES|NO`

Templates use placeholders `{{ENV_DESCRIPTION}}` and `{{SYMBOLIC_POLICY}}`,
with optional blocks `{{#NAME}}...{{/NAME}}` rendered only when the value
is non-empty.

## Run

```bash
cp .env.example .env  # set OPENROUTER_API_KEY
make experiments-dry  # dry run — generates prompts only
make experiments-run  # real run against the YAML default model
make aggregate        # per-RQ pass rates and failure list
```

`run.py` writes prompts to `03_prompts/sent/<game>/<model_key>/`, raw
results to `04_results/<game>/<model_key>/<id>_result.txt`, and a JSON
sidecar with response_id / model_resolved / usage / timestamp at
`<id>_meta.json`. Dry-run results contain a placeholder; existing
non-placeholder result files are not overwritten.

## Manual labeling

`04_results/<game>/<model_key>/summary.csv` is hand-maintained. Columns:
`experiment_id`, `policy_behavior`, `llm_behavior`, `pass`. `pass=true`
means the LLM explanation is behaviorally faithful to the policy.

## Source policies

`01_policies/scobots/{pong,freeway}/{aligned,ignore_ball,stay_bottom}.py`
are the source artifacts. `tools/blind_policy.py` derives blinded versions
into `01_policies/scobots/_blinded/` by renaming game-specific object
names (Ball/Player/Enemy/Chicken/Car → Obj_A/Agent/Obj_B/Hazard_*) and
stripping action-name comments. SCoBots DSL operators (`D`, `ED`, `LT`,
`C`, `V`, `DV`) are preserved as game-agnostic feature ops.
