# LLM Policy Faithfulness

Goal: prevent blind trust in LLM explanations by asking models to describe
the actual behavior of symbolic policies, including misaligned ones.

## Research questions

- Q1. Can LLMs detect when symbolic policies are working?
- Q2. Can LLMs detect the behavior of the policy without the reward function?
- Q3. Can LLMs detect misaligned policies?
- Q4. Can LLMs correctly predict how a correctly trained symbolic policy will adapt to a simplification?

Guess: Q1-Q4 are a "no".

## Experiment setup

Experiments are YAML-driven.

- `experiments/freeway.yml`
- `experiments/pong.yml`

Each YAML file defines:

- defaults (`game`, `model`, template paths, plus optional default context/reward/simplification/icl files)
- experiment rows (`id`, `rq`, `policy_file`, plus optional row-level overrides)

## Prompt templates (per question Q1-Q4)

Prompt structure is fully defined inside templates:

- `03_prompts/templates/q1_research.txt`
- `03_prompts/templates/q2_research.txt`
- `03_prompts/templates/q3_research.txt`
- `03_prompts/templates/q4_research.txt`

RQ semantics:

- `q1`: policy working check (template now enforces final `VERDICT: YES|NO`)
- `q2`: behavior description without forced verdict
- `q3`: alignment check (template now enforces final `VERDICT: ALIGNED|MISALIGNED`)
- `q4`: behavior prediction under simplification

Input toggles:

- `env_file`, `task_file`, `reward_file`, `simplification_file`, and `icl_file` are optional per experiment row.
- You can define optional defaults in `defaults` and override per row.
- Set a field to an empty string (`""`) in a row to explicitly disable an inherited default.

These templates use only placeholders for content:

- `{{ENV_DESCRIPTION}}`
- `{{TASK_DESCRIPTION}}`
- `{{REWARD_FUNCTION}}`
- `{{ENV_SIMPLIFICATION_DESCRIPTION}}`
- `{{SYMBOLIC_POLICY}}`
- `{{IN_CONTEXT_LEARNING_EXAMPLE}}`

Optional template blocks are supported:

- `{{#PLACEHOLDER_NAME}} ... {{/PLACEHOLDER_NAME}}`
- A block is included only if that placeholder has non-empty content for the experiment.

## Run

Freeway run:

```bash
uv run python run.py --file experiments/freeway.yml
```

Freeway dry run (No LLM Call):

```bash
uv run python run.py --file experiments/freeway.yml --dry
```

Pong run:

```bash
uv run python run.py --file experiments/pong.yml
```

Pong dry run (No LLM Call):

```bash
uv run python run.py --file experiments/pong.yml --dry
```

Dry-run behavior:

- Prompt files are generated as usual.
- Result files are also generated and contain: `---- This was a dry run`
- Existing result files are overwritten only if they contain exactly `---- This was a dry run`.
- Existing non-placeholder result files are left untouched.

Options:

- `--file` is required.

## Outputs

- Prompt artifacts: `03_prompts/sent/<game>/<model_key>/`
- Result files: `04_results/<game>/<model_key>/`
- Logs: `04_results/<game>/<model_key>/summary.csv`
  - Includes per-run flags: `uses_env`, `uses_task`, `uses_reward`, `uses_simplification`, `uses_icl`

## Available Policies

Trained aligned policies from `../master/policies/`. 12 out of 15 cells are non-degenerate.

| Game | SCoBots (decision tree) | INSIGHT (polynomial eqs) | NUDGE (FOL rules) |
|------|------------------------|--------------------------|-------------------|
| Pong | `pong/scobots/policy.py` | `pong/insight/equations.txt` | `pong/nudge/rules.txt` |
| Freeway | `freeway/scobots/policy.py` | degenerate (always DOWN) | degenerate (always UP) |
| Kangaroo | `kangaroo/scobots/policy.py` | partial (coconut features only) | `kangaroo/nudge/rules.txt` |
| Skiing | `skiing/scobots/policy.py` | degenerate (all zeros) | `skiing/nudge/rules.txt` |
| Seaquest | `seaquest/scobots/policy.py` | missing (OCAtari bug) | `seaquest/nudge/rules.txt` |

### Policy types

- **SCoBots**: Decision trees (if/else Python). Features like `state['Player1.y']`, `state['Ball1.x']`.
- **INSIGHT**: Polynomial equations. `logits_RIGHT = 0.12*ball_x_1*enemy_y_2 + ...`
- **NUDGE/BlendRL**: First-order logic rules. `up_track(X):-ball_above(B,P),ball_approaching(B,P).`

### Pre-recorded INSIGHT videos

Located in `../master/insight/output/videos/`:

| Game | Aligned (EQL) | Aligned (Neural) | Misaligned (basic_rf) |
|------|--------------|------------------|----------------------|
| Pong | `Pong..._oc_eql.mp4` | `Pong..._oc_neural.mp4` | `Pong..._basic_rf_oc_eql.mp4` |
| Freeway | `Freeway..._oc_eql.mp4` | `Freeway..._oc_neural.mp4` | `Freeway..._basic_rf_oc_eql.mp4` |

Additional Pong reward functions: `random_rf`, `up_and_down_rf`, `opposite_of_enemy_rf`, `close_but_no_hit_rf`.
Additional Freeway reward functions: `random_rf`, `stay_in_middle_rf`.

## Contexts

- Freeway [(AtariAge)](https://www.atariage.com/2600/manuals_old/freeway.html)
- Pong [(AtariAge)](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
