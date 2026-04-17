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

Environment setup:

```bash
cp .env.example .env
```

Set these in `.env` before any non-dry run:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (for example `qwen/qwen3.6-plus`) if you want a default model outside the YAML files

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
- `--model` optionally overrides `OPENROUTER_MODEL` and the YAML default, and accepts a raw OpenRouter model ID such as `qwen/qwen3.6-plus`.

## Harness Flow

Recommended end-to-end workflow for a model run:

1. Create or update the relevant experiment rows in `experiments/*.yml`.
2. Create `.env` from `.env.example` and set `OPENROUTER_API_KEY`, plus optionally `OPENROUTER_MODEL`.
3. Dry run once to generate prompts without paying for model calls:

```bash
uv run python run.py --file experiments/freeway.yml --model qwen/qwen3.6-plus --dry
```

4. Run the real batch with the target OpenRouter model:

```bash
uv run python run.py --file experiments/freeway.yml --model qwen/qwen3.6-plus
uv run python run.py --file experiments/pong.yml --model qwen/qwen3.6-plus
```

5. Inspect `04_results/<game>/<model_key>/summary.csv` to compare the grounded policy behavior against the saved LLM explanation.
6. Manually label faithfulness in `04_results/<game>/<model_key>/manual_review.csv` using:
   - `id`: experiment row id
   - `pass`: `true` if the explanation is behaviorally faithful, `false` otherwise
   - `notes`: short justification grounded in actual policy behavior

Manual review rules:

- Keep raw prompt artifacts in `03_prompts/sent/<game>/<model_key>/` unchanged.
- Keep raw model outputs in `04_results/<game>/<model_key>/` unchanged.
- Store review labels separately in `manual_review.csv` rather than rewriting result files.

## Outputs

- Prompt artifacts: `03_prompts/sent/<game>/<model_key>/`
- Result files: `04_results/<game>/<model_key>/`
- Logs: `04_results/<game>/<model_key>/summary.csv`
  - Columns: `experiment_id`, `policy_behavior`, `llm_behavior`
- Manual labels: `04_results/<game>/<model_key>/manual_review.csv`
  - Columns: `id`, `pass`, `notes`

## Available Policies

Paths are relative to `01_policies/`. Status notes reflect policy quality, not file presence.

| Game     | SCoBots                       | INSIGHT                                               | NUDGE                                                       |
| -------- | ----------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| Pong     | `scobots/pong/aligned.py`     | `insight/pong/aligned.txt` (works)                    | `nudge/pong/rules.txt` (not reliable)                       |
| Freeway  | `scobots/freeway/aligned.py`  | `insight/freeway/aligned.txt` (degenerate, always DOWN) | `nudge/freeway/rules.txt` (degenerate, always UP)         |
| Kangaroo | `scobots/kangaroo/aligned.py` | `insight/kangaroo/aligned.txt` (partial, coconut only) | `nudge/kangaroo/rules.txt` (does not avoid collisions)    |
| Skiing   | `scobots/skiing/aligned.py`   | `insight/skiing/aligned.txt` (degenerate, all zeros)  | `nudge/skiing/rules.txt` (always left)                      |
| Seaquest | `scobots/seaquest/aligned.py` | missing (OCAtari bug?)                                | `nudge/seaquest/rules.txt` (works)                          |

Misaligned / ablation variants (e.g. `instahit.txt`, `stay_bottom.py`, `ignore_ball.py`, `_ug` object-name-ablation files, `*_rf.*` reward-variant files) live alongside the aligned policy in each `<framework>/<game>/` directory.

## Contexts

- Freeway [(AtariAge)](https://www.atariage.com/2600/manuals_old/freeway.html)
- Pong [(AtariAge)](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
