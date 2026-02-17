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

- defaults (`game`, `model`, `icl_file`, template paths)
- experiment rows (`id`, `rq`, `include_reward`, and file paths)

## Prompt templates (per question Q1-Q4)

Prompt structure is fully defined inside templates:

- `03_prompts/templates/q1_research.txt`
- `03_prompts/templates/q2_research.txt`

These templates use only placeholders for content:

- `{{ENV_DESCRIPTION}}`
- `{{TASK_NATURAL_LANGUAGE_DESCRIPTION}}`
- `{{PYTHON_JAXATARI_REWARD_FUNCTION}}`
- `{{SYMBOLIC_POLICY}}`
- `{{IN_CONTEXT_LEARNING_EXAMPLE}}`

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

Options:

- `--file` is required.
- `--overwrite-existing` to overwrite existing result files.

## Outputs

- Prompt artifacts: `03_prompts/sent/<game>/`
- Result files: `04_results/<game>/`
- Logs: `04_results/<game>/<model_slug>_summary.csv`

## Contexts

- Freeway [(AtariAge)](https://www.atariage.com/2600/manuals_old/freeway.html)
- Pong [(AtariAge)](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
