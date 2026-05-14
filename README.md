# LLM Policy Faithfulness

Master's thesis harness. Asks: *do LLM natural-language explanations of symbolic RL policies actually describe what the policy does?*

**Claim:** they do not, in structurally predictable ways. LLMs trace transparent decision-tree code fine, then become unfaithful when correct interpretation requires (a) noticing an omitted task-relevant variable, (b) seeing past surface tree-structure on deep VIPER policies, or (c) predicting closed-loop behavior under changed environment dynamics.

**Takeaway:** LLM-generated policy explanations cannot be used as ground truth without behavioral verification.

## Research questions

| RQ | Question | Output |
| --- | --- | --- |
| Q1 | Will this policy *succeed* in rollout? | `YES` / `NO` |
| Q2 | What does this policy do? (no reward provided) | free-form description |
| Q3 | Are the policy's actions *directed at* the task? | `YES` / `NO` |
| Q4 | Performance direction under a simplification? | `BETTER` / `SAME` / `WORSE` / `UNCLEAR` |

Q1 is outcome (will it work). Q3 is intent (is it trying). They are distinct probes.

## Scope

- **Environments:** Pong, Freeway.
- **Policy framework:** SCoBots (object-centric decision trees, Delfosse et al. 2024).
- **Condition:** game names blinded; neutral object/action/task semantics preserved.
- **Active policies:** 7 (4 rollout-backed VIPER + 3 hand-crafted diagnostic controls).
- **Active rows:** 19 (`experiments.yml`).

## Run

```bash
cp .env.example .env       # set OPENROUTER_API_KEY
make dry                   # build prompts only, no API calls
make run                   # real run against YAML default model
```

Override model with `--model` flag or `OPENROUTER_MODEL` env var. Outputs land in `results/<model>/`. Labels go in `results/labels.csv` (hand-edited).

## Repository

```
policies/<game>/*.py        symbolic policies (blinded canonical)
contexts/<game>/            environment, task, simplification descriptions
prompts/templates/q*.txt    one template per RQ
ground_truth.csv            rollout-backed answer key
experiments.yml             experiment definitions
run.py                      batch runner via OpenRouter
results/labels.csv          single labels file across all models
AGENTS.md                   agent-facing repo contract
```

See `AGENTS.md` for the full thesis framing, labeling rubric, policy provenance, and workflow.
