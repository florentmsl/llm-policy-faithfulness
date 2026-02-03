# LLM Policy Faithfulness

Goal: prevent blind trust in LLM explanations by asking models to describe
the actual behavior of symbolic policies, including misaligned ones.

## Structure

- `01_policies/` -> symbolic policies
- `02_contexts/` -> game descriptions
- `03_prompts/template.txt` -> prompt template
- `03_prompts/sent/` -> saved prompts per run
- `04_results/` -> LLM outputs
- `experiment_tracker.csv` -> experiment matrix + run status
- `run_experiment.py` -> batch runner

## Contexts

- Freeway [(AtariAge)](https://www.atariage.com/2600/manuals_old/freeway.html)
