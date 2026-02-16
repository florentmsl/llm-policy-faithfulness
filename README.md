# LLM Policy Faithfulness

Goal: prevent blind trust in LLM explanations by asking models to describe
the actual behavior of symbolic policies, including misaligned ones.

## Structure

- `01_policies/` -> symbolic policies
- `02_contexts/` -> game descriptions
- `03_prompts/templates/` -> Q1-Q4 prompt templates
- `03_prompts/in_context/` -> shared in-context examples per policy type
- `03_prompts/sent/` -> saved prompts per run
- `04_results/` -> LLM outputs
- `experiment_tracker.csv` -> experiment matrix + run status
- `run_experiment.py` -> batch runner

## Contexts

- Freeway [(AtariAge)](https://www.atariage.com/2600/manuals_old/freeway.html)
- Pong [(AtariAge)](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)

## Runner extensions (Q1-Q4 + optional sections)

`run_experiment.py` now supports per-row prompt building for `q1`, `q2`, `q3`, `q4`.

Required CSV columns (same as before):

- `id`, `game`, `policy_type`, `policy_file`, `target_llm`, `status`

Optional CSV columns:

- `rq`: `q1` / `q2` / `q3` / `q4` (default: `q2`)
- `env_description_file`
- `task_description_file` or `task_description`
- `reward_function_file` or `reward_function`
- `include_in_context`: boolean (`true`/`false`)
- `in_context_file`: override; if omitted and `include_in_context=true`, default is `03_prompts/in_context/<policy_type>.txt`
- `use_cot`: boolean
- `simplification_file` or `simplification_description` (required for `q4`)
- `prompt_template_file`: optional override template path

Notes:

- Reward section is included only when a reward function is provided.
- In-context section is included only when requested/provided.
- `q1` enforces final `VERDICT: YES/NO`.
- `q3` enforces final `VERDICT: ALIGNED/MISALIGNED`.
- `q4` enforces final `VERDICT: ADAPTS/FAILS_TO_ADAPT`.

## Example rows

Freeway, Q1 with reward + CoT:

```csv
id,game,policy_type,policy_file,env_description_file,task_description_file,reward_function_file,rq,include_in_context,use_cot,target_llm,status
fw-q1-001,freeway,scobots,instahit.txt,freeway/environment.txt,02_contexts/freeway/task.txt,02_contexts/reward_functions/freeway_jaxatari.py,q1,true,true,openai/gpt-4o,pending
```

Pong, Q2 without reward (behavior-only):

```csv
id,game,policy_type,policy_file,env_description_file,task_description_file,rq,include_in_context,use_cot,target_llm,status
pg-q2-001,pong,scobots,<your_pong_policy>.txt,pong/environment.txt,02_contexts/pong/task.txt,q2,true,false,openai/gpt-4o,pending
```
