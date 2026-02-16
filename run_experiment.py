import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_CSV = "experiment_tracker.csv"
RESULTS_DIR = "04_results"
PROMPTS_DIR = "03_prompts/sent"
TEMPLATES_DIR = "03_prompts/templates"
IN_CONTEXT_DIR = "03_prompts/in_context"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TEMPLATE_BY_RQ = {
    "q1": "q1.txt",
    "q2": "q2.txt",
    "q3": "q3.txt",
    "q4": "q4.txt",
}


def call_llm(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _cell(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_path(value: str, candidates: list[Path]) -> Path | None:
    for base in candidates:
        candidate = base / value if str(base) != "." else Path(value)
        if candidate.is_file():
            return candidate
    return None


def _read_optional_file(value: str, candidates: list[Path], label: str) -> str:
    if value == "":
        return ""
    resolved = _resolve_path(value, candidates)
    if resolved is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"{label} file not found: '{value}'. Tried: {searched}")
    return resolved.read_text(encoding="utf-8").strip()


def _format_text_section(heading: str, body: str) -> str:
    if body == "":
        return ""
    return f"{heading}\n{body}"


def _format_code_section(heading: str, code: str, language: str = "python") -> str:
    if code == "":
        return ""
    return f"{heading}\n```{language}\n{code}\n```"


def _clean_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return f"{cleaned}\n"


def _load_template(row: pd.Series, rq: str) -> str:
    explicit_template = _cell(row, "prompt_template_file")
    if explicit_template:
        explicit_path = Path(explicit_template)
        if not explicit_path.is_file():
            explicit_path = Path("03_prompts") / explicit_template
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: '{explicit_template}'")
        return explicit_path.read_text(encoding="utf-8")

    if rq in TEMPLATE_BY_RQ:
        template_path = Path(TEMPLATES_DIR) / TEMPLATE_BY_RQ[rq]
        if not template_path.is_file():
            raise FileNotFoundError(f"RQ template not found: '{template_path}'")
        return template_path.read_text(encoding="utf-8")

    supported = ", ".join(sorted(TEMPLATE_BY_RQ))
    raise ValueError(f"Unsupported rq='{rq}'. Supported values: {supported}")


def _build_prompt(row: pd.Series) -> tuple[str, str]:
    game = _cell(row, "game")
    policy_type = _cell(row, "policy_type")
    policy_file = _cell(row, "policy_file")
    rq = _cell(row, "rq").lower() or "q2"

    if policy_file == "":
        raise ValueError("Missing required column value: policy_file")

    policy_path = Path("01_policies") / policy_type / game / policy_file
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: '{policy_path}'")
    policy = policy_path.read_text(encoding="utf-8").strip()

    env_description_file = _cell(row, "env_description_file")
    env_description = _read_optional_file(
        env_description_file,
        [Path("."), Path("02_contexts"), Path("02_contexts") / game],
        label="Environment description",
    )

    task_description = _cell(row, "task_description")
    if task_description == "":
        task_description = _read_optional_file(
            _cell(row, "task_description_file"),
            [Path("."), Path("02_contexts"), Path("02_contexts") / game, Path("02_contexts/tasks")],
            label="Task description",
        )

    reward_function = _cell(row, "reward_function")
    if reward_function == "":
        reward_function = _read_optional_file(
            _cell(row, "reward_function_file"),
            [Path("."), Path("02_contexts"), Path("02_contexts") / game, Path("02_contexts/reward_functions")],
            label="Reward function",
        )

    simplification_description = _cell(row, "simplification_description")
    if simplification_description == "":
        simplification_description = _read_optional_file(
            _cell(row, "simplification_file"),
            [Path("."), Path("02_contexts"), Path("02_contexts") / game, Path("02_contexts/simplifications")],
            label="Simplification",
        )

    include_in_context = _parse_bool(_cell(row, "include_in_context"), default=False)
    in_context_file = _cell(row, "in_context_file")
    if in_context_file == "" and include_in_context:
        in_context_file = f"{policy_type}.txt"

    in_context = _read_optional_file(
        in_context_file,
        [Path("."), Path(IN_CONTEXT_DIR)],
        label="In-context example",
    )

    use_cot = _parse_bool(_cell(row, "use_cot"), default=False)
    cot_instruction = (
        "Reason briefly through the policy conditions and returned actions before your final answer."
        if use_cot
        else "Give a concise answer grounded in the policy logic."
    )

    env_section = _format_text_section(
        "We have trained reinforcement-learning agents on the following environment:",
        env_description,
    )
    task_section = _format_text_section(
        "The task of the agent was to:",
        task_description,
    )
    reward_section = _format_code_section(
        "We used the following reward function to optimize the agent:",
        reward_function,
    )
    in_context_section = _format_text_section(
        "Reference in-context example (use only as reasoning style guidance):",
        in_context,
    )
    simplification_section = _format_text_section(
        "Now consider the following environment simplification:",
        simplification_description,
    )

    if rq == "q4" and simplification_section == "":
        raise ValueError(
            f"Row '{_cell(row, 'id')}' uses rq=q4 but no simplification is provided "
            "(set simplification_description or simplification_file)."
        )

    template = _load_template(row, rq)
    prompt = template
    replacements = {
        "{{RQ}}": rq.upper(),
        "{{GAME}}": game,
        "{{POLICY_TYPE}}": policy_type,
        "{{ENV_SECTION}}": env_section,
        "{{TASK_SECTION}}": task_section,
        "{{REWARD_SECTION}}": reward_section,
        "{{IN_CONTEXT_SECTION}}": in_context_section,
        "{{SIMPLIFICATION_SECTION}}": simplification_section,
        "{{SYMBOLIC_POLICY}}": policy,
        "{{COT_INSTRUCTION}}": cot_instruction,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return _clean_prompt(prompt), rq


def run_experiment(dry: bool = False, input_csv: str = INPUT_CSV):
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    client = None
    if not dry:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is missing. Set it in your .env file.")
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    for idx, row in df.iterrows():
        status = _cell(row, "status").lower()
        if status in {"done", "excluded", "skip", "skipped"}:
            continue

        row_id = _cell(row, "id")
        model = _cell(row, "target_llm")
        if row_id == "" or model == "":
            raise ValueError("Each row must define both 'id' and 'target_llm'.")

        prompt, rq = _build_prompt(row)

        model_slug = model.replace("/", "_")
        prompt_file = f"{row_id}_{rq}_{model_slug}_prompt.txt"
        prompt_path = Path(PROMPTS_DIR) / prompt_file
        prompt_path.write_text(prompt, encoding="utf-8")
        df.at[idx, "prompt_file"] = str(prompt_path)

        if dry:
            continue

        result = call_llm(client, model, prompt)
        result_file = f"{row_id}_{rq}_{model_slug}_result.txt"
        result_path = Path(RESULTS_DIR) / result_file
        result_path.write_text(result, encoding="utf-8")
        df.at[idx, "result_file"] = str(result_path)
        df.at[idx, "status"] = "done"

    df.to_csv(input_csv, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Skip LLM calls.",
    )
    parser.add_argument(
        "--csv",
        default=INPUT_CSV,
        help="Path to experiment tracker CSV (default: experiment_tracker.csv).",
    )
    args = parser.parse_args()
    run_experiment(dry=args.dry, input_csv=args.csv)
