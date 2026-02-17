import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPTIONAL_BLOCK_PATTERN = re.compile(r"{{#([A-Z0-9_]+)}}(.*?){{/\1}}", re.DOTALL)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    game: str
    rq: str
    policy_file: str
    env_file: str | None
    task_file: str | None
    reward_file: str | None
    simplification_file: str | None
    icl_file: str | None


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _read_optional_file(path: str | None) -> str:
    return _read_file(path) if path else ""


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _normalize_prompt(prompt: str) -> str:
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip() + "\n"


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def _resolve_optional_field(row: dict[str, object], field_name: str, default_value: str | None) -> str | None:
    if field_name in row:
        return _optional_string(row.get(field_name))
    return default_value


def _render_optional_blocks(template: str, replacements: dict[str, str]) -> str:
    def replace_block(match: re.Match[str]) -> str:
        key = match.group(1)
        block_content = match.group(2)
        value = replacements.get(key, "")
        return block_content if value.strip() else ""

    rendered = template
    while True:
        next_rendered = OPTIONAL_BLOCK_PATTERN.sub(replace_block, rendered)
        if next_rendered == rendered:
            break
        rendered = next_rendered
    return rendered


def _load_experiment_file(path: Path) -> tuple[str, dict[str, str], str, list[Experiment]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data["defaults"]

    model = str(defaults["model"]).strip()
    game = str(defaults["game"]).strip()
    default_env_file = _optional_string(defaults.get("env_file"))
    default_task_file = _optional_string(defaults.get("task_file"))
    default_reward_file = _optional_string(defaults.get("reward_file"))
    default_simplification_file = _optional_string(defaults.get("simplification_file"))
    default_icl_file = _optional_string(defaults.get("icl_file"))

    template_by_rq = {str(rq).strip().lower(): str(template_path).strip() for rq, template_path in defaults["templates"].items()}

    experiments: list[Experiment] = []
    for row in data.get("experiments", []):
        experiment_id = str(row["id"]).strip()
        rq = str(row["rq"]).strip().lower()
        policy_file = str(row["policy_file"]).strip()
        env_file = _resolve_optional_field(row, "env_file", default_env_file)
        task_file = _resolve_optional_field(row, "task_file", default_task_file)
        reward_file = _resolve_optional_field(row, "reward_file", default_reward_file)
        simplification_file = _resolve_optional_field(row, "simplification_file", default_simplification_file)
        icl_file = _resolve_optional_field(row, "icl_file", default_icl_file)

        experiments.append(
            Experiment(
                experiment_id=experiment_id,
                game=game,
                rq=rq,
                policy_file=policy_file,
                env_file=env_file,
                task_file=task_file,
                reward_file=reward_file,
                simplification_file=simplification_file,
                icl_file=icl_file,
            )
        )

    return model, template_by_rq, game, experiments


def _build_prompt(experiment: Experiment, template_path: str) -> str:
    template = _read_file(template_path)
    env_text = _read_optional_file(experiment.env_file)
    task_text = _read_optional_file(experiment.task_file)
    policy_text = _read_file(experiment.policy_file)
    icl_text = _read_optional_file(experiment.icl_file)
    reward_text = _read_optional_file(experiment.reward_file)
    simplification_text = _read_optional_file(experiment.simplification_file)

    replacement_values = {
        "ENV_DESCRIPTION": env_text,
        "TASK_DESCRIPTION": task_text,
        "REWARD_FUNCTION": reward_text,
        "ENV_SIMPLIFICATION_DESCRIPTION": simplification_text,
        "SYMBOLIC_POLICY": policy_text,
        "IN_CONTEXT_LEARNING_EXAMPLE": icl_text,
    }

    prompt = _render_optional_blocks(template, replacement_values)
    for key, value in replacement_values.items():
        placeholder = f"{{{{{key}}}}}"
        prompt = prompt.replace(placeholder, value)

    return _normalize_prompt(prompt)


def _call_llm(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run(experiments_file: Path, dry: bool, overwrite_existing: bool) -> None:
    model, template_by_rq, game, experiments = _load_experiment_file(experiments_file)
    model_slug = _model_slug(model)

    prompts_dir = Path("03_prompts/sent") / game
    results_dir = Path("04_results") / game
    prompts_dir.mkdir(parents=True, exist_ok=True)
    if not dry:
        results_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not dry:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))

    summary_rows: list[dict[str, str]] = []
    for experiment in experiments:
        template_path = template_by_rq[experiment.rq]

        prompt = _build_prompt(experiment, template_path=template_path)
        prompt_file = prompts_dir / f"{experiment.experiment_id}_{model_slug}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        result_file = results_dir / f"{experiment.experiment_id}_{model_slug}_result.txt"
        status = "dry"
        if not dry:
            if result_file.is_file() and not overwrite_existing:
                status = "skipped_existing"
            else:
                status = "done"
                response_text = _call_llm(client, model, prompt)
                result_file.write_text(response_text, encoding="utf-8")

        summary_rows.append(
            {
                "id": experiment.experiment_id,
                "game": experiment.game,
                "rq": experiment.rq,
                "uses_env": "true" if experiment.env_file else "false",
                "uses_task": "true" if experiment.task_file else "false",
                "uses_reward": "true" if experiment.reward_file else "false",
                "uses_simplification": "true" if experiment.simplification_file else "false",
                "uses_icl": "true" if experiment.icl_file else "false",
                "model": model,
                "status": status,
                "prompt_file": str(prompt_file),
                "result_file": str(result_file) if not dry else "",
            }
        )
        print(f"{status}: {experiment.experiment_id}")

    summary_path = results_dir / f"{model_slug}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "game",
                "rq",
                "uses_env",
                "uses_task",
                "uses_reward",
                "uses_simplification",
                "uses_icl",
                "model",
                "status",
                "prompt_file",
                "result_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Loaded experiments file: {experiments_file}")
    print(f"Experiments processed: {len(experiments)}")
    print(f"Prompt files: {prompts_dir}")
    print(f"Summary file: {summary_path}")
    if not dry:
        print(f"Result files: {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments from a YAML experiment definition.")
    parser.add_argument(
        "--file",
        required=True,
        help="YAML experiments file path.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Build prompt files only (no API calls).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing result files (default behavior is skip).",
    )
    args = parser.parse_args()

    run(
        experiments_file=Path(args.file),
        dry=args.dry,
        overwrite_existing=args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
