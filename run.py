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


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    game: str
    rq: str
    include_reward: bool
    policy_file: str
    env_file: str
    task_file: str
    reward_file: str
    icl_file: str


def _read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _load_experiment_file(path: Path) -> tuple[str, dict[str, str], str, list[Experiment]]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}; expected a mapping.")

    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError(f"{path} is missing a 'defaults' mapping.")

    model = defaults.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{path} defaults.model must be a non-empty string.")

    game = defaults.get("game")
    if not isinstance(game, str) or not game.strip():
        raise ValueError(f"{path} defaults.game must be a non-empty string.")

    icl_file = defaults.get("icl_file")
    if not isinstance(icl_file, str) or not icl_file.strip():
        raise ValueError(f"{path} defaults.icl_file must be a non-empty string.")

    templates = defaults.get("templates")
    if not isinstance(templates, dict):
        raise ValueError(f"{path} defaults.templates must be a mapping.")

    template_by_rq: dict[str, str] = {}
    for rq, template_path in templates.items():
        if not isinstance(rq, str) or not rq.strip():
            raise ValueError(f"{path} contains an empty template key.")
        if not isinstance(template_path, str) or not template_path.strip():
            raise ValueError(f"{path} template path for rq '{rq}' must be non-empty.")
        template_by_rq[rq.strip().lower()] = template_path.strip()

    experiment_rows = data.get("experiments")
    if not isinstance(experiment_rows, list):
        raise ValueError(f"{path} is missing an 'experiments' list.")

    experiments: list[Experiment] = []
    for row in experiment_rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a non-mapping experiment entry: {row!r}")

        experiment_id = row.get("id")
        rq = row.get("rq")
        include_reward = row.get("include_reward")
        policy_file = row.get("policy_file")
        env_file = row.get("env_file")
        task_file = row.get("task_file")
        reward_file = row.get("reward_file")

        required_strings = {
            "id": experiment_id,
            "rq": rq,
            "policy_file": policy_file,
            "env_file": env_file,
            "task_file": task_file,
            "reward_file": reward_file,
        }
        for field_name, field_value in required_strings.items():
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{path} experiment field '{field_name}' must be a non-empty string.")

        if not isinstance(include_reward, bool):
            raise ValueError(f"{path} experiment '{experiment_id}' has non-boolean include_reward.")

        experiments.append(
            Experiment(
                experiment_id=experiment_id.strip(),
                game=game.strip(),
                rq=rq.strip().lower(),
                include_reward=include_reward,
                policy_file=policy_file.strip(),
                env_file=env_file.strip(),
                task_file=task_file.strip(),
                reward_file=reward_file.strip(),
                icl_file=icl_file.strip(),
            )
        )

    return model.strip(), template_by_rq, game.strip(), experiments


def _build_prompt(experiment: Experiment, template_path: str) -> str:
    template = _read_file(template_path)
    env_text = _read_file(experiment.env_file)
    task_text = _read_file(experiment.task_file)
    policy_text = _read_file(experiment.policy_file)
    icl_text = _read_file(experiment.icl_file)

    if experiment.include_reward:
        reward_text = _read_file(experiment.reward_file)
    else:
        reward_text = "(Intentionally omitted in this condition.)"

    replacements = {
        "{{ENV_DESCRIPTION}}": env_text,
        "{{TASK_NATURAL_LANGUAGE_DESCRIPTION}}": task_text,
        "{{PYTHON_JAXATARI_REWARD_FUNCTION}}": reward_text,
        "{{SYMBOLIC_POLICY}}": policy_text,
        "{{IN_CONTEXT_LEARNING_EXAMPLE}}": icl_text,
    }

    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt.strip() + "\n"


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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is missing. Set it in your .env file.")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    summary_rows: list[dict[str, str]] = []
    for experiment in experiments:
        template_path = template_by_rq.get(experiment.rq)
        if template_path is None:
            supported = ", ".join(sorted(template_by_rq))
            raise ValueError(f"Experiment '{experiment.experiment_id}' has unsupported rq='{experiment.rq}'. Supported: {supported}")

        prompt = _build_prompt(experiment, template_path=template_path)
        prompt_file = prompts_dir / f"{experiment.experiment_id}_{model_slug}_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        result_file = results_dir / f"{experiment.experiment_id}_{model_slug}_result.txt"
        response_text = ""
        status = "dry"
        if not dry:
            if result_file.is_file() and not overwrite_existing:
                status = "skipped_existing"
                response_text = _read_file(str(result_file))
            else:
                status = "done"
                response_text = _call_llm(client, model, prompt)
                result_file.write_text(response_text, encoding="utf-8")

        summary_rows.append(
            {
                "id": experiment.experiment_id,
                "game": experiment.game,
                "rq": experiment.rq,
                "include_reward": "true" if experiment.include_reward else "false",
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
                "include_reward",
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
