"""Run LLM-policy-faithfulness experiments defined in experiments.yml.

For each row: build the prompt from a template + policy + context files, call the LLM
via OpenRouter, save the raw response. Labels are added by hand to results/labels.csv.

Precedence for model selection: --model > OPENROUTER_MODEL env > YAML `defaults.model`.
"""

import argparse
import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPTIONAL_BLOCK_PATTERN = re.compile(r"{{#([A-Z0-9_]+)}}(.*?){{/\1}}", re.DOTALL)
DRY_RUN_RESULT_TEXT = "---- This was a dry run\n"
MODEL_KEY_TO_OPENROUTER = {
    "gpt-5": "openai/gpt-5",
    "gpt-4o": "openai/gpt-4o",
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    rq: str
    policy_file: str
    env_file: str | None
    task_file: str | None
    reward_file: str | None
    simplification_file: str | None


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _read_optional_file(path: str | None) -> str:
    return _read_file(path) if path else ""


def _normalize_prompt(prompt: str) -> str:
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip() + "\n"


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def _resolve_optional(row: dict, field: str, default: str | None) -> str | None:
    if field in row:
        return _optional_string(row.get(field))
    return default


def _render_optional_blocks(template: str, replacements: dict[str, str]) -> str:
    def replace_block(match: re.Match[str]) -> str:
        key = match.group(1)
        block = match.group(2)
        return block if replacements.get(key, "").strip() else ""

    rendered = template
    while True:
        nxt = OPTIONAL_BLOCK_PATTERN.sub(replace_block, rendered)
        if nxt == rendered:
            return rendered
        rendered = nxt


def _load(path: Path) -> tuple[str, dict[str, str], list[Experiment]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data["defaults"]
    model = str(defaults["model"]).strip()
    templates = {str(k).strip().lower(): str(v).strip() for k, v in defaults["templates"].items()}
    default_env = _optional_string(defaults.get("env_file"))
    default_task = _optional_string(defaults.get("task_file"))
    default_reward = _optional_string(defaults.get("reward_file"))
    default_simp = _optional_string(defaults.get("simplification_file"))

    rows: list[Experiment] = []
    for r in data.get("experiments", []):
        rows.append(
            Experiment(
                experiment_id=str(r["id"]).strip(),
                rq=str(r["rq"]).strip().lower(),
                policy_file=str(r["policy_file"]).strip(),
                env_file=_resolve_optional(r, "env_file", default_env),
                task_file=_resolve_optional(r, "task_file", default_task),
                reward_file=_resolve_optional(r, "reward_file", default_reward),
                simplification_file=_resolve_optional(r, "simplification_file", default_simp),
            )
        )
    return model, templates, rows


def _model_dir_key(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_key).strip("_") or "model"


def _build_prompt(exp: Experiment, template_path: str) -> str:
    template = _read_file(template_path)
    replacements = {
        "ENV_DESCRIPTION": _read_optional_file(exp.env_file),
        "TASK_DESCRIPTION": _read_optional_file(exp.task_file),
        "REWARD_FUNCTION": _read_optional_file(exp.reward_file),
        "ENV_SIMPLIFICATION_DESCRIPTION": _read_optional_file(exp.simplification_file),
        "SYMBOLIC_POLICY": _read_file(exp.policy_file),
    }
    rendered = _render_optional_blocks(template, replacements)
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return _normalize_prompt(rendered)


def _call_llm(client, model: str, prompt: str) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    if not response.choices:
        dump = response.model_dump_json(indent=2) if hasattr(response, "model_dump_json") else repr(response)
        raise RuntimeError(f"LLM response did not contain choices: {dump}")
    text = response.choices[0].message.content or ""
    metadata = {
        "response_id": getattr(response, "id", None),
        "model_resolved": getattr(response, "model", None),
        "model_requested": model,
        "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    return text, metadata


def _can_overwrite(result_file: Path) -> bool:
    if not result_file.is_file():
        return True
    return result_file.read_text(encoding="utf-8").strip() == DRY_RUN_RESULT_TEXT.strip()


def run(yaml_path: Path, dry: bool, model_override: str | None = None) -> None:
    model_key, templates, experiments = _load(yaml_path)
    if env_override := _optional_string(os.getenv("OPENROUTER_MODEL")):
        model_key = env_override
    if model_override:
        model_key = model_override.strip()
    openrouter_model = MODEL_KEY_TO_OPENROUTER.get(model_key, model_key)
    model_dir = _model_dir_key(model_key)

    prompts_dir = Path("prompts/sent") / model_dir
    results_dir = Path("results") / model_dir
    prompts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not dry:
        from openai import OpenAI
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))

    for exp in experiments:
        template_path = templates[exp.rq]
        prompt = _build_prompt(exp, template_path)
        (prompts_dir / f"{exp.experiment_id}_prompt.txt").write_text(prompt, encoding="utf-8")

        result_file = results_dir / f"{exp.experiment_id}_result.txt"
        if not _can_overwrite(result_file):
            status = "skipped_existing_result"
        else:
            if dry:
                status = "dry"
                result_file.write_text(DRY_RUN_RESULT_TEXT, encoding="utf-8")
            else:
                try:
                    text, meta = _call_llm(client, openrouter_model, prompt)
                except Exception as exc:
                    status = f"failed: {type(exc).__name__}: {exc}"
                else:
                    status = "done"
                    result_file.write_text(text, encoding="utf-8")
                    (results_dir / f"{exp.experiment_id}_meta.json").write_text(
                        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                    )
        print(f"{status}: {exp.experiment_id}")

    print(f"Loaded: {yaml_path}  rows: {len(experiments)}  prompts: {prompts_dir}  results: {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-policy-faithfulness experiments.")
    parser.add_argument("--file", default="experiments.yml", help="YAML experiment file (default: experiments.yml).")
    parser.add_argument("--model", help="OpenRouter model override (else OPENROUTER_MODEL env, else YAML default).")
    parser.add_argument("--dry", action="store_true", help="Build prompts only, no API calls.")
    args = parser.parse_args()
    run(Path(args.file), dry=args.dry, model_override=args.model)


if __name__ == "__main__":
    main()
