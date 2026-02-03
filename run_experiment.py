import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_CSV = "experiment_tracker.csv"
RESULTS_DIR = "04_results"
PROMPTS_DIR = "03_prompts/sent"
PROMPT_TEMPLATE = "03_prompts/template.txt"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

os.makedirs(PROMPTS_DIR, exist_ok=True)


def call_llm(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def run_experiment(dry: bool = False):
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    template = Path(PROMPT_TEMPLATE).read_text()
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    for idx, row in df.iterrows():
        status = str(row["status"]).strip().lower()
        if status == "done":
            continue

        row_id = str(row["id"]).strip()
        model = str(row["target_llm"]).strip()
        policy_file = str(row["policy_file"]).strip()
        context_path = row["context_file"]

        if pd.isna(context_path) or str(context_path).strip() == "":
            context = ""
        else:
            context_value = str(context_path).strip()
            resolved_context = Path(context_value)
            if not resolved_context.is_file():
                resolved_context = Path("02_contexts") / context_value
            if not resolved_context.is_file():
                resolved_context = Path("02_contexts") / str(row["game"]).strip() / context_value
            if not resolved_context.is_file():
                raise FileNotFoundError(
                    f"Context file not found: '{context_value}'. "
                    "Tried raw path, 02_contexts/<file>, and 02_contexts/<game>/<file>."
                )
            context = resolved_context.read_text()
        policy_path = Path("01_policies") / str(row["policy_type"]).strip() / str(row["game"]).strip() / policy_file
        policy = policy_path.read_text()

        prompt = template.replace("{{CONTEXT}}", context).replace("{{POLICY}}", policy)

        prompt_file = f"{row_id}_{model.replace('/', '_')}_prompt.txt"
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        Path(prompt_path).write_text(prompt)
        df.at[idx, "prompt_file"] = prompt_path

        if dry:
            continue

        result = call_llm(client, model, prompt)
        result_file = f"{row_id}_{model.replace('/', '_')}_result.txt"
        result_path = os.path.join(RESULTS_DIR, result_file)
        Path(result_path).write_text(result)
        df.at[idx, "result_file"] = result_path
        df.at[idx, "status"] = "done"

    df.to_csv(INPUT_CSV, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Skip LLM calls.",
    )
    args = parser.parse_args()
    run_experiment(dry=args.dry)
