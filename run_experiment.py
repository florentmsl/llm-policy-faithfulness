import os

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


def call_llm(model: str, prompt: str) -> str:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def run_experiment(dry: bool = False):
    df = pd.read_csv(INPUT_CSV)

    with open(PROMPT_TEMPLATE) as f:
        template = f.read()

    for idx, row in df.iterrows():
        if str(row["status"]).strip() == "done":
            continue

        row_id = str(row["id"]).strip()
        model = row["target_llm"].strip()
        print(f"Running {row_id} on {model}...")

        context_file = str(row["context_file"]).strip()
        policy_type = str(row["policy_type"]).strip()
        policy_file = str(row["policy_file"]).strip()
        policy_path = os.path.join("01_policies", policy_type, policy_file)

        if context_file and context_file.lower() != "nan":
            context_path = os.path.join("02_contexts", context_file)
            with open(context_path) as f:
                context = f.read()
        else:
            context = ""
        with open(policy_path) as f:
            policy = f.read()

        prompt = template.replace("{{CONTEXT}}", context).replace("{{POLICY}}", policy)

        # Save prompt
        prompt_file = f"{row_id}_{model.replace('/', '_')}_prompt.txt"
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        with open(prompt_path, "w") as f:
            f.write(prompt)
        df.at[idx, "prompt_file"] = prompt_path

        if dry:
            print("Dry run: skipping LLM call.")
            continue

        try:
            result = call_llm(model, prompt)
            result_file = f"{row_id}_{model.replace('/', '_')}_result.txt"
            result_path = os.path.join(RESULTS_DIR, result_file)
            with open(result_path, "w") as f:
                f.write(result)
            df.at[idx, "result_file"] = result_path
            df.at[idx, "status"] = "done"
        except Exception as e:
            print(f"Failed: {e}")

    df.to_csv(INPUT_CSV, index=False)
    print("Batch complete.")


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
