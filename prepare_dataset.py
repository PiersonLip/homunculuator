"""
Converts cleaned_messages.txt into a JSONL dataset for fine-tuning.
Reads persona name from config.yaml.
"""

import json
import random
from pathlib import Path

import yaml


def load_config() -> dict:
    for p in [Path("/app/config.yaml"), Path("config.yaml")]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


config = load_config()
persona_name = config.get("persona", {}).get("name", "Pierson")

INPUT_FILE = Path("cleaned_messages.txt")
OUTPUT_TRAIN = Path("dataset_train.jsonl")
OUTPUT_EVAL = Path("dataset_eval.jsonl")

SYSTEM_PROMPT = (
    f"You are {persona_name}. Respond to messages exactly as {persona_name} would — "
    f"matching their tone, vocabulary, humor, and style from their Discord conversations."
)

EVAL_SPLIT = 0.02
MIN_MSG_LEN = 8

with open(INPUT_FILE, encoding="utf-8") as f:
    messages = [line.strip() for line in f if len(line.strip()) >= MIN_MSG_LEN]

random.seed(42)
random.shuffle(messages)

split = int(len(messages) * (1 - EVAL_SPLIT))
train_msgs = messages[:split]
eval_msgs = messages[split:]


def make_example(msg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": msg},
        ]
    }


def write_jsonl(path: Path, msgs: list):
    with open(path, "w", encoding="utf-8") as f:
        for msg in msgs:
            f.write(json.dumps(make_example(msg)) + "\n")


write_jsonl(OUTPUT_TRAIN, train_msgs)
write_jsonl(OUTPUT_EVAL, eval_msgs)

print(f"Train: {len(train_msgs):,} examples → {OUTPUT_TRAIN}")
print(f"Eval:  {len(eval_msgs):,} examples  → {OUTPUT_EVAL}")
