"""
Fine-tunes a base model on Discord messages using Unsloth + QLoRA.
Reads all settings from config.yaml.
Outputs a merged GGUF model to /app/models/gguf/ (or ./models/gguf/).
"""

import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


def load_config() -> dict:
    for p in [Path("/app/config.yaml"), Path("config.yaml")]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


config = load_config()

training = config.get("training", {})
advanced = config.get("advanced", {})

MODEL_NAME = training.get("base_model", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_SAMPLES = training.get("max_examples", 50_000)
EPOCHS = training.get("epochs", 1)

MAX_SEQ_LEN = advanced.get("max_seq_length", 512)
LORA_RANK = advanced.get("lora_rank", 16)
BATCH_SIZE = advanced.get("batch_size", 4)
GRAD_ACCUM = advanced.get("gradient_accumulation", 4)
LR = advanced.get("learning_rate", 2e-4)

# Output directories
models_base = Path("/app/models") if Path("/app").exists() else Path("models")
LORA_DIR = str(models_base / "lora")
GGUF_DIR = str(models_base / "gguf")

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(GGUF_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = load_dataset("json", data_files={
    "train": "dataset_train.jsonl",
    "validation": "dataset_eval.jsonl",
})


def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = dataset.map(format_example)


def within_length(example):
    length = tokenizer(example["text"], return_length=True)["length"][0]
    return length <= MAX_SEQ_LEN


dataset = dataset.filter(within_length, num_proc=4)

if len(dataset["train"]) > MAX_SAMPLES:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(MAX_SAMPLES))

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=LORA_DIR,
        dataset_text_field="text",
        eos_token=None,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=100,
        learning_rate=LR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    ),
)

trainer.train()
trainer.save_model(LORA_DIR)

# ── Export to GGUF ────────────────────────────────────────────────────────────

print(f"\nExporting GGUF model (Q4_K_M) to {GGUF_DIR}...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"GGUF saved to: {GGUF_DIR}/")
