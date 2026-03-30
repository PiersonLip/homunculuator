"""
Fine-tunes Llama 3.1 8B on Pierson's Discord messages using Unsloth + QLoRA.
Outputs a merged model in GGUF format ready for Ollama.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import torch

# Reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
OUTPUT_DIR   = "lora_model"
GGUF_DIR     = "gguf_model"

MAX_SEQ_LEN  = 512    # 3B fits fine; chat template adds ~80 token overhead
LORA_RANK    = 16
BATCH_SIZE   = 4
GRAD_ACCUM   = 4      # effective batch = 16
EPOCHS       = 1      # 1 pass over 50k examples is plenty for style learning
MAX_SAMPLES  = 50_000 # cap training set; 280k × 3 epochs would take ~48hrs
LR           = 2e-4

# ── Load model ────────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,           # auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,        # 0 lets Unsloth use fast kernel paths; saves VRAM
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

# Filter out examples that exceed MAX_SEQ_LEN after tokenization.
# Unsloth's fused loss crashes on truncated sequences, so drop them instead.
def within_length(example):
    length = tokenizer(example["text"], return_length=True)["length"][0]
    return length <= MAX_SEQ_LEN

dataset = dataset.filter(within_length, num_proc=8)

# Cap training set for speed
if len(dataset["train"]) > MAX_SAMPLES:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(MAX_SAMPLES))

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05,
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
trainer.save_model(OUTPUT_DIR)

# ── Export to GGUF ────────────────────────────────────────────────────────────

print("\nSaving merged GGUF model (Q4_K_M)...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",  # good balance of size/quality
)
print(f"GGUF saved to: {GGUF_DIR}/")
print("\nNext step — create the Ollama model:")
print(f"  ollama create pierson -f {GGUF_DIR}/Modelfile")
