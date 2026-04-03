"""
Training pipeline orchestrator.
Runs inside the trainer Docker container.

Steps:
  1. Extract Discord messages from data/
  2. Clean messages
  3. Prepare JSONL dataset
  4. Fine-tune the model
  5. Register the GGUF with Ollama
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml


def load_config() -> dict:
    config_path = Path("/app/config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_step(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)
    return result


def wait_for_ollama(host: str, timeout: int = 120):
    print(f"\nWaiting for Ollama at {host}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/api/tags", timeout=3)
            if r.status_code == 200:
                print("Ollama is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"Ollama did not become ready within {timeout}s")


def find_gguf(models_dir: Path) -> Path:
    """Find the GGUF file produced by finetune.py."""
    gguf_dir = models_dir / "gguf"
    files = list(gguf_dir.glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"No .gguf file found in {gguf_dir}")
    # Prefer single-shard files; take the first one sorted
    files.sort()
    return files[0]


def register_with_ollama(host: str, model_name: str, gguf_path: Path, system_prompt: str):
    """Create the Ollama model from the GGUF file."""
    # Ollama sees the model at /models/gguf/<filename> (shared volume mount)
    ollama_gguf_path = f"/models/gguf/{gguf_path.name}"

    modelfile = (
        f"FROM {ollama_gguf_path}\n"
        f'SYSTEM "{system_prompt}"\n'
    )

    print(f"\nRegistering model '{model_name}' with Ollama...")
    print(f"  GGUF: {ollama_gguf_path}")

    response = requests.post(
        f"{host}/api/create",
        json={"model": model_name, "modelfile": modelfile},
        stream=True,
        timeout=600,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                status = data.get("status", "")
                if status:
                    print(f"  {status}")
            except json.JSONDecodeError:
                pass

    print(f"\nModel '{model_name}' is ready in Ollama.")


def main():
    config = load_config()

    persona_name = config["persona"]["name"]
    model_name = persona_name.lower()
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    system_prompt = (
        f"You are {persona_name}. Respond to messages exactly as {persona_name} would — "
        f"matching their tone, vocabulary, humor, and style from their Discord conversations."
    )

    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)
    (models_dir / "gguf").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  Homunculator Training Pipeline")
    print("="*60)
    print(f"  Persona:  {persona_name}")
    print(f"  Model:    {model_name}")
    print(f"  Ollama:   {ollama_host}")

    run_step("Step 1/4 — Extracting Discord messages", "extract_messages.py")
    run_step("Step 2/4 — Cleaning messages", "clean_messages.py")
    run_step("Step 3/4 — Preparing dataset", "prepare_dataset.py")
    run_step("Step 4/4 — Fine-tuning model (this will take a while...)", "finetune.py")

    gguf_path = find_gguf(models_dir)
    print(f"\nGGUF model: {gguf_path}")

    wait_for_ollama(ollama_host)
    register_with_ollama(ollama_host, model_name, gguf_path, system_prompt)

    print("\n" + "="*60)
    print("  Training complete!")
    print(f"  Run start.bat (or start.sh) to launch the Discord bot.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
