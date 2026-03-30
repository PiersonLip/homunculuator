"""
Interface to the fine-tuned persona model via Ollama.
Reads model name and Ollama URL from config.yaml / environment.
"""

import os
import sys
from pathlib import Path

import requests
import yaml


def load_config() -> dict:
    for p in [Path("/app/config.yaml"), Path("config.yaml")]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


def _get_settings():
    config = load_config()
    persona_name = config.get("persona", {}).get("name", "Pierson")
    model_name = persona_name.lower()
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    system_prompt = (
        f"You are {persona_name}. Respond to messages exactly as {persona_name} would — "
        f"matching their tone, vocabulary, humor, and style from their Discord conversations."
    )
    return model_name, ollama_host, system_prompt, persona_name


def respond(user_message: str, history: list | None = None) -> str:
    model_name, ollama_host, system_prompt, _ = _get_settings()

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        f"{ollama_host}/api/chat",
        json={"model": model_name, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def main():
    _, _, _, persona_name = _get_settings()
    model_name = persona_name.lower()

    if len(sys.argv) > 1:
        print(respond(" ".join(sys.argv[1:])))
    else:
        history = []
        print(f"Chatting with {model_name}. Ctrl+C to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            reply = respond(user_input, history)
            print(f"{persona_name}: {reply}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
