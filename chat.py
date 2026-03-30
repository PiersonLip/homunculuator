"""
Interface to the fine-tuned Pierson model via Ollama.
Can be used as a library, CLI, or later as a Discord bot backend.

Usage:
    python chat.py "hey what do you think about this"
    python chat.py  # interactive mode
"""

import sys
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "pierson"

SYSTEM_PROMPT = (
    "You are Pierson. Respond to messages exactly as Pierson would — "
    "matching his tone, vocabulary, humor, and style from his Discord conversations."
)


def respond(user_message: str, history: list[dict] | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["message"]["content"]


def main():
    if len(sys.argv) > 1:
        # Single-shot CLI: python chat.py "your message"
        print(respond(" ".join(sys.argv[1:])))
    else:
        # Interactive loop
        history = []
        print(f"Chatting with {MODEL_NAME}. Ctrl+C to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            reply = respond(user_input, history)
            print(f"Pierson: {reply}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
