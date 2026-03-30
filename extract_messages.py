import json
import os
from pathlib import Path

MESSAGES_DIR = Path("package/Messages")
OUTPUT_FILE = Path("all_messages.txt")

entries = []

for channel_dir in MESSAGES_DIR.iterdir():
    messages_file = channel_dir / "messages.json"
    if not messages_file.exists():
        continue

    try:
        with open(messages_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Skipping {messages_file}: {e}")
        continue

    for msg in messages:
        content = msg.get("Contents", "").strip()
        if content:
            entries.append((msg.get("Timestamp", ""), content))

# Sort chronologically
entries.sort(key=lambda x: x[0])

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for timestamp, content in entries:
        f.write(f"[{timestamp}] {content}\n")

print(f"Wrote {len(entries)} messages to {OUTPUT_FILE}")
