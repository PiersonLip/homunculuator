"""
Extracts messages from a Discord data export.
Looks for a Messages/ directory inside ./data/ at up to 2 levels deep.
"""

import json
from pathlib import Path


def find_messages_dir(data_root: Path) -> Path:
    """Search for the Messages/ folder in the Discord export."""
    # Try common locations
    candidates = [
        data_root / "Messages",
        data_root / "package" / "Messages",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Fallback: search up to 2 levels deep
    for path in data_root.rglob("Messages"):
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not find a 'Messages' directory inside {data_root}.\n"
        "Make sure you extracted your Discord data export into the data/ folder."
    )


def main():
    data_root = Path("/app/data") if Path("/app/data").exists() else Path("data")
    output_file = Path("all_messages.txt")

    messages_dir = find_messages_dir(data_root)
    print(f"Found Messages directory: {messages_dir}")

    entries = []
    for channel_dir in messages_dir.iterdir():
        messages_file = channel_dir / "messages.json"
        if not messages_file.exists():
            continue
        try:
            with open(messages_file, encoding="utf-8") as f:
                messages = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Skipping {messages_file}: {e}")
            continue

        for msg in messages:
            content = msg.get("Contents", "").strip()
            if content:
                entries.append((msg.get("Timestamp", ""), content))

    entries.sort(key=lambda x: x[0])

    with open(output_file, "w", encoding="utf-8") as f:
        for timestamp, content in entries:
            f.write(f"[{timestamp}] {content}\n")

    print(f"Wrote {len(entries):,} messages to {output_file}")


if __name__ == "__main__":
    main()
