import re
from pathlib import Path

INPUT_FILE = Path("all_messages.txt")
OUTPUT_FILE = Path("cleaned_messages.txt")

# Patterns to remove or clean
DISCORD_EMOJI = re.compile(r"<a?:[a-zA-Z0-9_]+:\d+>")   # <:name:id> and <a:name:id>
URL = re.compile(r"https?://\S+")
UNICODE_EMOJI_ONLY = re.compile(r"^[\s\U0001F000-\U0001FFFF\u2000-\u206F\u2700-\u27BF]+$")
TIMESTAMP = re.compile(r"^\[.+?\] ")  # strip leading [timestamp]

MIN_LENGTH = 3  # skip very short messages after cleaning

kept = 0
skipped = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        # Strip timestamp
        content = TIMESTAMP.sub("", line).strip()

        # Remove discord custom emoji codes
        content = DISCORD_EMOJI.sub("", content)

        # Remove URLs
        content = URL.sub("", content)

        # Clean up extra whitespace
        content = re.sub(r"\s+", " ", content).strip()

        # Skip if too short after cleaning
        if len(content) < MIN_LENGTH:
            skipped += 1
            continue

        # Skip if it's only unicode emoji / punctuation
        if UNICODE_EMOJI_ONLY.match(content):
            skipped += 1
            continue

        fout.write(content + "\n")
        kept += 1

print(f"Kept:    {kept:,}")
print(f"Skipped: {skipped:,}")
print(f"Output:  {OUTPUT_FILE}")
