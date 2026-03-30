import re
from pathlib import Path

INPUT_FILE = Path("all_messages.txt")
OUTPUT_FILE = Path("cleaned_messages.txt")

DISCORD_EMOJI = re.compile(r"<a?:[a-zA-Z0-9_]+:\d+>")
URL = re.compile(r"https?://\S+")
UNICODE_EMOJI_ONLY = re.compile(r"^[\s\U0001F000-\U0001FFFF\u2000-\u206F\u2700-\u27BF]+$")
TIMESTAMP = re.compile(r"^\[.+?\] ")

MIN_LENGTH = 3

kept = skipped = 0

with open(INPUT_FILE, encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        content = TIMESTAMP.sub("", line).strip()
        content = DISCORD_EMOJI.sub("", content)
        content = URL.sub("", content)
        content = re.sub(r"\s+", " ", content).strip()

        if len(content) < MIN_LENGTH:
            skipped += 1
            continue

        if UNICODE_EMOJI_ONLY.match(content):
            skipped += 1
            continue

        fout.write(content + "\n")
        kept += 1

print(f"Kept:    {kept:,}")
print(f"Skipped: {skipped:,}")
print(f"Output:  {OUTPUT_FILE}")
