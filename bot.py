"""
Discord bot that responds as the trained persona.

Usage:
  @mention the bot in any channel to get a response.
  It maintains per-channel conversation history.

Config: edit config.yaml before running.
"""

import asyncio
from collections import defaultdict
from pathlib import Path

import discord
import yaml
from chat import respond


def load_config() -> dict:
    for p in [Path("/app/config.yaml"), Path("config.yaml")]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


config = load_config()

DISCORD_TOKEN = config.get("discord", {}).get("token", "")
PERSONA_NAME = config.get("persona", {}).get("name", "Pierson")
HISTORY_LIMIT = 10  # conversation turns remembered per channel

if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
    raise SystemExit(
        "ERROR: Set your Discord bot token in config.yaml under discord.token"
    )

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

history: dict[int, list[dict]] = defaultdict(list)


def trim_history(channel_id: int):
    h = history[channel_id]
    if len(h) > HISTORY_LIMIT * 2:
        history[channel_id] = h[-(HISTORY_LIMIT * 2):]


@client.event
async def on_ready():
    assert client.user
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print(f"Persona: {PERSONA_NAME}")
    print("Mention the bot in any channel to chat.")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    assert client.user
    if client.user not in message.mentions:
        return

    text = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not text:
        return

    async with message.channel.typing():
        channel_id = message.channel.id
        trim_history(channel_id)
        reply = await asyncio.to_thread(respond, text, list(history[channel_id]))
        history[channel_id].append({"role": "user", "content": text})
        history[channel_id].append({"role": "assistant", "content": reply})

    await message.reply(reply)


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
