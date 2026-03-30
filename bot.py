"""
Discord bot that responds as Pierson via the fine-tuned Ollama model.

Commands:
  !join        — bot joins your voice channel
  !ask         — bot listens for RECORD_SECONDS, transcribes, and responds via TTS
  !ask <text>  — skip STT, respond to <text> directly via TTS
  !leave       — bot leaves
  @mention     — text-only reply (no voice needed)
"""

import os
import asyncio
import tempfile
from collections import defaultdict
import discord
from discord.sinks import WaveSink
import edge_tts
from faster_whisper import WhisperModel
from chat import respond

DISCORD_TOKEN  = ''
LISTEN_CHANNEL_ID = None
HISTORY_LIMIT  = 10
TTS_VOICE      = "en-US-GuyNeural"
RECORD_SECONDS = 6   # how long !ask listens before processing

whisper_model = WhisperModel("small", device="cuda", compute_type="float16")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

history: dict[int, list[dict]] = defaultdict(list)


class SilenceSource(discord.AudioSource):
    """Plays silent audio to keep the voice connection alive."""
    def read(self) -> bytes:
        return b"\x00" * 3840  # 20ms of silence at 48kHz stereo 16-bit
    def is_opus(self) -> bool:
        return False


def start_silence(vc: discord.VoiceClient):
    if vc.is_connected() and not vc.is_playing():
        vc.play(SilenceSource())


def trim_history(channel_id: int):
    h = history[channel_id]
    if len(h) > HISTORY_LIMIT * 2:
        history[channel_id] = h[-(HISTORY_LIMIT * 2):]


async def speak(vc: discord.VoiceClient, text: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    await edge_tts.Communicate(text, TTS_VOICE).save(tmp_path)

    vc.stop()  # stop silence (or anything else playing)

    done: asyncio.Event = asyncio.Event()
    def after(err):
        if err:
            print(f"TTS playback error: {err}")
        done.set()

    vc.play(discord.FFmpegPCMAudio(tmp_path), after=after)
    await done.wait()
    os.unlink(tmp_path)

    start_silence(vc)  # resume keep-alive


def transcribe_sink(audio_data: dict) -> str:
    best_audio = None
    best_size  = 0
    for audio in audio_data.values():
        audio.file.seek(0, 2)
        size = audio.file.tell()
        audio.file.seek(0)
        if size > best_size:
            best_size  = size
            best_audio = audio

    if best_audio is None or best_size < 4000:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(best_audio.file.read())
        tmp_path = f.name
    try:
        segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text for seg in segments if seg.no_speech_prob < 0.6).strip()
    finally:
        os.unlink(tmp_path)


async def ask_and_respond(vc: discord.VoiceClient, channel: discord.TextChannel, text: str | None = None):
    """Core logic: get text (via STT or directly), generate reply, speak it."""
    channel_id = channel.id

    if text is None:
        # Record voice
        await channel.send(f"*(listening for {RECORD_SECONDS}s...)*")

        done: asyncio.Event = asyncio.Event()
        captured: list      = [None]

        async def on_done(sink, *args, _d=done, _c=captured):
            _c[0] = sink
            _d.set()

        vc.start_recording(WaveSink(), on_done)
        await asyncio.sleep(RECORD_SECONDS)
        vc.stop_recording()

        try:
            await asyncio.wait_for(done.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            await channel.send("*(recording timed out)*")
            return

        sink = captured[0]
        if not sink or not sink.audio_data:
            await channel.send("*(no audio captured)*")
            return

        text = await asyncio.to_thread(transcribe_sink, sink.audio_data)
        if not text:
            await channel.send("*(couldn't make out what was said)*")
            return

        await channel.send(f"*(heard: {text})*")

    print(f"Prompt: {text}")
    trim_history(channel_id)
    reply = await asyncio.to_thread(respond, text, history[channel_id])
    history[channel_id].append({"role": "user",      "content": text})
    history[channel_id].append({"role": "assistant", "content": reply})

    await channel.send(reply)
    if vc.is_connected():
        await speak(vc, reply)


@client.event
async def on_ready():
    assert client.user
    print(f"Logged in as {client.user} (id: {client.user.id})")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    content = message.content.strip()

    # !join
    if content == "!join":
        if not message.author.voice:  # type: ignore[union-attr]
            await message.reply("you're not in a voice channel")
            return
        assert message.guild
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect(force=True)
            await asyncio.sleep(0.5)
        vc = await message.author.voice.channel.connect()  # type: ignore[union-attr]
        start_silence(vc)
        await message.reply("joined")
        return

    # !leave
    if content == "!leave":
        if message.guild and message.guild.voice_client:
            await message.guild.voice_client.disconnect(force=True)
        return

    # !ask [optional text]
    if content.startswith("!ask"):
        assert message.guild
        vc = message.guild.voice_client
        if not isinstance(vc, discord.VoiceClient) or not vc.is_connected():
            await message.reply("not in a voice channel — use !join first")
            return
        inline_text = content[4:].strip() or None
        asyncio.create_task(ask_and_respond(vc, message.channel, inline_text))  # type: ignore[arg-type]
        return

    # @mention → text reply
    assert client.user
    if client.user not in message.mentions:
        if not (LISTEN_CHANNEL_ID and message.channel.id == LISTEN_CHANNEL_ID):
            return

    text = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not text:
        return

    async with message.channel.typing():
        channel_id = message.channel.id
        trim_history(channel_id)
        reply = await asyncio.to_thread(respond, text, history[channel_id])
        history[channel_id].append({"role": "user",      "content": text})
        history[channel_id].append({"role": "assistant", "content": reply})

    await message.reply(reply)


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
