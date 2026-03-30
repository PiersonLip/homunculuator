# PersonaBot

Fine-tune a local LLM on your Discord messages and deploy it as a bot that talks like you.

**How it works:** Your Discord data export is processed into a training dataset, used to fine-tune Llama 3.2 3B with QLoRA, exported as a GGUF model, and served via Ollama. The Discord bot responds to @mentions using that model.

Everything runs in Docker — no Python installation required.

---

## Prerequisites

### Windows

1. **NVIDIA GPU** with up-to-date drivers ([download](https://www.nvidia.com/drivers))
2. **Docker Desktop** ([download](https://www.docker.com/products/docker-desktop/))
   - During install, enable WSL 2 integration when prompted
   - After install, open Docker Desktop → Settings → Resources → WSL Integration → enable for your distro
3. **NVIDIA Container Toolkit for WSL 2**
   - Open a WSL terminal and run:
     ```bash
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```

### Linux

1. **NVIDIA GPU** with up-to-date drivers
2. **Docker** ([install guide](https://docs.docker.com/engine/install/))
3. **NVIDIA Container Toolkit**:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

## Setup

### 1. Get your Discord data export

1. Open Discord → User Settings → Privacy & Safety → **Request all of my Data**
2. Wait for the email (can take up to 30 days, usually a few hours)
3. Download and extract the zip
4. Copy the extracted folder contents into the `data/` directory in this project

Your `data/` folder should contain a `Messages/` directory (either directly or inside a `package/` subfolder — both are handled automatically).

### 2. Create a Discord bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Click **New Application** → give it a name
3. Go to **Bot** → click **Reset Token** → copy the token
4. Under **Privileged Gateway Intents**, enable **Message Content Intent**
5. Go to **OAuth2 → URL Generator** → select scopes: `bot` → permissions: `Send Messages`, `Read Message History`
6. Copy the generated URL and use it to add the bot to your server

### 3. Configure

Edit `config.yaml`:

```yaml
discord:
  token: "YOUR_DISCORD_BOT_TOKEN_HERE"  # paste token from step 2

persona:
  name: "YourName"  # your name, used in the system prompt

training:
  max_examples: 50000  # increase for more data, decrease to train faster
  epochs: 1
```

---

## Training

Training processes your Discord messages, fine-tunes the model, and registers it with Ollama. Takes **30–90 minutes** with a GPU.

**Windows** — double-click `train.bat`

**Linux / macOS:**
```bash
./train.sh
```

Or manually:
```bash
docker compose --profile train up --build
```

When training finishes, the model is automatically registered with Ollama and ready to use.

---

## Running the bot

**Windows** — double-click `start.bat`

**Linux / macOS:**
```bash
./start.sh
```

Or manually:
```bash
docker compose --profile bot up -d
docker compose --profile bot logs -f bot
```

To stop the bot:
```bash
docker compose --profile bot down
```

---

## Usage

In any Discord server where the bot has been added, **@mention** it to get a response:

```
@YourBot hey what do you think of this
```

The bot remembers the last 10 turns of conversation per channel.

---

## Project structure

```
├── config.yaml          # Edit this — Discord token, persona name, training settings
├── train.bat / train.sh # Run to train the model
├── start.bat / start.sh # Run to start the Discord bot
│
├── docker-compose.yml   # Defines trainer and bot services
├── Dockerfile.train     # CUDA + Unsloth training environment
├── Dockerfile.bot       # Lightweight bot environment
│
├── pipeline.py          # Orchestrates the full training pipeline
├── finetune.py          # QLoRA fine-tuning with Unsloth
├── extract_messages.py  # Extracts messages from Discord export
├── clean_messages.py    # Filters/cleans raw messages
├── prepare_dataset.py   # Builds JSONL training dataset
│
├── bot.py               # Discord bot
├── chat.py              # Ollama model interface
│
├── data/                # Put your Discord export here
└── models/              # Trained model output (auto-generated)
```

---

## Troubleshooting

**`docker: Error response from daemon: could not select device driver "nvidia"`**
The NVIDIA Container Toolkit is not installed or Docker wasn't restarted after installing it. Follow the prerequisites section again and make sure to run `sudo systemctl restart docker`.

**Bot is online but not responding**
Make sure **Message Content Intent** is enabled on your bot in the Discord developer portal.

**Training runs on CPU (very slow)**
Check that your GPU is visible inside Docker: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`. If this fails, revisit the NVIDIA Container Toolkit setup.

**Out of GPU memory during training**
Reduce `batch_size` and/or `max_seq_length` in the `advanced` section of `config.yaml`.
