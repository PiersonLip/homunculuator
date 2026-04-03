#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Homunculator - Starting Discord Bot"
echo "============================================================"
echo ""

docker compose --profile bot up -d --build

echo ""
echo "Bot is online! Showing logs (Ctrl+C to stop viewing logs)."
echo "To stop the bot: docker compose --profile bot down"
echo ""
docker compose --profile bot logs -f bot
