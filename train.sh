#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Homunculator - Training Pipeline"
echo "============================================================"
echo ""
echo "Training will take 30-90 minutes with a GPU."
echo ""

docker compose --profile train up --build

echo ""
echo "============================================================"
echo " Done! Run ./start.sh to launch your Discord bot."
echo "============================================================"
