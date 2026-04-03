@echo off
echo ============================================================
echo  Homunculator - Starting Discord Bot
echo ============================================================
echo.
echo Requirements:
echo   - Docker Desktop must be running
echo   - Training must be complete (run train.bat first)
echo   - config.yaml must have your Discord bot token
echo.

docker compose --profile bot up -d --build

echo.
echo Bot is online! Showing logs (Ctrl+C to stop viewing logs,
echo the bot will keep running in the background).
echo To stop the bot: docker compose --profile bot down
echo.
docker compose --profile bot logs -f bot
