@echo off
echo ============================================================
echo  Homunculator - Training Pipeline
echo ============================================================
echo.
echo Requirements:
echo   - Docker Desktop must be running
echo   - Your Discord data export must be in the data\ folder
echo   - config.yaml must have your persona name filled in
echo.
echo Training will take 30-90 minutes with a GPU.
echo.
pause

docker compose --profile train up --build
echo.
echo ============================================================
echo  Done! Run start.bat to launch your Discord bot.
echo ============================================================
pause
