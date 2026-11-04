@echo off
title P1 Gateway - port 6066
cd /d "D:\桌面\多轮对话机器人项目"

echo ==========================================
echo   P1 Gateway  /  port 6066
echo ==========================================
echo   Docs : http://127.0.0.1:6066/docs
echo   Stop : close this window
echo ==========================================
echo.

E:\anaconda3\envs\llm-dev\python.exe -m uvicorn fastapi_bot:app --port 6066

echo.
echo [stopped]
pause