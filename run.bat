@echo off
rem 用 llm-rag 环境的 python 跑脚本。用法：run.bat tools\quick_eval.py 10
rem 为什么需要：系统默认的 python 是 base 环境，没装 jieba/chromadb 那些。
set PY=E:\anaconda3\envs\llm-rag\python.exe
cd /d "%~dp0"
"%PY%" %*
