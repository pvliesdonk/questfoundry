@echo off
rem Unset old athena URL
set OLLAMA_API_BASE=
rem Set localhost
set OLLAMA_HOST=http://localhost:11434
set QF_SPEC_SOURCE=monorepo
cd lib\runtime
echo Testing with OLLAMA_HOST=%OLLAMA_HOST%
uv run qf -v -v -v ask --provider ollama "simple two scene story"
