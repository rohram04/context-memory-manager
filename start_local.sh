#!/usr/bin/env bash
# Start Ollama + LiteLLM proxy for local pipeline testing.
# Usage: ./start_local.sh [model]
# Default model: ollama/llama3.2:3b
# Then in another terminal: python cli.py --local

set -e

MODEL="${1:-llama3.1:8b}"
PORT=4000

# Start ollama server if not already running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting ollama server..."
    ollama serve &
    sleep 2
fi

echo "Starting LiteLLM proxy on port $PORT → ollama/$MODEL"
echo "Run in another terminal: python cli.py --local [--local-model ollama/$MODEL]"
echo ""
litellm --model "ollama/$MODEL" --port "$PORT"
