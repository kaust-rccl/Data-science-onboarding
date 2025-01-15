#!/bin/bash

# Define variables
OLLAMA_URL="https://ollama.com/download/ollama-linux-amd64.tgz"
OLLAMA_TAR="ollama-linux-amd64.tgz"
OLLAMA_BINARY="$(pwd)/bin/ollama"


# Check if Ollama binary exists
if [ ! -f "$OLLAMA_BINARY" ]; then
    echo "Ollama binary not found. Downloading..."
    curl -L $OLLAMA_URL -o $OLLAMA_TAR

    echo "Extracting Ollama binary..."
    tar -xzf $OLLAMA_TAR

    echo "Cleaning up..."
    rm $OLLAMA_TAR
fi

# # Run Ollama binary in the background
echo "Running Ollama binary in the background..."
nohup $OLLAMA_BINARY serve &

# # Verify that Ollama is running
echo "Verifying Ollama is running..."
$OLLAMA_BINARY -v

# # download qwnen2.5
echo "Downloading qwnen2.5..."
$OLLAMA_BINARY pull qwen2.5-coder:1.5b

# # download pull llama3.1
$OLLAMA_BINARY pull llama3.1:8b
$OLLAMA_BINARY pull llama3.2:3b


echo "Setup complete."
