#!/bin/bash

# Create cache directory in the working directory
mkdir -p $PWD/cache/gpt4all

# List of model URLs
urls=(
    "http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
    "https://gpt4all.io/models/ggml-gpt4all-j-v1.2-jazzy.bin"
    "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    "https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf"
    "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf"
    "https://gpt4all.io/models/gguf/gpt4all-falcon-q4_0.gguf"
    "https://gpt4all.io/models/gguf/wizardlm-13b-v1.2.Q4_0.gguf"
    "https://gpt4all.io/models/gguf/nous-hermes-llama2-13b.Q4_0.gguf"
    "https://gpt4all.io/models/gguf/gpt4all-13b-snoozy-q4_0.gguf"
    "https://gpt4all.io/models/gguf/mpt-7b-chat-merges-q4_0.gguf"
    "https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf"
    "https://gpt4all.io/models/gguf/starcoder-q4_0.gguf"
    "https://gpt4all.io/models/gguf/rift-coder-v0-7b-q4_0.gguf"
    "https://gpt4all.io/models/gguf/all-MiniLM-L6-v2-f16.gguf"
)

# Download each model if they are not in cache folder
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    echo "Downloading $filename"
    if [ ! -f "$PWD/cache/gpt4all/$filename" ]; then
        curl -LO --output-dir $PWD/cache/gpt4all "$url"
    else
        echo "$filename already exists in cache, skipping download."
    fi
done