#!/bin/bash
pip install vllm pandas huggingface_hub
echo "Dependencies installed. Now running login..."
huggingface-cli login