#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

model_path="/workspaces/Pai-Megatron-Patch/mnt/mistral-ckpts/Mistral-7B-v0.1-mcore-to-HF4"

output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \