#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

cd /home/aiscuser/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_mcore_mistral.sh  \
# Running environment: dlc, dsw
dsw  \
# Path to Megatron Patch code
../../ \
# Model size: 7B, 13B
7B   \
# Per GPU batch size: 4, 8
1    \
# Global batch size
8 \
# Learning rate: 1e-5, 5e-5
1e-5   \
# Minimum learning rate: 1e-6, 5e-6
1e-6   \
# Sequence length
128  \
# Padding length: 100
128  \
# Vocabulary expansion size
0   \
# Precision: fp16, bf16
bf16  \
# Tensor parallelism
4   \
# Pipeline parallelism
1  \
# Activation checkpointing mode: sel, full
sel  \
# Use Megatron's Zero-1 memory optimizer: true, false
true   \
# Use Flash Attention: true, false
false  \
# Use sequence parallelism: true, false
false   \
# Use Transformer Engine: true, false
false   \
# Enable MoE: true, false
false \
# Checkpoint save interval
100000  \
# Training dataset path
/mnt/mistral-datasets/wudao_mistralbpe_content_document \
# Pre-trained model path
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1   \
# Number of training tokens
100000000   \
# Number of warmup tokens
10000   \
# Output path for training
/mnt/output_mcore_mistral


# ENV=$1                          # Running environment: dlc, dsw
# MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
# MODEL_SIZE=$3                   # Model size: 7B, 13B
# BATCH_SIZE=$4                   # Per GPU batch size: 4, 8
# GLOBAL_BATCH_SIZE=$5            # Global batch size
# LR=$6                           # Learning rate: 1e-5, 5e-5
# MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
# SEQ_LEN=$8                      # Sequence length
# PAD_LEN=$9                      # Padding length: 100
# EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
# PR=${11}                        # Precision: fp16, bf16
# TP=${12}                        # Model parallelism
# PP=${13}                        # Pipeline parallelism
# AC=${14}                        # Activation checkpointing mode: sel, full
# DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
# FL=${16}                        # Use Flash Attention: true, false
# SP=${17}                        # Use sequence parallelism: true, false
# TE=${18}                        # Use Transformer Engine: true, false
# MOE=${19}                       # Enable MoE: true, false
# SAVE_INTERVAL=${20}             # Checkpoint save interval
# DATASET_PATH=${21}              # Training dataset path
# PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-trained model path
# TRAIN_TOKENS=${23}              # Number of training tokens
# WARMUP_TOKENS=${24}             # Number of warmup tokens
# OUTPUT_BASEPATH=${25}           # Output path for training