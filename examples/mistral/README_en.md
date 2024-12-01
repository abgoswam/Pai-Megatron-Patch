# Table of Contents
   * [Installation](#Installation)
   * [Dataset & Model Download](#Dataset-and-Model-Download)
   * [Megatron-LM-Dense Model Training Process](#Megatron-LM-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-LM-Dense-Model-Format-Conversion)
      * [Continued Pre-training](#Megatron-LM-Dense-Continued-Pre-training)
      * [Instruction Fine-tuning](#Megatron-LM-Dense-Instruction-Fine-tuning)
   * [Megatron-Core-Dense Model Training Process](#Megatron-Core-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-Core-Dense-Model-Format-Conversion)
      * [Continued Pre-training](#Megatron-Core-Dense-Continued-Pre-training)
      * [Instruction Fine-tuning](#Megatron-Core-Dense-Instruction-Fine-tuning)
   * [Megatron-Core-MoE Model Training Process](#Megatron-Core-MoE-Model-Training-Process)
      * [Model Format Conversion](#Megatron-Core-MoE-Model-Format-Conversion)
      * [Continued Pre-training](#Megatron-Core-MoE-Continued-Pre-training)
      * [Instruction Fine-tuning](#Megatron-Core-MoE-Instruction-Fine-tuning)
   * [Downstream Task Evaluation](#Downstream-Task-Evaluation)
      * [Megatron-LM Model Format Conversion](#Megatron-LM-Dense-Model-to-Huggingface-Format-Conversion)
      * [Running Evaluation Tools](#Running-Evaluation-Tools)

# Installation
It is recommended to use the official NVIDIA image `nvcr.io/nvidia/pytorch:23.12-py3` to create the container.

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# Dataset and Model Download
```bash
cd /mnt
mkdir mistral-ckpts
cd mistral-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-ckpts/Mistral-7B-v0.1.tgz
tar -zxf Mistral-7B-v0.1.tgz

mkdir mistral-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-valid.json
```

# Megatron-LM-Dense Model Training Process (Legacy)

Run the `hf2megatron_convertor.sh` script with the following parameters:
```
MEGATRON_PATH=$1                   # Path to Megatron-LM
SOURCE_CKPT_PATH=$2                # Path to the original checkpoint
TARGET_CKPT_PATH=$3                # Path to the target checkpoint
TP=$4                              # Tensor parallelism
PP=$5                              # Pipeline parallelism
MN=$6                              # Mistral-7B
EXTRA_VOCAB_SIZE=$7                # Extra vocabulary size
mg2hf=$8                           # Whether to perform mg2hf conversion
```

Run the `run_pretrain_megatron_mistral.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to the Megatron Patch code
MODEL_SIZE=$3                   # Model parameter size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Checkpoint save interval
DATASET_PATH=${20}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${21}  # Pre-training model path
TRAIN_TOKENS=${22}              # Number of training tokens
WARMUP_TOKENS=${23}             # Number of warm-up tokens
OUTPUT_BASEPATH=${24}           # Output file path for training
```

Run the `run_finetune_megatron_mistral_withGA.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to the Megatron Patch code
MODEL_SIZE=$3                   # Model parameter size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Checkpoint save interval
DATASET_PATH=${20}              # Training dataset path
VALID_DATASET_PATH=${21}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-training model path
TRAIN_ITERS=${23}               # Number of training steps
WARMUP_ITERS=${24}              # Number of warm-up steps
OUTPUT_BASEPATH=${25}           # Output file path for training
```

## Megatron-LM-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1    \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
4  \
1  \
mistral-7b \
0 \
false
```

## Megatron-LM-Dense Continue Pre-training
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_megatron_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

## Megatron-LM-Dense Instruction Fine-tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_megatron_mistral_withGA.sh  \
dsw  \
../../ \
7B     \
1      \
8      \
1e-5   \
1e-6   \
128   \
128     \
0      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1   \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

# Megatron-Core-Dense Model Training Workflow (MCore)

Run the `hf2mcore_convertor.sh` script with the following parameters:
```
MODEL_SIZE=$1                  # Model size: 7B/8x7B
HG_CKPT_PATH=$2                # HF checkpoint path
MEGATRON_PATH=$3               # Megatron-LM root directory
SOURCE_CKPT_PATH=$4            # Source path
TARGET_CKPT_PATH=$5            # Target path
TP=$6                          # Model parallelism
PP=$7                          # Pipeline parallelism
EXTRA_VOCAB_SIZE=$8            # Additional vocabulary size
NUM_EXPERTS=$9                 # Number of experts
EXPERTS_TOPK=${10}             # Top-k experts for routing
EP=${11}                       # Expert parallelism
mg2hf=${12}                    # Whether to perform mcore2hf conversion
WS=${13}                       # World size for 8x7B
```

Run the `run_pretrain_mcore_mistral.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Per GPU batch size: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpointing mode: sel, full
DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
MOE=${19}                       # Enable MoE: true, false
SAVE_INTERVAL=${20}             # Checkpoint save interval
DATASET_PATH=${21}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-trained model path
TRAIN_TOKENS=${23}              # Number of training tokens
WARMUP_TOKENS=${24}             # Number of warmup tokens
OUTPUT_BASEPATH=${25}           # Output path for training
```

Run the `run_finetune_mcore_mistral_withGA.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Per GPU batch size: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpointing mode: sel, full
DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
MOE=${19}                       # Enable MoE: true, false
SAVE_INTERVAL=${20}             # Checkpoint save interval
DATASET_PATH=${21}              # Training dataset path
VALID_DATASET_PATH=${22}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${23}  # Pre-trained model path
TRAIN_ITERS=${24}               # Number of training steps
WARMUP_ITERS=${25}              # Number of warmup steps
OUTPUT_BASEPATH=${26}           # Output path for training
```

Here is the English translation of the commands for working with the Megatron-Core-Dense and Megatron-Core-MoE models:

## Megatron-Core-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1  \
4  \
1  \
0  \
0  \
0  \
0 \
false
```

## Megatron-Core-Dense Continue Pre-training
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1   \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

## Megatron-Core-Dense Instruction Fine-tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_mcore_mistral_withGA.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4  \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1   \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

# Megatron-Core-MoE Model Training Process

## Megatron-Core-MoE Model Format Conversion
Conversion from Dense to MoE model format based on Sparse-Upcycled
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
4  \
1  \
0  \
2  \
2  \
1 \
false
```

Directly converting Mixtral-8x7B model to Mcore format
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
8x7B \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1-to-mcore-tp4-pp1-ep4-exp8-ws16 \
4  \
1  \
0  \
8  \
2  \
4 \
false \
16
```

## Megatron-Core-MoE Continue Pre-training
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
true \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

## Megatron-Core-MoE Instruction Fine-tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_mcore_mistral_withGA.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
true \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

# Downstream Task Evaluation

## Convert Megatron-LM-Dense Model to Huggingface Format
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1/release  \
/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf    \
4  \
1  \
mistral-7b \
0 \
true
```

## Run Evaluation Tool
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```