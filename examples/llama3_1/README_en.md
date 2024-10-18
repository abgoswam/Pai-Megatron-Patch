# Best Practices for Meta LLaMA 3.1 Model in Pai-Megatron-Patch

## Table of Contents
   * [Installation](#installation)
   * [Dataset & Model Download](#dataset-and-model-download)
   * [Megatron-Core Model Training Workflow](#megatron-core-model-training-workflow)
      * [Model Format Conversion](#model-format-conversion)
      * [Continued Pre-Training](#continued-pre-training)
      * [Instruction Tuning](#instruction-tuning)
   * [Evaluation on Downstream Tasks](#evaluation-on-downstream-tasks)
      * [Megatron-Core Model Format Conversion](#evaluation-format-conversion)
      * [Running Evaluation Tools](#running-evaluation-tools)

## Installation

In Alibaba Cloud AI platform PAI, provide the custom image address: `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07`

Clone the Pai-Megatron-Patch repository:
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

Currently, LLaMA 3.1 supports FlashAttention-3 for accelerated computation, but it can only run on Hopper architecture GPUs. To use FlashAttention-3 on Hopper GPUs, install it in the DSW container as follows:
```bash
pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
```

## Dataset and Model Download

```bash
cd /mnt
mkdir llama3-ckpts
cd llama3-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-ckpts/Meta-Llama-3.1-8B.tgz
tar -zxf Meta-Llama-3.1-8B.tgz
cd ..
mkdir llama3-datasets
cd llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```

## Megatron-Core Model Training Workflow

### Model Format Conversion
Run the `hf2mcore_convertor_llama3_1.sh` script with the following parameters:
```bash
MODEL_SIZE=$1                 # Model size: 8B/70B
SOURCE_CKPT_PATH=$2           # Source checkpoint path
TARGET_CKPT_PATH=$3           # Target checkpoint path
TP=$4                         # Tensor parallelism
PP=$5                         # Pipeline parallelism
mg2hf=$6                      # Perform mcore2hf conversion
CHECK=$7                      # Check if model outputs match before and after conversion
CHECK_ONLY=$8                 # Only check model output, no conversion
PR=$9                         # Precision: fp16/bf16/fp32
HF_CKPT_PATH=${10}            # HF checkpoint path [required if mg2hf=true]
```

Example:
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
bash hf2mcore_convertor_llama3_1.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B/mcore-tp4-pp2 \
4 \
2 \
false \
true \
false \
bf16
```

### Megatron-Core Pre-Training and Instruction Tuning
Pre-training and fine-tuning for LLaMA 3.1 have been integrated into the `run_mcore_llama.sh` script. Parameters vary depending on the scenario.

#### Pre-Training & Fine-Tuning Command
Parameters:
```bash
ENV=$1                          # Environment: dsw for single machine, dlc for multi-machine
MODEL_SIZE=$2                   # Model size: 8B, 70B
BATCH_SIZE=$3                   # Batch size per data parallel instance
GLOBAL_BATCH_SIZE=$4            # Total batch size across all data parallel instances
LR=$5                           # Learning rate
MIN_LR=$6                       # Minimum learning rate
SEQ_LEN=$7                      # Sequence length
PAD_LEN=$8                      # Padding length
PR=${9}                         # Precision: fp16, bf16, fp8
TP=${10}                        # Tensor parallelism
PP=${11}                        # Pipeline parallelism
CP=${12}                        # Context parallelism
SP=${13}                        # Sequence parallelism: true, false
DO=${14}                        # Megatron Zero-1 optimizer: true, false
FL=${15}                        # Use Flash Attention: true, false
SFT=${16}                       # Perform fine-tuning: true, false
AC=${17}                        # Activation checkpoint mode: sel, full, offload, false
OPTIMIZER_OFFLOAD=${18}         # Offload optimizer: false, static, auto
SAVE_INTERVAL=${19}             # Checkpoint save interval
DATASET_PATH=${20}              # Training dataset path
VALID_DATASET_PATH=${21}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-trained model path
TRAIN_TOKENS_OR_ITERS=${23}     # Training tokens or iterations
WARMUP_TOKENS_OR_ITERS=${24}    # Warmup tokens or iterations        
OUTPUT_BASEPATH=${25}           # Output path for logs
```

#### Continued Pre-Training Example
Use this command to continue training LLaMA 3.1. When `AC=offload` or `full`, you can set the `MP_AC_LAYERS` environment variable to control checkpointing or offloading of Transformer layers (default: `1`).
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3_1
sh run_mcore_llama3_1.sh \
dsw \
8B \
1 \
8 \
1e-5 \
1e-6 \
128 \
128 \
bf16 \
4 \
2 \
1 \
true \
true \
true \
false \
false \
false \
100000 \
/mnt/llama3-datasets/wudao_llama3bpe_content_document \
/mnt/llama3-datasets/wudao_llama3bpe_content_document \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B/mcore-tp4-pp2 \
10000 \
100 \
/workspace/output_mcore_llama3_1
```

#### Instruction Tuning Example
To perform instruction tuning, set the SFT flag to `true`. Create an index map for the fine-tuning dataset as described [here](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing).
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3_1
sh run_mcore_llama3_1.sh \
dsw \
8B \
1 \
8 \
1e-5 \
1e-6 \
128 \
128 \
bf16 \
4 \
2 \
1 \
true \
true \
true \
true \
false \
false \
100000 \
/mnt/llama3-datasets/path_to_your_dataset \
/mnt/llama3-datasets/path_to_your_dataset \
/path/to/pretraining/checkpoint \
10000 \
100 \
/workspace/output_mcore_llama3_1
```

You can also use JSON datasets by setting `MP_DATASET_TYPE` to `raw`.
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/llama3_1
sh run_mcore_llama3_1.sh \
dsw \
8B \
1 \
8 \
1e-5 \
1e-6 \
128 \
128 \
bf16 \
4 \
2 \
1 \
true \
true \
true \
true \
false \
false \
100000 \
/mnt/llama3-datasets/alpaca_zh-llama3-train.json \
/mnt/llama3-datasets/alpaca_zh-llama3-valid.json \
/path/to/pretraining/checkpoint \
10000 \
100 \
/workspace/output_mcore_llama3_1
```

## Downstream Task Evaluation

### Evaluation Format Conversion
To perform inference evaluation, you need to convert the saved Megatron-Core model after training/fine-tuning to Hugging Face format.

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
bash hf2mcore_convertor_llama3_1.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B/mcore-tp4-pp2    \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B/hf-from-mg  \
4  \
2  \
true \
true \
false \
bf16 \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B
```

### Running Evaluation Tools
Download the evaluation data:
```bash
# In container
cd /workspace

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/cmmlu.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz 

tar -xvzf cmmlu.tgz 
tar -xvzf ceval.tgz 
tar -xvzf evaluate.tgz
```

Run the following command to evaluate the converted model.
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/llama3-ckpts/Meta-Llama-3.1-8B/hf-from-mg,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16
```