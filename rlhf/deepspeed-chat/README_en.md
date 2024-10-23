## RLHF (Reinforcement Learning with Human Feedback)

This section demonstrates how to use the DeepSpeed-Chat codebase for reward model (RM) training and reinforcement learning optimization (PPO). If you're using a model in Megatron format, you need to first convert the Megatron format model files into Hugging Face format, as described [here](../README.md).

### Installation Guide

Download and install the open-source DeepSpeed-Chat code:

```bash
cd PAI-Megatron-Patch/rlhf/deepspeed-chat
git clone https://github.com/microsoft/DeepSpeedExamples.git
cp -f rm_main.py DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py
cp -f utils.py DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/utils.py
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt
```

### Reward Model Training (RM)
Training the reward model based on the LLaMA2 model:

```bash
cd training/step2_reward_model_finetuning/ && bash training_scripts/llama2/run_llama2_7b.sh
```

### Reinforcement Learning Optimization (PPO)
Training with reinforcement learning optimization based on LLaMA2:

```bash
cd training/step3_rlhf_finetuning/ && bash training_scripts/llama2/run_llama2_7b_lora.sh
```