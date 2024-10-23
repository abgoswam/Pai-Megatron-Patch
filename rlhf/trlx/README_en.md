## RLHF (Reinforcement Learning with Human Feedback)

This guide demonstrates how to use the `trlx` codebase to train a reward model (RM) and optimize using reinforcement learning (PPO). If you're using an SFT model in Megatron format, you need to convert the model files to Hugging Face format first. You can refer to [this guide](../README.md) for more details.

### Installation Guide

Download and install the open-source `trlx` codebase:
```bash
cd PAI-Megatron-Patch/rlhf/trlx
git clone https://github.com/CarperAI/trlx.git
cp trlx_bloom_rlhf.py trlx_bloom_rlhf_test.py trlx/examples/summarize_rlhf/
cp train_reward_model_bloom.py reward_model_bloom.py ds_config_bloom.json trlx/examples/summarize_rlhf/reward_model/
cp -f ds_config_trlx_gptj_summarize.json trlx/examples/summarize_rlhf/configs/
cd trlx
pip install -e .
```

### Reward Model Training (RM)

Training a reward model based on the BLOOM model:
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_bloom.py
```

Training a reward model based on the GPT-J model:
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_gptj.py
```

### Reinforcement Learning Optimization (PPO)

Optimizing with reinforcement learning based on the BLOOM model:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf.py
```

Optimizing with reinforcement learning based on the GPT-J model:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
```

#### PPO Unit Test

If you want to skip the Supervised Fine-Tuning (SFT) and Reward Model Training (RM) steps and directly test the PPO module, you can run the following command for a unit test:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf_test.py
```