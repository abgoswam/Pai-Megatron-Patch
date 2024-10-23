# RLHF (Reinforcement Learning with Human Feedback)

Generally, a model that has been fine-tuned with SFT (Supervised Fine-Tuning) will already perform well in dialogue scenarios. If you want to further improve the model’s performance, RLHF training can be added. This includes training the Reward Model (RM) and Reinforcement Learning (PPO). Below is a guide on how to use the most commonly used open-source RLHF frameworks, DeepSpeed-Chat and trlx, to train the reward function (RM) and perform reinforcement learning optimization (PPO).

## Model Format Conversion

If you are using a Hugging Face format model for reward model training (RM) and reinforcement learning optimization (PPO), you can skip this step.

If you are using a Megatron format model, such as an SFT model trained with PAI-Megatron-Patch, and want to train RM and PPO, you will need to use the provided model conversion script to convert the Megatron format model files to Hugging Face format.

LLaMA2 model conversion:
```bash
cd PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/llama2
bash model_convertor.sh \
/path/to/Megatron-LM \
/path/to/megatron_llama2_ckpt \
/path/to/hf_llama2_ckpt \
1 \
1 \
llama-7b \
0 \
true
```

BLOOM model conversion:
```bash
cd PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/bloom
bash model_convertor_huggingface_megatron.sh \
/path/to/Megatron-LM \
/path/to/megatron_bloom_ckpt \
/path/to/hf_bloom_ckpt \
1 \
1 \
true
```

## DeepSpeed-Chat

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
Training a reward model based on LLaMA2:
```bash
cd training/step2_reward_model_finetuning/ && bash training_scripts/llama2/run_llama2_7b.sh
```

### Reinforcement Learning Optimization (PPO)
Performing reinforcement learning optimization based on LLaMA2:
```bash
cd training/step3_rlhf_finetuning/ && bash training_scripts/llama2/run_llama2_7b_lora.sh
```

## trlx

### Installation Guide

Download and install the open-source trlx code:
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
Training a reward model based on BLOOM:
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_bloom.py
```

Training a reward model based on GPT-J:
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_gptj.py
```

### Reinforcement Learning Optimization (PPO)
Performing reinforcement learning optimization based on BLOOM:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf.py
```

Performing reinforcement learning optimization based on GPT-J:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
```

#### PPO Unit Test
If you want to skip the SFT and RM steps and directly test the performance of the PPO module, you can run the following command to perform a standalone PPO test:
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf_test.py
```