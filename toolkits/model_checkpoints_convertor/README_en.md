
Here is the English translation:

## hf-to-megatron

hf-to-megatron is a model checkpoint conversion tool that allows users to easily convert Hugging Face checkpoints to Megatron format, enabling the use of Megatron-LM's distributed training capabilities for large language models (LLMs). The converted models need to be used with the PAI-Megatron-Patch codebase. Currently, the following models are supported:

+ Bloom
+ Llama/Alpaca
+ ChatGLM
+ Galactica
+ GLM
+ GLM130B
+ Falcon
+ StarCoder

The converted models are stored at: `oss://atp-modelzoo/release/models/pai-megatron-patch/`