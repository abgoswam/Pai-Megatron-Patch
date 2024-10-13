### Fine-tuning Data Preparation

The data for fine-tuning should be organized such that each line represents a sample, formatted as a JSON dictionary. An example is shown below:
```shell
{"instruction": "Read the paragraph below and find a metaphor.", "input": "\"My troubles grew wings and flew into the sky.\"", "output": "Metaphor: My troubles grew wings and flew into the sky."}
```
You can also download a sample fine-tuning dataset provided by us, as shown:
```bash
mkdir /mnt/workspace/qwen-datasets
cd /mnt/workspace/qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/qwen_sft.json
```

### Code Preparation
Visit the [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) repository to download the source code for the Pai-Megatron-Patch tool used for Megatron model training. Copy the code to your working directory at `/mnt/workspace/`.
```bash
# Clone the repository
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

### Analyzing Dataset Length Distribution
To analyze the distribution of sample lengths in the dataset, run the following commands:
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
python sample_stats.py /mnt/workspace/qwen-datasets/qwen_sft.json
```
The output will be similar to:
```bash
count    50151.000000            # Total number of samples
mean       113.771131            # Average length of samples
std         90.073800            # Standard deviation
min          6.000000            # Minimum sample length
25%         51.000000            # 25% of samples are shorter than 51
50%         93.000000            # 50% of samples are shorter than 93
75%        150.000000            # 75% of samples are shorter than 150
max       2458.000000            # Maximum sample length
```

### Creating an MMAP Format Pre-training Dataset
MMAP data is a pre-tokenized format that can significantly reduce data loading time during fine-tuning, especially with large datasets.

Navigate to the code directory: `/mnt/workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing` in the terminal, and examine the `run_build_idxmap_data_for_sft.sh` script. The script requires five input arguments:
```
input_data_path=$1                # Set input file path
tokenizer=$2                      # Set tokenizer
seq_len=$3                        # Set sequence length for training
output_data_path=$4               # Set output file path  
load_dir=$5                       # Set the path for the HF model
```
Example of running the script:
```bash
sh run_build_idxmap_data_for_sft.sh \
/mnt/workspace/qwen-datasets/qwen_sft.json \
Qwen2Tokenizer \
256 \
/mnt/workspace/qwen-datasets/mmap_qwen2_sft_datasets \
/mnt/workspace/qwen-ckpts/Qwen2-0.5B
```
After execution, the `qwen-datasets` folder will contain two MMAP files with the same name but different extensions. You need to use `/mnt/workspace/qwen-datasets/mmap_qwen2_sft_datasets_text_document` in your training script:
```bash
qwen-datasets
   ├── mmap_qwen2_sft_datasets_text_document.bin
   └── mmap_qwen2_sft_datasets_text_document.idx
```

### Sequence Packing

Pai-Megatron-Patch now supports Sequence Packing for some models (LLaMA3.1, Qwen-2, etc.) based on the MMAP format. Follow these steps to prepare packed datasets:

1. **Download the JSON dataset**
2. **Pack SFT samples:** To enable Sequence Packing, group multiple JSON samples in a list during preprocessing. For example:
   ```json
   {"instruction": "Find two examples of binary classification problems.", "input": "", "output": "1. Spam filtering: classifying emails as spam or not spam.\n2. Credit risk assessment: classifying loan applicants as high-risk or low-risk."}
   {"instruction": "Rewrite the given sentence as a rhetorical question.", "input": "He had never seen the ocean before.", "output": "Had he ever seen the ocean before?"}
   {"instruction": "Rephrase the given sentence using different words.", "input": "He always tried to stay ahead in life.", "output": "He constantly strived for success."}
   ```
   becomes:
   ```json
   [{"instruction": "Find two examples of binary classification problems.", "input": "", "output": "1. Spam filtering: classifying emails as spam or not spam.\n2. Credit risk assessment: classifying loan applicants as high-risk or low-risk."},
   {"instruction": "Rewrite the given sentence as a rhetorical question.", "input": "He had never seen the ocean before.", "output": "Had he ever seen the ocean before?"}]
   [{"instruction": "Rephrase the given sentence using different words.", "input": "He always tried to stay ahead in life.", "output": "He constantly strived for success."}]
   ```
   You can use the built-in script for automatic packing by enabling the `--sequence-packing` option.

3. **Run the following command to create a packed MMAP dataset for LLaMA3.1 SFT:**
   ```
   bash run_build_packed_idxmap_sft_dataset.sh \
   /workspace/llama-datasets/packed_qwen_sft.json \
   LLama3Tokenizer \
   2048 \
   /workspace/llama-datasets/packed_sft_dataset \
   /workspace/Meta-Llama-3.1-8B
   ```

4. **Fine-tuning:** Set `SFT=true` and enable `MP_SFT_PACKING=true` to use Sequence Packing.

### Downloading a Small Preprocessed Dataset for Testing
To make testing easier, we provide a small preprocessed dataset:
```bash
cd /mnt/workspace/qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen2_sft_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen2_sft_datasets_text_document.idx
```