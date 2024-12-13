## llama-factory
1. Install LLaMA Factory
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Prepare the data
Convert the data into json (Skip this if it's already json)
```python
import pandas as pd

input_parquet = "dataset.parquet"     # Path to your input Parquet file
output_json = "dataset.json"          # Path to your output JSON file

# Load the parquet file into a pandas DataFrame
df = pd.read_parquet(input_parquet, engine="pyarrow")

# Convert DataFrame to JSON
# orient="records" creates a list of dictionaries for each row
# lines=True outputs in JSON Lines (one JSON object per line)
df.to_json(output_json, orient="records", lines=True, force_ascii=False)

print(f"Successfully converted {input_parquet} to {output_json}")

```

Move the data to `<LLaMA-Factory path>/data/`

```bash
cp dataset.json <LLaMA-Factory path>/data/
```

In `<LLaMA-Factory path>/data/dataset_info.json`
Add the arkts data
```
"arkts": {
    "file_name": "dataset.json",
    "columns": {
      "prompt": "content"
    }
},
```

3. Modify the training config
Modify `<LLaMA-Factory path>/examples/train_lora/qwen2vl_lora_sft.yaml`
Other parts keep the same
```
### model
model_name_or_path: <path to Qwen2.5-Coder-0.5B>

### method
stage: pt
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: arkts  # video: mllm_video_demo
template: qwen2_vl
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
```

4. Start training
```bash
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
```