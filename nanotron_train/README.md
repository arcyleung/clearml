
## nanotron
1. convert arrow to parquet, (optional: rename "content" column to "text" since most pretrain corpus use "text")
```python
import datasets

ds = datasets.load_from_disk("/home/arthur/Projects/arkts/arks_code_dataset_v1")
ds.to_parquet("/home/arthur/Projects/arkts/arks_code_dataset_v1_parquet")
```

2. run `nanotron/tools/preprocess_data.py`
```python
python3 tools/preprocess_data.py \
--tokenizer-name-or-path HuggingFaceTB/SmolLM-135M \
--output-folder datasets/arkts \
--n-tasks 16 \
hf \
--dataset /home/arthur/Projects/arkts/arkts_code_dataset_v1_parquet
--column content ## (optional) if you haven't renamed it yet, defaults to "text"
```


3. Edit the pretrain config depending on number of GPUs and vRAM available (below config is single node, 2x4080 Super)
```yaml
# smollm/pre-training/smollm1/config_smollm1_135M.yaml
...
parallelism:
  dp: 2 # 4 nodes
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  recompute_layer: false
  tp: 1
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
  tp_recompute_allgather: true
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: HuggingFaceTB/SmolLM-135M
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: 2
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 4 # GBS = 8*2*32*sequence_length = 512*sequence_length = 1M tokens
  sequence_length: 2048
  train_steps: 600000
  val_check_interval: -1
```

4. run `nanotron/examples/custom-dataloader/run_train.py`
```python
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
--nproc_per_node=2 run_train.py \
--config-file ../smollm/pre-training/smollm1/ \
config_smollm1_135M.yaml
```
