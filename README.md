# LoRA OpenLLaMA 13B — HH-RLHF

Lightweight LoRA adapters trained on Anthropic HH-RLHF for `openlm-research/open_llama_13b`. Includes training script, SLURM jobs, inference CLI, and Hugging Face upload guidance.

## Artifacts
- Final adapter: `sagnik-das/lora-openllama-13b-hh-rlhf`
- Checkpoints: `outputs/checkpoint-{1500,3000,4500,6000,7000}/`
- Training script: `train_lora.py`
- Inference: `inference.py` + SLURM job `inference.model`

## Quickstart (Cluster)
1) Activate env (module + venv assumed in .model scripts):  
   `source .venv/bin/activate`
2) Run training (A40):  
   `sbatch lora.model`
3) Run inference (A40, prompt via env):  
   `PROMPT="Explain why good documentation matters." sbatch inference.model`
   Output goes to `logs/lora/inference_<jobid>.out`.
4) Interactive inference on node:  
   `python inference.py --prompt "Give me a short safety tip."`

## Hugging Face Upload
1) Login once (no tokens in files):  
   `HUGGINGFACE_HUB_TOKEN=*** huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"`
2) Create repo (once):  
   `huggingface-cli repo create lora-openllama-13b-hh-rlhf --type model --private`
3) Upload final adapter:  
   `huggingface-cli upload lora-openllama-13b-quality sagnik-das/lora-openllama-13b-hh-rlhf --commit-message "Add final adapter"`
4) Upload checkpoints:  
   `huggingface-cli upload outputs sagnik-das/lora-openllama-13b-hh-rlhf --commit-message "Add training checkpoints"`

## Use the Adapter
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "openlm-research/open_llama_13b"
adapter = "sagnik-das/lora-openllama-13b-hh-rlhf"

tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

prompt = "Explain why good documentation matters."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Setup (train_lora.py)
- Base: `openlm-research/open_llama_13b`
- Data: `Anthropic/hh-rlhf` (chosen/rejected pairs -> quality labels)
- LoRA: r=8, alpha=16, dropout=0.05, targets q/k/v/o/gate/up/down proj, bias=none
- Optim: AdamW lr=2e-4, wd=0.01, warmup=100, max_steps=7000, grad_accum=16
- Batching: bs=1 train/eval, seq len 256, pad
- Precision: bf16 when available
- Checkpoints: every 1500 steps to `outputs/`

## SLURM Jobs
- `lora.model`: training on A40 (uses env caches, optional `HUGGINGFACE_HUB_TOKEN`)
- `inference.model`: batch inference on A40; set `PROMPT` or `PROMPT_FILE`; optional `OUTPUT_FILE`

## Repo Notes
- `.gitignore` excludes .model scripts, logs, outputs, adapters; HF uploads bypass gitignore when paths are specified.
- Keep tokens out of tracked files; provide via env (`HUGGINGFACE_HUB_TOKEN`) or cached `huggingface-cli login`.

## License
MIT for code; model card/license in adapter READMEs.
