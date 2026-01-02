import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def build_labeled_split(split):
    def expand(batch):
        texts = []
        for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
            texts.append(f"Prompt:\n{chosen}\nQuality: good")
            texts.append(f"Prompt:\n{rejected}\nQuality: bad")
        return {"text": texts}

    return (
        split.map(expand, batched=True, remove_columns=split.column_names)
        .shuffle(seed=42)
    )


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors=None,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def format_pair(prompt, response):
    return f"Prompt:\n{prompt}\nResponse:\n{response}"


def log_probs_from_model(model, tokenizer, prompts, responses, max_length=512):
    model.eval()
    device = model.device
    inputs = [format_pair(p, r) for p, r in zip(prompts, responses)]
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    prompt_only = tokenizer(
        [format_pair(p, "") for p in prompts],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )
    prompt_lengths = (prompt_only.input_ids != tokenizer.pad_token_id).sum(dim=1).to(
        device
    )

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    shift_mask = attention_mask[..., 1:]

    batch_logprobs = []
    for i in range(len(prompts)):
        resp_start = prompt_lengths[i].item()
        labels = shift_labels[i]
        logit = shift_logits[i]
        att = shift_mask[i]

        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[resp_start - 1 :] = True

        valid = (att == 1) & mask
        logprobs = F.log_softmax(logit, dim=-1)
        selected = logprobs[valid, labels[valid]]
        batch_logprobs.append(selected.sum())

    return torch.stack(batch_logprobs)


def benchmark_preference(model, tokenizer, dataset, samples=64, max_length=512):
    subset = dataset.select(range(min(samples, len(dataset))))
    prompts = ["" for _ in range(len(subset))]
    chosens = subset["chosen"]
    rejecteds = subset["rejected"]
    chosen_scores = log_probs_from_model(model, tokenizer, prompts, chosens, max_length)
    rejected_scores = log_probs_from_model(
        model, tokenizer, prompts, rejecteds, max_length
    )
    win_rate = (chosen_scores > rejected_scores).float().mean().item()
    margin = (chosen_scores - rejected_scores).mean().item()
    return {"win_rate": win_rate, "avg_logprob_margin": margin}


def main():
    model_name = "openlm-research/open_llama_13b"
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_split = build_labeled_split(dataset["train"])
    eval_split = build_labeled_split(dataset["test"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenized_train = tokenize_dataset(train_split, tokenizer)
    tokenized_eval = tokenize_dataset(eval_split, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=10,
        save_steps=1500,
        num_train_epochs=1,
        max_steps=7000,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    save_dir = "lora-openllama-13b-quality"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    metrics = benchmark_preference(model, tokenizer, dataset["test"])
    print(metrics)


if __name__ == "__main__":
    main()
