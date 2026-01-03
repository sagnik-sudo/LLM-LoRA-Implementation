import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(adapter_dir: str, base_model: str, device_map: str = "auto"):
    """Load tokenizer + LoRA-wrapped model."""
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if Path(adapter_dir).is_dir() else base_model,
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        torch_dtype=dtype,
    )
    return tokenizer, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoRA OpenLLaMA 13B inference.")
    parser.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL", "openlm-research/open_llama_13b"),
        help="Base model ID to load.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=os.getenv("ADAPTER_DIR", "lora-openllama-13b-quality"),
        help="Path or HF repo for the LoRA adapter.",
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text to generate from.")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing the prompt.",
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=256,
        help="Maximum tokens for the input prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable sampling (default: enabled).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers (default: auto).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional file path to write the generated text.",
    )
    return parser


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def main():
    parser = build_parser()
    args = parser.parse_args()
    prompt = read_prompt(args)

    tokenizer, model = load_models(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        device_map=args.device_map,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

    if args.output_file:
        args.output_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
