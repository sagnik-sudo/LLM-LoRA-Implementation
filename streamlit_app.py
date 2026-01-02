import os
import torch
import streamlit as st
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = os.getenv("BASE_MODEL", "openlm-research/open_llama_13b")
DEFAULT_ADAPTER = os.getenv("ADAPTER_DIR", "lora-openllama-13b-quality")


@st.cache_resource
def load_models(adapter_dir, base_model):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if os.path.isdir(adapter_dir) else base_model,
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return tokenizer, model


def main():
    st.title("LoRA OpenLLaMA 13B Inference")
    adapter_dir = st.text_input("Adapter checkpoint dir", DEFAULT_ADAPTER)
    base_model = st.text_input("Base model", BASE_MODEL)

    if not adapter_dir:
        st.stop()

    tokenizer, model = load_models(adapter_dir, base_model)

    prompt = st.text_area(
        "Prompt",
        value="Explain why good documentation matters.",
        height=180,
    )
    max_new_tokens = st.slider("Max new tokens", 8, 256, 64, 8)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    do_sample = st.checkbox("Use sampling", value=True)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown("**Output**")
        st.write(text)


if __name__ == "__main__":
    main()
