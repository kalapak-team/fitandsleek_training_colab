"""
Fitandsleek LoRA fine-tune (needs GPU — use Google Colab / Kaggle / HF).

Example (Colab):
  !pip install -U transformers datasets peft accelerate bitsandbytes trl
  !python scripts/train_lora.py

Outputs adapter to: models/fitandsleek-lora

Then on PC set in .env:
  MODEL_ID=Qwen/Qwen2.5-3B-Instruct
  LORA_ADAPTER=models/fitandsleek-lora
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "fitandsleek_train.jsonl"
OUT_DIR = ROOT / "models" / "fitandsleek-lora"
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def load_rows():
    rows = []
    with DATA_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU not found. Run this script on Colab/Kaggle/HF with a GPU.\n"
            "On your PC without GPU, keep using FAQ + DB + RAG only."
        )

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python scripts/prepare_train_data.py")

    gpu_name = torch.cuda.get_device_name(0)
    # Colab T4 reports bf16 support in some torch builds but AMP bf16 ops fail.
    # Force a stable path: float16 compute + no GradScaler AMP.
    compute_dtype = torch.float16
    print(f"TRAIN_FIX_V2 | GPU={gpu_name} | compute_dtype=float16 | amp=off")

    rows = load_rows()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows).map(to_text)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    args = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=[],
        max_length=512,
        gradient_checkpointing=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    marker = OUT_DIR / "adapter_config.json"
    if not marker.exists():
        raise SystemExit(f"Training finished but adapter missing: {marker}")
    print(f"Saved LoRA adapter -> {OUT_DIR}")
    print("OK: adapter_config.json found")


if __name__ == "__main__":
    main()
