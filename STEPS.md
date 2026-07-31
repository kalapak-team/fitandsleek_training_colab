# Full stack: FAQ + DB + RAG + LLM (+ optional Train/LoRA)

## Step 1 — FAQ (Knowledge Base)
Edit `store_info.json` (policies + `faq` keywords/answers).
AI reloads within ~10 seconds.

## Step 2 — Database
Set Neon URL in `.env`:
```
DATABASE_URL=postgresql://...
```
Used for products, stock, bestsellers.

## Step 3 — RAG + LLM (already in app.py)
Run chatbot:
```powershell
.\venv\Scripts\activate
python app.py
```
Open http://127.0.0.1:7860

Flow: FAQ fast-path → else retrieve FAQ/DB (RAG) → Qwen answers in Khmer.

## Step 4 — Prepare Train dataset
```powershell
python scripts/prepare_train_data.py
```
Creates `data/fitandsleek_train.jsonl` from `store_info.json`.

## Step 5 — Train LoRA (needs GPU — Colab / Kaggle / HF)
On a GPU machine:
```powershell
pip install -U transformers datasets peft accelerate bitsandbytes trl
python scripts/train_lora.py
```
Saves adapter to `models/fitandsleek-lora`.

## Step 6 — Use trained adapter on PC
Copy `models/fitandsleek-lora` to this project, then `.env`:
```
MODEL_ID=Qwen/Qwen2.5-3B-Instruct
USE_LLM=1
LORA_ADAPTER=models/fitandsleek-lora
```
```powershell
pip install peft
python app.py
```

## Notes
- Without GPU: Steps 1–3 are enough to know the store clearly.
- Train (Steps 4–6) improves brand voice / Khmer style; FAQ+DB still required for live prices/stock.
