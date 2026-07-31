import json
import os
import re
import sys
import time

# On Hugging Face ZeroGPU, `spaces` must be imported before any CUDA-related package.
try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr
import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Windows console encoding for Khmer / Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Prefer DATABASE_URL (Neon). Fallback to separate DB_* vars for HF Spaces.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "fitandsleek_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_PORT = os.getenv("DB_PORT", "5432")
BASE_DIR = os.path.dirname(__file__)
PRODUCTS_FILE = os.path.join(BASE_DIR, "products.json")
STORE_INFO_FILE = os.path.join(BASE_DIR, "store_info.json")

# Speed knobs (CPU local is slow — keep generation rare + short)
DB_CACHE_TTL_SEC = 30
STORE_CACHE_TTL_SEC = 10
MAX_HISTORY_TURNS = 1
MAX_NEW_TOKENS = 96
PRODUCT_LIMIT = 6
LOT_LIMIT = 8

_db_cache = {"data": None, "expires_at": 0.0}
_store_info_cache = {"data": None, "expires_at": 0.0, "mtime": None}
_json_products_cache = {"data": None, "mtime": None}


def get_db_connection():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        connect_timeout=5,
    )


def compact_product(row):
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "price": row.get("price"),
        "stock": row.get("stock"),
        "sizes": row.get("sizes"),
        "category": row.get("category"),
    }


def compact_lot(row):
    return {
        "product": row.get("product_name"),
        "size": row.get("size"),
        "color": row.get("color"),
        "qty": row.get("quantity_on_hand"),
        "price": row.get("unit_price"),
    }


def fetch_db_products():
    """Return compact Fitandsleek products + inventory lots from Neon."""
    try:
        with get_db_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        p.id,
                        p.name,
                        p.price,
                        p.stock,
                        p.sizes,
                        c.name AS category
                    FROM products p
                    LEFT JOIN categories c ON c.id = p.category_id
                    WHERE COALESCE(p.is_active, TRUE) = TRUE
                    ORDER BY p.updated_at DESC NULLS LAST, p.id DESC
                    LIMIT {PRODUCT_LIMIT}
                    """
                )
                products = [compact_product(dict(row)) for row in cursor.fetchall()]

                cursor.execute(
                    f"""
                    SELECT
                        p.name AS product_name,
                        il.size,
                        il.color,
                        il.quantity_on_hand,
                        il.unit_price
                    FROM inventory_lots il
                    JOIN products p ON p.id = il.product_id
                    WHERE COALESCE(il.is_sellable, TRUE) = TRUE
                      AND COALESCE(il.quantity_on_hand, 0) > 0
                    ORDER BY il.quantity_on_hand DESC, p.name
                    LIMIT {LOT_LIMIT}
                    """
                )
                lots = [compact_lot(dict(row)) for row in cursor.fetchall()]
                return {"products": products, "inventory_lots": lots}
    except psycopg2.Error as error:
        print(f"PostgreSQL unavailable: {error}")
        return None


def get_db_products(use_cache=True):
    now = time.time()
    if use_cache and _db_cache["data"] is not None and now < _db_cache["expires_at"]:
        return _db_cache["data"]

    data = fetch_db_products()
    if data is not None:
        _db_cache["data"] = data
        _db_cache["expires_at"] = now + DB_CACHE_TTL_SEC
    return data


_top_sellers_cache = {"data": None, "expires_at": 0.0}


def fetch_top_sellers(limit=5):
    """Best-selling products from real order_items sales data."""
    try:
        with get_db_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT
                        COALESCE(oi.name, p.name) AS product_name,
                        SUM(oi.qty)::int AS total_qty,
                        ROUND(SUM(oi.line_total)::numeric, 2) AS total_sales,
                        ROUND(MAX(oi.price)::numeric, 2) AS sample_price
                    FROM order_items oi
                    LEFT JOIN products p ON p.id = oi.product_id
                    WHERE COALESCE(oi.qty, 0) > 0
                    GROUP BY COALESCE(oi.name, p.name)
                    ORDER BY total_qty DESC NULLS LAST, total_sales DESC NULLS LAST
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
    except psycopg2.Error as error:
        print(f"Top sellers query failed: {error}")
        return None


def get_top_sellers(use_cache=True):
    now = time.time()
    if (
        use_cache
        and _top_sellers_cache["data"] is not None
        and now < _top_sellers_cache["expires_at"]
    ):
        return _top_sellers_cache["data"]

    data = fetch_top_sellers()
    if data is not None:
        _top_sellers_cache["data"] = data
        _top_sellers_cache["expires_at"] = now + DB_CACHE_TTL_SEC
    return data


def format_top_seller_answer(rows):
    if not rows:
        return "មិនទាន់មានទិន្នន័យការលក់ក្នុង Database ទេ។"

    top = rows[0]
    name = top.get("product_name") or "N/A"
    qty = top.get("total_qty") or 0
    sales = top.get("total_sales") or 0
    price = top.get("sample_price")

    answer = (
        f"ទំនិញលក់ដាច់បំផុតបច្ចុប្បន្នគឺ «{name}» "
        f"(លក់បាន {qty} ដើម"
    )
    if sales:
        answer += f" សរុបប្រហែល ${sales}"
    if price is not None:
        answer += f" តម្លៃគំរូ ${price}"
    answer += ")។"

    if len(rows) > 1:
        runners = []
        for row in rows[1:3]:
            runners.append(
                f"{row.get('product_name')} ({row.get('total_qty') or 0} ដើម)"
            )
        answer += " បន្ទាប់មក៖ " + " និង ".join(runners) + "។"

    return answer


def load_json_file(file_path, default):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"JSON file not found: {file_path}")
    except (json.JSONDecodeError, OSError) as error:
        print(f"Could not read {file_path}: {error}")
    return default


def load_json_products():
    global _json_products_cache
    try:
        mtime = os.path.getmtime(PRODUCTS_FILE)
    except OSError:
        mtime = None

    if (
        _json_products_cache["data"] is not None
        and _json_products_cache["mtime"] == mtime
    ):
        return _json_products_cache["data"]

    data = load_json_file(PRODUCTS_FILE, [])
    products = data if isinstance(data, list) else []
    _json_products_cache = {"data": products, "mtime": mtime}
    return products


def load_store_info():
    """Reload store rules from JSON often so policy edits apply quickly."""
    global _store_info_cache
    now = time.time()
    try:
        mtime = os.path.getmtime(STORE_INFO_FILE)
    except OSError:
        mtime = None

    cached = _store_info_cache
    if (
        cached["data"] is not None
        and cached["mtime"] == mtime
        and now < cached["expires_at"]
    ):
        return cached["data"]

    data = load_json_file(STORE_INFO_FILE, {})
    store = data if isinstance(data, dict) else {}
    _store_info_cache = {
        "data": store,
        "expires_at": now + STORE_CACHE_TTL_SEC,
        "mtime": mtime,
    }
    return store


def store_template_vars(store_info):
    return {
        "store_name": store_info.get("store_name", "Fitandsleek"),
        "assistant_name": store_info.get(
            "assistant_name", "Fitandsleek AI Assistant"
        ),
        "owner_model": store_info.get("owner_model", ""),
        "target_customers": store_info.get("target_customers", ""),
        "categories": ", ".join(store_info.get("product_categories", [])),
        "product_source": store_info.get("product_source", ""),
        "price_range": store_info.get("price_range", ""),
        "sales_type": store_info.get("sales_type", ""),
        "payments": ", ".join(store_info.get("payment_methods", [])),
        "delivery": ", ".join(store_info.get("delivery_services", [])),
        "online_payment_policy": store_info.get("online_payment_policy", ""),
        "damage_policy": store_info.get("damage_policy", ""),
        "opening_hours": store_info.get("opening_hours", ""),
        "location": store_info.get("location", ""),
        "contact": store_info.get("contact", ""),
        "return_policy": store_info.get("return_policy", ""),
        "language_note": store_info.get("language_note", ""),
    }


def fill_store_answer(template, store_info):
    if not template:
        return ""
    try:
        return str(template).format(**store_template_vars(store_info)).strip()
    except (KeyError, ValueError):
        return str(template).strip()


def match_faq_answer(cleaned, store_info):
    """Match FAQ rules from store_info.json (editable without code changes)."""
    for item in store_info.get("faq", []) or []:
        keywords = item.get("keywords") or []
        if any(str(k).lower() in cleaned for k in keywords):
            return fill_store_answer(item.get("answer", ""), store_info)
    return None


def message_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return message_text(content.get("text", content.get("content", "")))
    if isinstance(content, (list, tuple)):
        return "\n".join(filter(None, (message_text(part) for part in content)))
    return str(content)


def needs_product_data(text):
    keywords = [
        "តម្លៃ",
        "price",
        "size",
        "ទំហំ",
        "ស្តុក",
        "stock",
        "មានលក់",
        "ផលិតផល",
        "ទំនិញ",
        "បញ្ជី",
        "list",
        "អាវ",
        "ខោ",
        "ស្បែកជើង",
        "កាបូប",
        "មួក",
        "នាឡិកា",
        "ស្រោម",
        "ខ្សែ",
        "hat",
        "backpack",
        "jacket",
        "chino",
        "product",
    ]
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def is_bestseller_question(text):
    keywords = [
        "លក់ដាច់",
        "លក់ខ្លាំង",
        "best seller",
        "bestseller",
        "best-selling",
        "best selling",
        "top seller",
        "top-selling",
        "popular",
        "most popular",
        "trending",
        "ពេញនិយម",
        "ដាច់បំផុត",
        "លក់ច្រើន",
        "ទំនិញលក់ដាច់",
    ]
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


ASSISTANT_NAME = "Fitandsleek AI Assistant"


def format_product_line(item):
    name = item.get("name") or item.get("product") or "N/A"
    price = item.get("price")
    stock = item.get("stock")
    if stock is None:
        stock = item.get("qty")
    size = item.get("size") or item.get("sizes")
    color = item.get("color")

    parts = [f"«{name}»"]
    if price is not None and price != "":
        parts.append(f"តម្លៃ ${price}")
    if size not in (None, "", []):
        parts.append(f"ទំហំ {size}")
    if color:
        parts.append(f"ពណ៌ {color}")
    if stock is not None and stock != "":
        parts.append(f"ស្តុក {stock}")
    return " — ".join(parts)


def identity_answer(store_info):
    for item in store_info.get("faq") or []:
        if item.get("id") == "identity":
            return fill_store_answer(item.get("answer", ""), store_info)
    return fill_store_answer(
        "ខ្ញុំជា {assistant_name} របស់ហាង {store_name}។ ខ្ញុំមិនមានឈ្មោះផ្សេងទេ។",
        store_info,
    )


def sanitize_reply(answer, store_info):
    """Stop small models from inventing fake names / English identity."""
    if not answer:
        return identity_answer(store_info)

    text = answer.strip()
    lower = text.lower()
    bad_name = re.search(
        r"\bmy name is\b|\bi am (koh|alex|sam|ai|assistant)\b|ឈ្មោះខ្ញុំគឺ",
        lower,
    )
    if bad_name and "fitandsleek" not in lower:
        return identity_answer(store_info)

    if any(k in text for k in ("បញ្ជាក់ការលក់", "order confirmed", "checkout")):
        return (
            "ខ្ញុំអាចបង្ហាញព័ត៌មានទំនិញ តម្លៃ និងស្តុកបាន។ "
            "សូមផ្ញើឈ្មោះទំនិញ ឬសួរ «បញ្ជីផលិតផល»។"
        )

    return text


def try_product_answer(text):
    """Answer product/stock/price questions from DB/JSON without the LLM."""
    if is_bestseller_question(text):
        return None

    db_data = get_db_products(use_cache=True) or {}
    products = list(db_data.get("products") or [])
    lots = list(db_data.get("inventory_lots") or [])
    if not products:
        products = load_json_products()

    cleaned = text.strip().lower()
    wants_list = any(
        k in cleaned
        for k in (
            "បញ្ជី",
            "ទាំងអស់",
            "all product",
            "product list",
            "list product",
            "មានអ្វីខ្លះ",
            "បង្ហាញទំនិញ",
            "បង្ហាញផលិតផល",
        )
    )

    tokens = [t for t in re.split(r"[\s,./?؟!]+", cleaned) if len(t) >= 3]

    def score(item):
        blob = " ".join(
            str(item.get(k, ""))
            for k in ("name", "product", "category", "color", "size", "sizes")
        ).lower()
        hits = sum(1 for t in tokens if t in blob)
        pname = str(item.get("name") or item.get("product") or "").lower()
        if pname and pname in cleaned:
            hits += 10
        return hits

    ranked_products = sorted(products, key=score, reverse=True)
    ranked_lots = sorted(lots, key=score, reverse=True)
    matched_products = [p for p in ranked_products if score(p) > 0][:5]
    matched_lots = [lot for lot in ranked_lots if score(lot) > 0][:5]

    if matched_products and score(matched_products[0]) >= 3:
        lines = [format_product_line(p) for p in matched_products[:3]]
        return "ព័ត៌មានទំនិញ៖\n- " + "\n- ".join(lines)

    if not needs_product_data(text) and not wants_list and not matched_products:
        return None

    wants_stock = any(k in cleaned for k in ("ស្តុក", "stock", "មានទេ", "នៅសល់"))
    wants_price = any(k in cleaned for k in ("តម្លៃ", "price", "ថ្លៃ", "ប៉ុន្មាន"))

    if matched_lots and (wants_stock or wants_price or matched_products):
        lines = [format_product_line(lot) for lot in matched_lots]
        return "ព័ត៌មានទំនិញពីស្តុក៖\n- " + "\n- ".join(lines)

    if matched_products:
        lines = [format_product_line(p) for p in matched_products]
        return "ទំនិញដែលពាក់ព័ន្ធ៖\n- " + "\n- ".join(lines)

    if products and (wants_list or needs_product_data(text)):
        lines = [format_product_line(p) for p in products[:8]]
        return "ទំនិញបច្ចុប្បន្នមានដូចខាងក្រោម៖\n- " + "\n- ".join(lines)

    if needs_product_data(text) or wants_list:
        return "មិនទាន់មានទិន្នន័យផលិតផលទេ។"
    return None


def try_fast_answer(text, store_info):
    """Instant answers for greetings and common store FAQ (no model)."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if not cleaned:
        return "សួស្តី! សូមស្វាគមន៍មកកាន់ Fitandsleek។ តើអ្នកចង់សួរអ្វីដែរ?"

    greetings = {"hi", "hello", "hey", "សួស្តី", "សួស្ដី", "ជំរាបសួរ", "សួស្តី!"}
    if cleaned in greetings or cleaned.rstrip("!?.") in greetings:
        return (
            "សួស្តី! ខ្ញុំជា Fitandsleek AI Assistant។ "
            "តើអ្នកចង់សួរអ្វីអំពីហាង Fitandsleek?"
        )

    how_are_you = {
        "how are you",
        "how about you",
        "how's it going",
        "អ្នកសុខសប្បាយទេ",
    }
    if cleaned in how_are_you or cleaned.rstrip("!?.") in how_are_you:
        return (
            "សុខសប្បាយជាទេ! ខ្ញុំជា Fitandsleek AI Assistant។ "
            "តើអ្នកចង់សួរអ្វីអំពីទំនិញ ឬហាង?"
        )

    thanks = {"អរគុណ", "សូមអរគុណ", "thanks", "thank you", "thx"}
    if cleaned in thanks or cleaned.rstrip("!?.") in thanks:
        return "សូមអរគុណ! បើចង់សួរបន្តអំពីទំនិញ ឬហាង Fitandsleek សូមសរសេរមកបាន។"

    if is_bestseller_question(cleaned):
        rows = get_top_sellers(use_cache=True)
        if rows is None:
            return "មិនអាចទាញទិន្នន័យលក់ពី Database បានទេ។ សូមព្យាយាមម្តងទៀត។"
        return format_top_seller_answer(rows)

    # FAQ rules live in store_info.json — edit that file to update AI knowledge
    faq_answer = match_faq_answer(cleaned, store_info)
    if faq_answer:
        return faq_answer

    product_answer = try_product_answer(text)
    if product_answer:
        return product_answer

    return None


def build_compact_context(json_data, db_data, store_info, include_products):
    store_bits = {
        "store_name": store_info.get("store_name"),
        "assistant_name": store_info.get(
            "assistant_name", "Fitandsleek AI Assistant"
        ),
        "target_customers": store_info.get("target_customers"),
        "product_categories": store_info.get("product_categories"),
        "price_range": store_info.get("price_range"),
        "payment_methods": store_info.get("payment_methods"),
        "delivery_services": store_info.get("delivery_services"),
        "online_payment_policy": store_info.get("online_payment_policy"),
        "sales_type": store_info.get("sales_type"),
        "damage_policy": store_info.get("damage_policy"),
        "opening_hours": store_info.get("opening_hours"),
        "location": store_info.get("location"),
        "contact": store_info.get("contact"),
    }
    parts = ["STORE: " + json.dumps(store_bits, ensure_ascii=False, default=str)]

    if not include_products:
        return "\n".join(parts)

    if db_data and db_data.get("products"):
        parts.append(
            "DB_PRODUCTS: "
            + json.dumps(db_data["products"], ensure_ascii=False, default=str)
        )
        parts.append(
            "DB_LOTS: "
            + json.dumps(db_data.get("inventory_lots", []), ensure_ascii=False, default=str)
        )
    elif json_data:
        slim = [
            {
                "name": p.get("name"),
                "price": p.get("price"),
                "sizes": p.get("sizes"),
                "stock": p.get("stock_quantity", p.get("in_stock")),
            }
            for p in json_data[:PRODUCT_LIMIT]
        ]
        parts.append("JSON_PRODUCTS: " + json.dumps(slim, ensure_ascii=False, default=str))
    else:
        parts.append("PRODUCTS: unavailable")

    return "\n".join(parts)


def retrieve_rag_context(question, store_info):
    """Retrieve FAQ + DB snippets relevant to the user question (keyword RAG)."""
    cleaned = re.sub(r"\s+", " ", question.strip().lower())
    parts = []

    # Score FAQ entries
    scored = []
    for item in store_info.get("faq") or []:
        kws = [str(k).lower() for k in (item.get("keywords") or [])]
        score = sum(1 for k in kws if k and k in cleaned)
        # also soft-match tokens
        tokens = [t for t in re.split(r"[\s,./?؟!]+", cleaned) if len(t) >= 3]
        blob = " ".join(kws)
        score += sum(1 for t in tokens if t in blob)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)

    faq_bits = []
    for _, item in scored[:4]:
        ans = fill_store_answer(item.get("answer", ""), store_info)
        if ans:
            faq_bits.append(f"- {item.get('id', 'faq')}: {ans}")
    if faq_bits:
        parts.append("FAQ_HITS:\n" + "\n".join(faq_bits))
    else:
        # Always give core store facts so the LLM stays grounded
        vars_map = store_template_vars(store_info)
        parts.append(
            "STORE_FACTS: "
            + json.dumps(
                {
                    "store_name": vars_map["store_name"],
                    "assistant_name": vars_map["assistant_name"],
                    "categories": vars_map["categories"],
                    "payments": vars_map["payments"],
                    "delivery": vars_map["delivery"],
                    "price_range": vars_map["price_range"],
                },
                ensure_ascii=False,
            )
        )

    if is_bestseller_question(cleaned):
        rows = get_top_sellers(use_cache=True) or []
        parts.append(
            "TOP_SELLERS: " + json.dumps(rows[:5], ensure_ascii=False, default=str)
        )

    include_products = needs_product_data(question)
    if include_products:
        db_data = get_db_products(use_cache=True)
        json_data = load_json_products()
        parts.append(
            build_compact_context(json_data, db_data, store_info, include_products=True)
        )
    else:
        parts.append(
            build_compact_context([], None, store_info, include_products=False)
        )

    return "\n".join(parts)


# Model works on both local PC and Hugging Face Spaces.
# Best on this PC (no GPU, 32GB RAM): Qwen/Qwen2.5-3B-Instruct
# Smaller / faster:                sail/Sailor2-1B-Chat
# Needs GPU:                       sail/Sailor2-8B-Chat or Qwen3-8B
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct").strip()
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or None
)
print(f"Loading Model: {MODEL_ID} ...")

# ZeroGPU allocates the GPU only inside @spaces.GPU functions, so the model is
# loaded on CPU here and moved to CUDA on first use.
# Set FORCE_CPU=1 to skip ZeroGPU entirely and avoid its per-account run quota.
FORCE_CPU = os.getenv("FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
ON_ZERO_GPU = (
    spaces is not None and bool(os.getenv("SPACES_ZERO_GPU")) and not FORCE_CPU
)

torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

# Optional LoRA adapter from fine-tune (Colab/GPU). Set LORA_ADAPTER=models/fitandsleek-lora
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "").strip()
if LORA_ADAPTER:
    adapter_path = LORA_ADAPTER
    if not os.path.isabs(adapter_path):
        adapter_path = os.path.join(BASE_DIR, adapter_path)
    if os.path.isdir(adapter_path):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            print(f"Loaded LoRA adapter: {adapter_path}")
        except Exception as error:
            print(f"Could not load LoRA adapter ({adapter_path}): {error}")
    else:
        print(f"LORA_ADAPTER path not found: {adapter_path}")

_model_device = "cpu"


def gpu_task(func):
    if not ON_ZERO_GPU:
        return func
    return spaces.GPU(duration=30)(func)


@gpu_task
def generate_reply(prompt):
    global _model_device

    device = "cpu"
    if ON_ZERO_GPU and torch.cuda.is_available():
        device = "cuda"
        if _model_device != "cuda":
            model.to(device=device, dtype=torch.float16)
            _model_device = "cuda"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def predict(message, history):
    text = message_text(message).strip()
    store_info = load_store_info()

    # 1) Instant FAQ / greeting / product lookup (no model = reliable)
    fast = try_fast_answer(text, store_info)
    if fast:
        return fast

    # 2) Prefer FAQ guide over weak free-form LLM (set USE_LLM=1 to enable)
    use_llm = os.getenv("USE_LLM", "0").strip().lower() in {"1", "true", "yes"}
    if not use_llm:
        categories = ", ".join(store_info.get("product_categories", [])[:6])
        return (
            f"ខ្ញុំជា {ASSISTANT_NAME}។ "
            "សូមសួរអំពីទំនិញ តម្លៃ ស្តុក ទំនិញលក់ដាច់ ការទូទាត់ ឬដឹកជញ្ជូន។ "
            f"ឧទាហរណ៍៖ «លក់អ្វីខ្លះ?» «បញ្ជីផលិតផល» «ទំនិញណាលក់ដាច់?»។ "
            f"ប្រភេទទំនិញ៖ {categories}។"
        )

    # 3) LLM + RAG (retrieve FAQ/DB first, then generate)
    context = retrieve_rag_context(text, store_info)
    assistant = store_info.get("assistant_name", ASSISTANT_NAME)

    system_prompt = (
        f"ឈ្មោះរបស់អ្នកគឺ {assistant} តែមួយគត់។ កុំប្រឌិតឈ្មោះផ្សេង។ "
        "ឆ្លើយតែជាភាសាខ្មែរ ខ្លីៗ ច្បាស់។ "
        "ប្រើតែព័ត៌មានខាងក្រោម (RAG context)។ បើគ្មានទិន្នន័យ និយាយថាមិនទាន់មានព័ត៌មាន។ "
        "កុំប្រឌិតតម្លៃ ស្តុក ឬការបញ្ជាទិញ។ កុំឆ្លើយអង់គ្លេស។\n"
        f"{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    recent = (history or [])[-MAX_HISTORY_TURNS:] if MAX_HISTORY_TURNS else []
    for item in recent:
        if isinstance(item, dict):
            role = item.get("role")
            content = message_text(item.get("content"))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": message_text(item[0])})
            messages.append({"role": "assistant", "content": message_text(item[1])})

    messages.append({"role": "user", "content": text})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return sanitize_reply(generate_reply(prompt), store_info)


demo = gr.ChatInterface(
    fn=predict,
    title="Fitandsleek AI Assistant",
    description=(
        "សួរព័ត៌មានអំពីទំនិញ តម្លៃ ស្តុក ការទូទាត់ "
        "និងសេវាដឹកជញ្ជូនរបស់ហាង Fitandsleek។"
    ),
    examples=[
        "សួស្តី!",
        "តើហាងនេះមានឈ្មោះអ្វី?",
        "ហាងមានលក់អ្វីខ្លះ?",
        "ទំនិញណាលក់ដាច់ជាងគេ?",
        "តើបង់លុយតាមអ្វីបាន?",
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        footer_links=[],
    )