"""
Step: build Fitandsleek training JSONL from store_info.json (+ extra pairs).
Run:  python scripts/prepare_train_data.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE_FILE = ROOT / "store_info.json"
OUT_FILE = ROOT / "data" / "fitandsleek_train.jsonl"


def fill(template: str, vars_map: dict) -> str:
    try:
        return template.format(**vars_map).strip()
    except (KeyError, ValueError):
        return template.strip()


def store_vars(store: dict) -> dict:
    return {
        "store_name": store.get("store_name", "Fitandsleek"),
        "assistant_name": store.get("assistant_name", "Fitandsleek AI Assistant"),
        "owner_model": store.get("owner_model", ""),
        "target_customers": store.get("target_customers", ""),
        "categories": ", ".join(store.get("product_categories", [])),
        "product_source": store.get("product_source", ""),
        "price_range": store.get("price_range", ""),
        "sales_type": store.get("sales_type", ""),
        "payments": ", ".join(store.get("payment_methods", [])),
        "delivery": ", ".join(store.get("delivery_services", [])),
        "online_payment_policy": store.get("online_payment_policy", ""),
        "damage_policy": store.get("damage_policy", ""),
        "opening_hours": store.get("opening_hours", ""),
        "location": store.get("location", ""),
        "contact": store.get("contact", ""),
        "return_policy": store.get("return_policy", ""),
        "language_note": store.get("language_note", ""),
    }


def main() -> None:
    store = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    vars_map = store_vars(store)
    rows: list[dict] = []

    for item in store.get("faq") or []:
        answer = fill(item.get("answer", ""), vars_map)
        if not answer:
            continue
        for kw in item.get("keywords") or []:
            q = str(kw).strip()
            if not q:
                continue
            # Natural question forms
            candidates = [
                q,
                f"តើ{q}?",
                f"{q}?",
                f"សួរអំពី {q}",
            ]
            for question in candidates:
                rows.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    f"អ្នកជា {vars_map['assistant_name']} របស់ហាង "
                                    f"{vars_map['store_name']}។ ឆ្លើយជាភាសាខ្មែរ ខ្លីៗ ច្បាស់។"
                                ),
                            },
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    }
                )

    # Extra brand / behavior pairs
    extras = [
        (
            "សួស្តី",
            f"សួស្តី! ខ្ញុំជា {vars_map['assistant_name']}។ តើអ្នកចង់សួរអ្វីអំពីហាង {vars_map['store_name']}?",
        ),
        (
            "hello",
            f"សួស្តី! ខ្ញុំជា {vars_map['assistant_name']}។ តើអ្នកចង់សួរអ្វីអំពីហាង {vars_map['store_name']}?",
        ),
        (
            "what is your name?",
            f"ខ្ញុំជា {vars_map['assistant_name']}។ ខ្ញុំមិនមានឈ្មោះផ្សេងទេ។",
        ),
        (
            "My name is Koh",
            f"ខ្ញុំមិនមែនឈ្មោះ Koh ទេ។ ខ្ញុំជា {vars_map['assistant_name']} របស់ហាង {vars_map['store_name']}។",
        ),
        (
            "tell me about the store",
            (
                f"ហាង {vars_map['store_name']} លក់សម្រាប់បុរស។ "
                f"ប្រភេទទំនិញ៖ {vars_map['categories']}។ "
                f"បង់តាម៖ {vars_map['payments']}។ ដឹក៖ {vars_map['delivery']}។"
            ),
        ),
    ]
    for question, answer in extras:
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"អ្នកជា {vars_map['assistant_name']} របស់ហាង "
                            f"{vars_map['store_name']}។ ឆ្លើយជាភាសាខ្មែរ ខ្លីៗ ច្បាស់។"
                        ),
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples -> {OUT_FILE}")


if __name__ == "__main__":
    main()
