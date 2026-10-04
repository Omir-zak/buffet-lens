"""Register model prices in Langfuse so it can compute the cost of each LLM call.

Run: python -m scripts.setup_langfuse_models

Langfuse only knows prices of well-known models; Groq models (openai/gpt-oss-120b) would show empty
costs in traces. Prices come from eval/prices.json and are ASSUMED list prices: on the free tier the
real spend is zero.
"""

import json
from pathlib import Path

import httpx

from app.config import get_settings

PRICES = json.loads((Path(__file__).resolve().parents[1] / "eval" / "prices.json").read_text(encoding="utf-8"))


def main() -> None:
    settings = get_settings()
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key)
    base = settings.langfuse_host.rstrip("/")
    existing = {
        m["modelName"]
        for m in httpx.get(f"{base}/api/public/models", auth=auth, params={"limit": 100}, timeout=30)
        .json()
        .get("data", [])
    }
    for model, (price_in, price_out) in ((k, v) for k, v in PRICES.items() if not k.startswith("_")):
        if model in existing:
            print(f"уже есть: {model}")
            continue
        response = httpx.post(
            f"{base}/api/public/models",
            auth=auth,
            timeout=30,
            json={
                "modelName": model,
                "matchPattern": f"(?i)^{model}$",
                "unit": "TOKENS",
                "inputPrice": price_in / 1_000_000,  # price per single token
                "outputPrice": price_out / 1_000_000,
            },
        )
        print(f"{'создано' if response.is_success else 'ОШИБКА ' + str(response.status_code)}: {model}")


if __name__ == "__main__":
    main()
