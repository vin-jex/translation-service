from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
import time

HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://vinjex-translation-ml.hf.space")

TIMEOUT_POST = 300
IDLE_TIMEOUT = 60

app = FastAPI(title="LearnSci Translation Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str

class BatchItem(BaseModel):
    text: str

class BatchRequest(BaseModel):
    items: list[BatchItem]

class TeachRequest(BaseModel):
    bad: str
    good: str


def call_space(api_name: str, data: list, retries: int = 3):
    post_url = f"{HF_SPACE_URL}/gradio_api/call/{api_name}"

    for attempt in range(retries):
        try:
            r = requests.post(
                post_url,
                json={"data": data},
                timeout=TIMEOUT_POST
            )
            r.raise_for_status()

            event_id = r.json().get("event_id")
            if not event_id:
                raise Exception(f"No event_id returned: {r.text}")

            result_url = f"{post_url}/{event_id}"

            final_payload = None
            last_activity = time.time()

            with requests.get(result_url, stream=True, timeout=None) as res:
                res.raise_for_status()

                for line in res.iter_lines():
                    if time.time() - last_activity > IDLE_TIMEOUT:
                        raise Exception("Stream stalled (idle timeout)")

                    if not line:
                        continue

                    decoded = line.decode()

                    if not decoded.startswith("data:"):
                        continue

                    payload = decoded.replace("data:", "").strip()

                    if payload == "[DONE]":
                        break

                    if payload == "null":
                        continue

                    try:
                        parsed = json.loads(payload)
                        final_payload = parsed
                        last_activity = time.time()
                    except Exception:
                        continue

            if final_payload is None:
                raise Exception("No valid payload received from stream")

            if isinstance(final_payload, dict) and "data" in final_payload:
                return final_payload["data"]

            return final_payload

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=502, detail="HF Space unavailable")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "space": HF_SPACE_URL
    }


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return {"translated": ""}

    result = call_space("translate", [text])

    translated = None

    if isinstance(result, list) and result:
        translated = result[0]
    elif isinstance(result, str):
        translated = result

    if not translated:
        return {
            "translated": "",
            "warning": "Model returned empty output"
        }

    return {"translated": translated}


@app.post("/translate/batch")
def translate_batch(req: BatchRequest):
    texts = [item.text for item in req.items if item.text.strip()]

    if not texts:
        return {"translations": []}

    payload = json.dumps(texts)

    result = call_space("translate_batch", [payload])

    if isinstance(result, list):
        return {"translations": result}

    return {"translations": []}


@app.post("/teach")
def teach(req: TeachRequest):
    if not req.bad.strip() or not req.good.strip():
        raise HTTPException(status_code=400, detail="Invalid input")

    result = call_space("teach", [req.bad, req.good])

    status = None

    if isinstance(result, list) and result:
        status = result[0]
    elif isinstance(result, str):
        status = result

    return {
        "status": status or "ok"
    }