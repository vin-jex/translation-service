from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
import time

# -------- CONFIG --------

API_KEY = os.getenv("TRANSLATION_API_KEY", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://vinjex-translation-ml.hf.space")

TIMEOUT_POST = 60
TIMEOUT_STREAM = 120

# -------- APP INIT --------

app = FastAPI(title="LearnSci Translation Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# -------- SCHEMAS --------

class TranslateRequest(BaseModel):
    text: str

class BatchItem(BaseModel):
    text: str

class BatchRequest(BaseModel):
    items: list[BatchItem]

class TeachRequest(BaseModel):
    bad: str
    good: str

# -------- CORE: CALL HF SPACE --------

def call_space(api_name: str, data: list, retries: int = 3):
    post_url = f"{HF_SPACE_URL}/gradio_api/call/{api_name}"

    for attempt in range(retries):
        try:
            # STEP 1: submit job
            r = requests.post(
                post_url,
                json={"data": data},
                timeout=TIMEOUT_POST
            )
            r.raise_for_status()

            event_id = r.json().get("event_id")
            if not event_id:
                raise Exception("No event_id returned")

            # STEP 2: stream result
            result_url = f"{post_url}/{event_id}"

            with requests.get(result_url, stream=True, timeout=TIMEOUT_STREAM) as res:
                res.raise_for_status()

                for line in res.iter_lines():
                    if not line:
                        continue

                    decoded = line.decode()

                    if decoded.startswith("data:"):
                        payload = decoded.replace("data:", "").strip()
                        return json.loads(payload)

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=502, detail="HF Space unavailable")

# -------- ROUTES --------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "space": HF_SPACE_URL
    }

@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = req.text.strip()
    if not text:
        return {"translated": ""}

    result = call_space("translate", [text])

    return {
        "translated": result[0] if isinstance(result, list) else result
    }

@app.post("/translate/batch")
def translate_batch(req: BatchRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    texts = [item.text for item in req.items if item.text.strip()]

    if not texts:
        return {"translations": []}

    # NOTE: your Gradio app expects JSON string input
    payload = json.dumps(texts)

    result = call_space("translate_batch", [payload])

    return {
        "translations": result if isinstance(result, list) else []
    }

@app.post("/teach")
def teach(req: TeachRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.bad.strip() or not req.good.strip():
        raise HTTPException(status_code=400, detail="Invalid input")

    result = call_space("teach", [req.bad, req.good])

    return {
        "status": result[0] if isinstance(result, list) else "ok"
    }