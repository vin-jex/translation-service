from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
import time

API_KEY = os.getenv("TRANSLATION_API_KEY", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://vinjex-translation-ml.hf.space")

app = FastAPI(title="LearnSci Translation Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "eng_Latn"
    tgt_lang: str = "ibo_Latn"


class BatchRequest(BaseModel):
    items: list[TranslateRequest]


def call_space(fn_index: int, data: list, retries: int = 3) -> dict:
    url = f"{HF_SPACE_URL}/run/predict"
    payload = {"fn_index": fn_index, "data": data}
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"Space returned {r.status_code}: {r.text}")
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
        time.sleep(2 ** i)
    raise HTTPException(status_code=502, detail="HF Space unavailable")


@app.get("/health")
def health():
    return {"status": "ok", "space": HF_SPACE_URL}


@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not req.text.strip():
        return {"translated": req.text}

    # fn_index 0 = translate_single
    result = call_space(0, [req.text, req.src_lang, req.tgt_lang])
    return {"translated": result["data"][0]}


@app.post("/translate/batch")
def translate_batch(req: BatchRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    texts = [i.text for i in req.items]
    if not texts:
        return {"translations": []}

    # fn_index 1 = translate_batch
    result = call_space(1, [json.dumps(texts)])
    translations = json.loads(result["data"][0])
    return {"translations": translations}