from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
import time

API_KEY = os.getenv("TRANSLATION_API_KEY")
HF_API_URL = "https://vinjex-translation-ml.hf.space/run/predict"

app = FastAPI(title="Translation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    items: list[TranslateRequest]


def call_hf(payload, retries=3):
    for i in range(retries):
        try:
            r = requests.post(HF_API_URL, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        time.sleep(2 ** i)
    raise HTTPException(status_code=502, detail="HF Space unavailable")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = req.text if isinstance(req.text, str) else ""

    if not text.strip():
        return {"translated": text}

    result = call_hf({
        "data": [text],
        "fn_index": 0
    })

    return {"translated": result["data"][0]}


@app.post("/translate/batch")
def translate_batch(req: BatchRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    texts = [i.text for i in req.items if isinstance(i.text, str)]

    if not texts:
        return {"translations": []}

    result = call_hf({
        "data": [json.dumps(texts)],
        "fn_index": 1
    })

    return {"translations": result["data"][0]}