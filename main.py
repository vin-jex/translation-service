from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

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


@app.get("/health")
def health():
    return {"status": "ok", "provider": "huggingface-space"}


@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = req.text if isinstance(req.text, str) else ""

    if not text.strip():
        return {"translated": text}

    try:
        response = requests.post(
            HF_API_URL,
            json={
                "data": [text],
                "fn_index": 0
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Upstream error")

        result = response.json()

        translated = result["data"][0]

        return {"translated": translated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch")
def translate_batch(req: BatchRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    texts = [
        item.text for item in req.items
        if isinstance(item.text, str)
    ]

    if not texts:
        return {"translations": []}

    try:
        response = requests.post(
            HF_API_URL,
            json={
                "data": [json.dumps(texts)],
                "fn_index": 1
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Upstream error")

        result = response.json()

        translations = result["data"][0]

        return {"translations": translations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))