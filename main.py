from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import logging

# ---------------- CONFIG ----------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"
API_KEY = os.getenv("TRANSLATION_API_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- APP ----------------

app = FastAPI(title="Translation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------

tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        model.to(device)
        model.eval()

        logger.info(f"Model loaded on {device}")


@app.on_event("startup")
def startup():
    load_model()

# ---------------- SCHEMAS ----------------


class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "eng_Latn"
    tgt_lang: str = "ibo_Latn"


class BatchRequest(BaseModel):
    items: list[TranslateRequest]


# ---------------- CORE ----------------


def run_translation(texts, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
# ---------------- ROUTES ----------------


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.text.strip():
        return {
            "translated": req.text,
            "src_lang": req.src_lang,
            "tgt_lang": req.tgt_lang,
        }

    try:
        result = run_translation([req.text], req.src_lang, req.tgt_lang)[0]
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")

    return {
        "translated": result,
        "src_lang": req.src_lang,
        "tgt_lang": req.tgt_lang,
    }


@app.post("/translate/batch")
def translate_batch(req: BatchRequest, x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.items:
        return {"translations": []}

    texts = [item.text for item in req.items if item.text.strip()]

    try:
        results = run_translation(
            texts,
            req.items[0].src_lang,
            req.items[0].tgt_lang,
        )
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail="Batch failed")

    return {"translations": results}