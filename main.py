from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import spacy
import os
from typing import Optional
import logging

# AI/LLM components
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import requests

# Speech processing
import speech_recognition as sr
from gtts import gTTS
import io
import base64

app = FastAPI(title="LAA - Legal Accompanying Agent",
              description="AI-powered legal assistance for Tamil Nadu and Andhra Pradesh")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

# Load dataset
DATA_PATH = Path(__file__).parent / "data/laachat_dataset_sample.json"
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
except FileNotFoundError:
    logger.error(f"Dataset file not found at {DATA_PATH}")
    DATA = []

# Initialize AI models (with lazy loading)
class AIModels:
    _instance = None

    def __init__(self):
        self.llm = None
        self.llm_tokenizer = None
        self.stt_model = None
        self.translation_models = {}
        self.qa_pipeline = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_llm(self):
        if not self.llm:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                model_name = "facebook/bart-base"  # Smaller alternative that doesn't need sentencepiece
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                logger.info("Loaded BART model (no sentencepiece needed)")
            except Exception as e:
                logger.error(f"Failed to load LLM: {str(e)}")

    def load_stt(self):
        if not self.stt_model:
            try:
                # Using AI4Bharat's IndicSTT would be ideal here
                self.stt_model = sr.Recognizer()
                logger.info("Initialized speech recognition")
            except Exception as e:
                logger.error(f"Failed to initialize STT: {str(e)}")

    def load_translation(self, lang):
        if lang not in self.translation_models:
            try:
                if lang == "ta":
                    model_name = "ai4bharat/indictrans2-ta-en"
                elif lang == "te":
                    model_name = "ai4bharat/indictrans2-te-en"
                else:
                    return

                self.translation_models[lang] = pipeline(
                    "translation",
                    model=model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info(f"Loaded translation model for {lang}")
            except Exception as e:
                logger.error(f"Failed to load translation model for {lang}: {str(e)}")

# Request schemas
class QueryRequest(BaseModel):
    query: str
    lang: str = "en"
    use_llm: bool = False

class VoiceQueryRequest(BaseModel):
    audio: bytes
    lang: str = "en"
    use_llm: bool = False

# Initialize AI models
ai_models = AIModels.get_instance()

def initialize_models():
    """Lazy loading of models on first request"""
    ai_models.load_llm()
    ai_models.load_stt()
    ai_models.load_translation("ta")
    ai_models.load_translation("te")

# Hugging Face API call for LLm response
def query_huggingface_bart(input_text: str) -> str:
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise ValueError("HF_API_TOKEN not set in environment variables")

    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    json_data = {
        "inputs": input_text,
        "parameters": {
            "max_length": 200
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-base",
        headers=headers,
        json=json_data
    )

    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error: {response.text}")

    output = response.json()
    return output[0]["generated_text"] if isinstance(output, list) else output


# Enhanced retrieval with semantic similarity fallback
def retrieve_answer(query: str, lang: str = "en", use_llm: bool = False):
    ...
    if use_llm:
        input_text = f"Answer this legal question for India concisely: {query}"
        llm_response = query_huggingface_bart(input_text)
        ...

# Enhanced retrieval with semantic similarity fallback
def retrieve_answer(query: str, lang: str = "en", use_llm: bool = False):
    # First try exact match
    for entry in DATA:
        question_in_lang = entry["query"].get(lang, "").strip().lower()
        if question_in_lang == query.strip().lower():
            return entry["response"], entry["translated_response"].get(lang, {})

    # If LLM is requested and available, use it
    if use_llm and ai_models.llm:
        try:
            # Translate to English if needed
            if lang != "en":
                ai_models.load_translation(lang)
                if use_llm:
                    input_text = f"Answer this legal question for India concisely: {query}"
                    llm_response = query_huggingface_bart(input_text)

            # Generate response with LLM
            input_text = f"Answer this legal question for India concisely: {query}"
            inputs = ai_models.llm_tokenizer(input_text, return_tensors="pt")
            outputs = ai_models.llm.generate(**inputs, max_length=200)
            llm_response = ai_models.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "description": llm_response,
                "section": None,
                "substantive_guidance": "This is an AI-generated response. For official legal advice, please consult a lawyer.",
                "lawyer_panel": "Panel: Legal Aid Team"
            }, {}
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")

    # Fallback response
    return {
        "description": "Sorry, I couldn't find a match for your query.",
        "section": None,
        "substantive_guidance": "You may want to consult a human lawyer for this matter.",
        "lawyer_panel": "Panel: Legal Aid Team"
    }, {}

# Text-to-speech generation
def generate_tts(text: str, lang: str = "en") -> Optional[str]:
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return base64.b64encode(audio_bytes.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        return None

# Speech-to-text processing
def process_stt(audio_data: bytes, lang: str = "en") -> Optional[str]:
    try:
        ai_models.load_stt()
        if not ai_models.stt_model:
            return None

        audio_file = io.BytesIO(audio_data)
        with sr.AudioFile(audio_file) as source:
            audio = ai_models.stt_model.record(source)

        # For production, replace with AI4Bharat's IndicSTT
        if lang == "en":
            text = ai_models.stt_model.recognize_google(audio)
        elif lang == "ta":
            text = ai_models.stt_model.recognize_google(audio, language="ta-IN")
        elif lang == "te":
            text = ai_models.stt_model.recognize_google(audio, language="te-IN")
        else:
            text = ai_models.stt_model.recognize_google(audio)

        return text
    except Exception as e:
        logger.error(f"STT processing failed: {str(e)}")
        return None

# Endpoints
@app.post("/query")
async def ask_question(body: QueryRequest):
    initialize_models()
    response, translated = retrieve_answer(body.query, body.lang, body.use_llm)

    result = {
        "query": body.query,
        "response": {
            "en": {
                "section": response.get("section"),
                "description": response.get("description"),
                "cultural_reference": response.get("cultural_reference"),
                "substantive_guidance": response.get("substantive_guidance"),
                "lawyer_panel": response.get("lawyer_panel")
            }
        }
    }

    if body.lang != "en":
        result["response"][body.lang] = {
            "description": translated.get("description", response.get("description")),
            "cultural_reference": translated.get("cultural_reference", ""),
            "substantive_guidance": translated.get("substantive_guidance", "")
        }

    return result

@app.post("/voice_query")
async def ask_question_with_voice(audio: UploadFile = File(...), lang: str = "en", use_llm: bool = False):
    initialize_models()

    audio_data = await audio.read()
    query_text = process_stt(audio_data, lang)

    if not query_text:
        raise HTTPException(status_code=400, detail="Could not process audio")

    response, translated = retrieve_answer(query_text, lang, use_llm)

    # Generate audio response
    response_text = translated.get("description", response.get("description"))
    audio_response = generate_tts(response_text, lang) if response_text else None

    result = {
        "query": query_text,
        "response": {
            "en": {
                "section": response.get("section"),
                "description": response.get("description"),
                "cultural_reference": response.get("cultural_reference"),
                "substantive_guidance": response.get("substantive_guidance"),
                "lawyer_panel": response.get("lawyer_panel")
            },
            "audio_response": audio_response
        }
    }

    if lang != "en":
        result["response"][lang] = {
            "description": translated.get("description", response.get("description")),
            "cultural_reference": translated.get("cultural_reference", ""),
            "substantive_guidance": translated.get("substantive_guidance", "")
        }

    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": {
        "llm": ai_models.llm is not None,
        "stt": ai_models.stt_model is not None,
        "translation_ta": "ta" in ai_models.translation_models,
        "translation_te": "te" in ai_models.translation_models
    }}
