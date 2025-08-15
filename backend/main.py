from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from dotenv import load_dotenv

# === OpenAI SDK v1 ===
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

import anyio
import datetime

# Load environment variables
try:
    load_dotenv(override=True)
    print("✅ Environment variables loaded successfully")
except Exception as e:
    print(f"❌ Error loading .env file: {e}")

# Configure OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # ใช้รุ่นใหม่ที่เร็วและประหยัดกว่า
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Debug: Print environment variables
if OPENAI_API_KEY:
    print(f"✅ OpenAI API Key: {OPENAI_API_KEY[:20]}...")
    print(f"✅ OpenAI Model: {OPENAI_MODEL}")
else:
    print("❌ No OpenAI API key found")
    print("⚠️  Running in Basic Mode")

app = FastAPI(title="Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    is_user: bool

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

SYSTEM_MESSAGE = (
    "คุณเป็น AI Chatbot ที่เป็นมิตรและช่วยเหลือผู้ใช้ "
    "ตอบคำถามเป็นภาษาไทยหรือภาษาอังกฤษตามที่ผู้ใช้ถาม "
    "ให้คำตอบที่ถูกต้อง กระชับ และเป็นประโยชน์ "
    "ใช้โทนเสียงที่เป็นมิตรและสุภาพ"
)

# ---- OpenAI-powered chatbot logic (OpenAI SDK v1) ----
def _call_openai(user_message: str) -> str:
    if not client:
        return "ขออภัยครับ ยังไม่ได้ตั้งค่า OpenAI API key กรุณาตั้งค่าก่อนใช้งาน"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except AuthenticationError:
        return "ขออภัยครับ เกิดข้อผิดพลาดในการยืนยันตัวตน OpenAI API key ไม่ถูกต้อง"
    except RateLimitError:
        return "ขออภัยครับ เกินขีดจำกัดการใช้งาน OpenAI API กรุณาลองใหม่ในภายหลัง"
    except APIError as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดจาก OpenAI API: {str(e)}"
    except Exception as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"

# ทำเป็น async แบบไม่บล็อก event loop
async def get_openai_response(user_message: str) -> str:
    return await anyio.to_thread.run_sync(_call_openai, user_message)

# ---- Fallback chatbot logic ----
def get_fallback_response(user_message: str) -> str:
    low = user_message.lower()
    if "สวัสดี" in low or "hello" in low:
        return "สวัสดีครับ! ยินดีต้อนรับสู่ AI Chatbot ของเรา 😊"
    elif "ชื่ออะไร" in low or "what's your name" in low:
        return "ฉันชื่อ AI Chatbot ครับ! ยินดีที่ได้รู้จักคุณ"
    elif "ช่วยเหลือ" in low or "help" in low:
        return "ฉันสามารถช่วยตอบคำถามทั่วไปได้ครับ ลองถามอะไรก็ได้!"
    elif "ขอบคุณ" in low or "thank" in low:
        return "ยินดีครับ! มีอะไรให้ช่วยอีกไหมครับ?"
    elif "ลาก่อน" in low or "bye" in low:
        return "ลาก่อนครับ! ขอให้มีวันที่ดีนะครับ 👋"
    else:
        return "ขออภัยครับ ฉันยังไม่เข้าใจคำถามนี้ ลองถามใหม่ได้ไหมครับ?"

@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        try:
            response = await get_openai_response(request.message)
        except Exception:
            response = get_fallback_response(request.message)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ChatResponse(response=response, timestamp=timestamp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/openai-status")
async def openai_status():
    if not client:
        return {
            "status": "not_configured",
            "message": "OpenAI API key ไม่ได้ตั้งค่า",
            "openai_available": False,
        }
    try:
        # ping แบบสั้น ๆ
        _ = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return {
            "status": "connected",
            "message": "เชื่อมต่อ OpenAI สำเร็จ",
            "openai_available": True,
            "model": OPENAI_MODEL,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "openai_available": False,
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
