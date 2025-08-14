from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from dotenv import load_dotenv
import openai
load_dotenv()
# Load environment variables
try:
    load_dotenv(override=True)
    print("✅ Environment variables loaded successfully")
except Exception as e:
    print(f"❌ Error loading .env file: {e}")

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Debug: Print environment variables
if openai.api_key:
    print(f"✅ OpenAI API Key: {openai.api_key[:20]}...")
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

# OpenAI-powered chatbot logic
async def get_openai_response(user_message: str) -> str:
    try:
        if not openai.api_key:
            return "ขออภัยครับ ยังไม่ได้ตั้งค่า OpenAI API key กรุณาติดตั้งก่อนใช้งาน"
        
        # Create system message for Thai chatbot
        system_message = """คุณเป็น AI Chatbot ที่เป็นมิตรและช่วยเหลือผู้ใช้ 
        ตอบคำถามเป็นภาษาไทยหรือภาษาอังกฤษตามที่ผู้ใช้ถาม 
        ให้คำตอบที่ถูกต้อง กระชับ และเป็นประโยชน์ 
        ใช้โทนเสียงที่เป็นมิตรและสุภาพ"""
        
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.error.AuthenticationError:
        return "ขออภัยครับ เกิดข้อผิดพลาดในการยืนยันตัวตน OpenAI API key ไม่ถูกต้อง"
    except openai.error.RateLimitError:
        return "ขออภัยครับ เกินขีดจำกัดการใช้งาน OpenAI API กรุณาลองใหม่ในภายหลัง"
    except openai.error.APIError as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดจาก OpenAI API: {str(e)}"
    except Exception as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"

# Fallback chatbot logic for basic responses
def get_fallback_response(user_message: str) -> str:
    user_message = user_message.lower()
    
    if "สวัสดี" in user_message or "hello" in user_message:
        return "สวัสดีครับ! ยินดีต้อนรับสู่ AI Chatbot ของเรา 😊"
    elif "ชื่ออะไร" in user_message or "what's your name" in user_message:
        return "ฉันชื่อ AI Chatbot ครับ! ยินดีที่ได้รู้จักคุณ"
    elif "ช่วยเหลือ" in user_message or "help" in user_message:
        return "ฉันสามารถช่วยตอบคำถามทั่วไปได้ครับ ลองถามอะไรก็ได้!"
    elif "ขอบคุณ" in user_message or "thank" in user_message:
        return "ยินดีครับ! มีอะไรให้ช่วยอีกไหมครับ?"
    elif "ลาก่อน" in user_message or "bye" in user_message:
        return "ลาก่อนครับ! ขอให้มีวันที่ดีนะครับ 👋"
    else:
        return "ขออภัยครับ ฉันยังไม่เข้าใจคำถามนี้ ลองถามใหม่ได้ไหมครับ?"

@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        import datetime
        
        # Try OpenAI first, fallback to basic responses if it fails
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
    try:
        if not openai.api_key:
            return {
                "status": "not_configured",
                "message": "OpenAI API key ไม่ได้ตั้งค่า",
                "openai_available": False
            }
        
        # Test OpenAI connection with a simple request
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        return {
            "status": "connected",
            "message": "เชื่อมต่อ OpenAI สำเร็จ",
            "openai_available": True,
            "model": OPENAI_MODEL
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "openai_available": False
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
