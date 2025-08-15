from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# ------- Load environment -------
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------- OpenAI (new SDK) -------
# pip install --upgrade openai>=1.30.0 python-dotenv
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    client = None

app = FastAPI(title="Chatbot API", version="1.0.0")

# ------- CORS -------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # เพิ่ม origin อื่นได้ตามต้องการ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# ------- OpenAI-powered chatbot logic -------
async def get_openai_response(user_message: str) -> str:
    if not OPENAI_API_KEY or client is None:
        return "ขออภัยครับ ยังไม่ได้ตั้งค่า OpenAI API key กรุณาตั้งค่าก่อนใช้งาน"

    system_message = (
        "คุณเป็น AI Chatbot ที่เป็นมิตรและช่วยเหลือผู้ใช้ "
        "ตอบคำถามเป็นภาษาไทยหรือภาษาอังกฤษตามที่ผู้ใช้ถาม "
        "ให้คำตอบที่ถูกต้อง กระชับ และเป็นประโยชน์ ใช้โทนสุภาพ"
    )

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # ตีกลับข้อความ error แบบอ่านง่าย
        return f"ขออภัยครับ เกิดข้อผิดพลาดจาก OpenAI: {str(e)}"

# ------- Fallback -------
def get_fallback_response(user_message: str) -> str:
    t = user_message.lower()
    if "สวัสดี" in t or "hello" in t:
        return "สวัสดีครับ! ยินดีต้อนรับสู่ AI Chatbot 😊"
    if "ชื่ออะไร" in t or "what's your name" in t:
        return "ฉันชื่อ AI Chatbot ครับ!"
    if "ช่วยเหลือ" in t or "help" in t:
        return "ฉันช่วยตอบคำถามทั่วไปได้ครับ ลองถามมาเลย!"
    if "ขอบคุณ" in t or "thank" in t:
        return "ยินดีครับ! มีอะไรให้ช่วยอีกไหม?"
    if "ลาก่อน" in t or "bye" in t:
        return "ลาก่อนครับ! ขอให้มีวันที่ดี 👋"
    return "ขออภัยครับ ฉันยังไม่เข้าใจคำถามนี้ ลองถามใหม่ได้ไหมครับ?"

# ------- Endpoints -------
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    import datetime
    try:
        try:
            response = await get_openai_response(request.message)
        except Exception:
            response = get_fallback_response(request.message)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ChatResponse(response=response, timestamp=ts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/openai-status")
async def openai_status():
    if not OPENAI_API_KEY or client is None:
        return {
            "status": "not_configured",
            "message": "OpenAI API key ไม่ได้ตั้งค่า",
            "openai_available": False,
        }
    try:
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
    import uvicorn
    # ตรงนี้ห้ามมีตัวอักษรอื่นปะปน (ลบ "เชคที" ออก)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
