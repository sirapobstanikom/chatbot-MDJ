from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
from dotenv import load_dotenv

# === OpenAI SDK v1 ===
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

import anyio
import asyncio
import datetime

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
try:
    load_dotenv(override=True)
    print("✅ Environment variables loaded successfully")
except Exception as e:
    print(f"❌ Error loading .env file: {e}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if OPENAI_API_KEY:
    print(f"✅ OpenAI API Key: {OPENAI_API_KEY[:20]}...")
    print(f"✅ OpenAI Model: {OPENAI_MODEL}")
else:
    print("❌ No OpenAI API key found")
    print("⚠️  Running in Basic Mode")

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Chatbot API", version="1.2.0 (memory+autosummary)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

# -----------------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "คุณเป็น AI Chatbot ที่เป็นมิตรและช่วยเหลือผู้ใช้ "
    "ตอบคำถามเป็นภาษาไทยหรือภาษาอังกฤษตามที่ผู้ใช้ถาม "
    "ให้คำตอบที่ถูกต้อง กระชับ และเป็นประโยชน์ "
    "ใช้โทนเสียงที่เป็นมิตรและสุภาพ"
)

# -----------------------------------------------------------------------------
# In-memory stores
# -----------------------------------------------------------------------------
CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}
STORE_LOCK = asyncio.Lock()

# ขีดจำกัดความยาว/การตัดทอน
MAX_TURNS = 12                  # เก็บรอบล่าสุด (user+assistant ไม่รวม system)
SOFT_CHAR_LIMIT = 12000         # ถ้าความยาวรวมเกิน จะสรุปให้สั้นลงก่อนเรียกโมเดล
KEEP_RECENT_TURNS_AFTER_SUM = 4 # เก็บท้ายๆ ไว้หลังสรุป

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _init_session_if_needed(session_id: str):
    if session_id not in CHAT_STORE:
        CHAT_STORE[session_id] = [{"role": "system", "content": SYSTEM_MESSAGE}]

def _trim_turns(session_id: str):
    msgs = CHAT_STORE.get(session_id, [])
    if not msgs:
        return
    system = msgs[0:1]
    turns = msgs[1:]
    if len(turns) > MAX_TURNS:
        turns = turns[-MAX_TURNS:]
    CHAT_STORE[session_id] = system + turns

def _total_chars(session_id: str) -> int:
    return sum(len(m.get("content", "")) for m in CHAT_STORE.get(session_id, []))

def resolve_session_id(body_sid: Optional[str], req: Request) -> str:
    return (body_sid or "").strip() or (req.headers.get("X-Session-Id", "").strip()) or "default"

def _summary_prompt(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    สร้าง prompt ให้โมเดลสรุปบทสนทนาที่ผ่านมา (ย่อให้สั้น ชัด และใช้งานต่อได้)
    """
    # เอาเฉพาะ user/assistant (ข้าม system ตัวแรก)
    lines = []
    for m in history[1:]:
        role = m["role"]
        if role in ("user", "assistant"):
            lines.append(f"{role.upper()}: {m['content']}")
    joined = "\n".join(lines)[:8000]  # safety cut
    return [
        {"role": "system", "content": "คุณคือผู้ช่วยสรุปบทสนทนาอย่างย่อเพื่อเก็บเป็นบริบทถาวร ใช้ภาษาไทยแบบกระชับ ชัดเจน เป็น bullet/หัวข้อสำคัญ พร้อมข้อสรุป/ข้อเท็จจริง/ข้อกำหนดที่ต้องจำต่อไป"},
        {"role": "user", "content": f"สรุปบทสนทนาต่อไปนี้ให้สั้น กระชับ ชัดเจน (ไม่เกิน 12 บรรทัด) และเน้นสิ่งที่ต้องจำเพื่อคุยต่อ:\n\n{joined}"}
    ]

def _insert_summary_as_system(session_id: str, summary_text: str):
    """
    รีเซ็ตแชทให้มี system เดิม + system สรุป + เก็บท้ายๆ อีกนิด
    """
    msgs = CHAT_STORE.get(session_id, [])
    if not msgs:
        _init_session_if_needed(session_id)
        msgs = CHAT_STORE[session_id]

    system = msgs[0:1]
    turns = msgs[1:]
    keep_tail = turns[-(KEEP_RECENT_TURNS_AFTER_SUM*2):]  # ประมาณ user/assistant สลับกัน

    summarized = system + [
        {"role": "system", "content": f"[สรุปบริบทก่อนหน้า]\n{summary_text.strip()}"}
    ] + keep_tail

    CHAT_STORE[session_id] = summarized
    _trim_turns(session_id)

def _compress_if_needed(session_id: str):
    """
    ถ้าความยาวเกิน SOFT_CHAR_LIMIT จะสรุปและย่อก่อน (หนึ่งครั้งต่อการเรียก)
    """
    if _total_chars(session_id) <= SOFT_CHAR_LIMIT:
        return

    history = CHAT_STORE.get(session_id, [])
    if not history or len(history) < 2:
        return

    # เรียกโมเดลให้สรุป
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=_summary_prompt(history),
            max_tokens=320,
            temperature=0.2,
        )
        summary = (resp.choices[0].message.content or "").strip()
        _insert_summary_as_system(session_id, summary)
    except Exception as e:
        # ถ้าสรุปไม่สำเร็จ จะไม่ล้ม — แค่บีบด้วยการเหลือท้ายๆ
        msgs = CHAT_STORE.get(session_id, [])
        system = msgs[0:1]
        turns = msgs[1:]
        CHAT_STORE[session_id] = system + turns[-(KEEP_RECENT_TURNS_AFTER_SUM*2):]
        _trim_turns(session_id)

# -----------------------------------------------------------------------------
# OpenAI call with retry-on-context
# -----------------------------------------------------------------------------
def _call_openai(session_id: str) -> str:
    if not client:
        return "ขออภัยครับ ยังไม่ได้ตั้งค่า OpenAI API key กรุณาตั้งค่าก่อนใช้งาน"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=CHAT_STORE[session_id],
            max_tokens=500,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except APIError as e:
        msg = str(e)
        # ถ้าเกินขนาดคอนเท็กซ์ ให้สรุปแล้วยิงใหม่หนึ่งครั้ง
        if "maximum context length" in msg.lower() or "too many tokens" in msg.lower():
            _compress_if_needed(session_id)
            resp2 = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=CHAT_STORE[session_id],
                max_tokens=500,
                temperature=0.7,
            )
            return (resp2.choices[0].message.content or "").strip()
        raise
    except AuthenticationError:
        return "ขออภัยครับ OpenAI API key ไม่ถูกต้อง"
    except RateLimitError:
        return "ขออภัยครับ เกินขีดจำกัดการใช้งาน OpenAI API กรุณาลองใหม่ภายหลัง"
    except Exception as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}"

async def get_ai_reply(session_id: str) -> str:
    return await anyio.to_thread.run_sync(_call_openai, session_id)

# -----------------------------------------------------------------------------
# Fallback
# -----------------------------------------------------------------------------
def fallback_reply(user_message: str) -> str:
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

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Chatbot API (memory+autosummary) is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/openai-status")
async def openai_status():
    if not client:
        return {"status": "not_configured", "message": "OpenAI API key ไม่ได้ตั้งค่า", "openai_available": False}
    try:
        _ = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return {"status": "connected", "message": "เชื่อมต่อ OpenAI สำเร็จ", "openai_available": True, "model": OPENAI_MODEL}
    except Exception as e:
        return {"status": "error", "message": f"เกิดข้อผิดพลาด: {str(e)}", "openai_available": False}

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    async with STORE_LOCK:
        _init_session_if_needed(session_id)
        return HistoryResponse(session_id=session_id, messages=CHAT_STORE[session_id])

@app.delete("/history/{session_id}")
async def reset_history(session_id: str):
    async with STORE_LOCK:
        CHAT_STORE[session_id] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    return {"status": "reset", "session_id": session_id}

@app.post("/chat", response_model=ChatResponse)
async def chat(req_body: ChatRequest, req: Request):
    try:
        session_id = resolve_session_id(req_body.session_id, req)

        async with STORE_LOCK:
            _init_session_if_needed(session_id)
            CHAT_STORE[session_id].append({"role": "user", "content": req_body.message})
            _trim_turns(session_id)
            _compress_if_needed(session_id)

        try:
            reply = await get_ai_reply(session_id)
        except Exception:
            reply = fallback_reply(req_body.message)

        async with STORE_LOCK:
            CHAT_STORE[session_id].append({"role": "assistant", "content": reply})
            _trim_turns(session_id)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ChatResponse(response=reply, timestamp=ts, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # โปรดักชันควรใช้ process manager (เช่น uvicorn/gunicorn) ภายนอกไฟล์นี้
    uvicorn.run(app, host="0.0.0.0", port=8000)
