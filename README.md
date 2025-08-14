# AI Chatbot - React + FastAPI

Chatbot ที่มีหน้าบ้านเป็น React และหลังบ้านเป็น FastAPI โดยใช้โทนสีดำและเหลือง

## คุณสมบัติ

- 🎨 UI สวยงามด้วยโทนสีดำและเหลือง
- 💬 ระบบแชทแบบ Real-time
- 📱 Responsive Design รองรับทุกขนาดหน้าจอ
- 🚀 FastAPI Backend ที่รวดเร็ว
- ⚡ React Frontend ที่ทันสมัย

## การติดตั้ง

### Backend (FastAPI)

1. ติดตั้ง Python dependencies:
```bash
pip install -r requirements.txt
```

2. ตั้งค่า OpenAI API (ไม่บังคับ):
   - สร้างไฟล์ `.env` ในโฟลเดอร์หลัก
   - เพิ่ม OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   ```
   - หากไม่ตั้งค่า chatbot จะใช้โหมด Basic Mode

2. รัน FastAPI server:
```bash
cd backend
python main.py
```

Backend จะรันที่ `http://localhost:8000`

### Frontend (React)

1. ติดตั้ง Node.js dependencies:
```bash
cd frontend
npm install
```

2. รัน React development server:
```bash
npm run dev
```

Frontend จะรันที่ `http://localhost:3000`

## การใช้งาน

1. เปิดเบราว์เซอร์ไปที่ `http://localhost:3000`
2. เริ่มแชทกับ AI Chatbot
3. พิมพ์ข้อความและกด Enter หรือคลิกปุ่มส่ง

## API Endpoints

- `GET /` - Health check
- `POST /chat` - ส่งข้อความและรับการตอบกลับ
- `GET /health` - สถานะของ API
- `GET /openai-status` - สถานะการเชื่อมต่อ OpenAI

## โครงสร้างโปรเจค

```
AI MDJ/
├── backend/
│   └── main.py          # FastAPI server
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js       # React component หลัก
│   │   ├── App.css      # CSS styles
│   │   └── index.js     # Entry point
│   └── package.json
├── requirements.txt      # Python dependencies
└── README.md
```

## เทคโนโลยีที่ใช้

- **Backend**: FastAPI, Python, OpenAI API
- **Frontend**: React, CSS3
- **Styling**: Custom CSS with black & yellow theme
- **Communication**: REST API
- **AI**: OpenAI GPT-3.5-turbo (optional)

## การพัฒนาเพิ่มเติม

- เพิ่มระบบ Authentication
- เชื่อมต่อกับ AI models อื่นๆ
- เพิ่มระบบเก็บประวัติการแชท
- เพิ่มการรองรับไฟล์แนบ
- เพิ่มระบบแจ้งเตือน

## License

MIT License
