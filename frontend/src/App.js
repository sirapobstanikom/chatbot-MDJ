import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "สวัสดีครับ! ยินดีต้อนรับสู่ AI Chatbot ของเรา 😊",
      isUser: false,
      timestamp: new Date().toLocaleTimeString('th-TH')
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [openaiStatus, setOpenaiStatus] = useState('checking');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check OpenAI status on component mount
  useEffect(() => {
    checkOpenAIStatus();
  }, []);

  const checkOpenAIStatus = async () => {
    try {
      const response = await fetch('/openai-status');
      if (response.ok) {
        const data = await response.json();
        setOpenaiStatus(data.openai_available ? 'connected' : 'not_configured');
      } else {
        setOpenaiStatus('error');
      }
    } catch (error) {
      setOpenaiStatus('error');
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const currentMessage = inputMessage; // Store the current message
    const userMessage = {
      id: Date.now(),
      text: currentMessage,
      isUser: true,
      timestamp: new Date().toLocaleTimeString('th-TH')
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: currentMessage }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          text: data.response,
          isUser: false,
          timestamp: new Date().toLocaleTimeString('th-TH')
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อ กรุณาลองใหม่อีกครั้ง",
        timestamp: new Date().toLocaleTimeString('th-TH')
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="chat-header">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <h1>AI Chatbot</h1>
          </div>
          <div className="status">
            <span className="status-dot"></span>
            Online
            {openaiStatus === 'connected' && (
              <span className="openai-status">• OpenAI</span>
            )}
            {openaiStatus === 'not_configured' && (
              <span className="openai-status-warning">• Basic Mode</span>
            )}
          </div>
        </div>

        <div className="chat-messages">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.isUser ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                <div className="message-text">{message.text}</div>
                <div className="message-timestamp">{message.timestamp}</div>
              </div>
              <div className="message-avatar">
                {message.isUser ? '👤' : '🤖'}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
              <div className="message-avatar">🤖</div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="พิมพ์ข้อความของคุณที่นี่..."
              rows="1"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="send-button"
            >
              <span className="send-icon">➤</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
