@echo off
echo Setting up OpenAI API for Chatbot...
echo.

if not exist .env (
    echo Creating .env file...
    echo OPENAI_API_KEY=your_openai_api_key_here > .env
    echo OPENAI_MODEL=gpt-3.5-turbo >> .env
    echo.
    echo .env file created successfully!
    echo Please edit .env file and add your actual OpenAI API key
) else (
    echo .env file already exists!
)

echo.
echo To get OpenAI API key:
echo 1. Go to https://platform.openai.com/
echo 2. Sign up or login
echo 3. Go to API Keys section
echo 4. Create new API key
echo 5. Copy the key and paste it in .env file
echo.
echo After setting up, restart the backend server
pause
