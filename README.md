# Gemini Vision API

A FastAPI-based application that uses Google's Gemini AI to analyze and describe images. This project provides a simple API and web interface to upload images and get detailed descriptions using Google's Gemini Vision AI.

## Features

- Image upload and analysis using Google's Gemini AI
- RESTful API endpoints with FastAPI
- Simple web interface for testing
- Detailed image descriptions using Gemini 2.0 Flash
- Error handling and temporary file management
- CORS support for cross-origin requests

## Prerequisites

- Python 3.8 or higher
- Google Cloud API key with Gemini API access
- Google Cloud service account credentials (for Vision API)
- Modern web browser
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```
git clone <your-repository-url>
cd gemini-vision-poc
```

2. Create and activate a virtual environment:
```
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```
pip install fastapi uvicorn python-multipart google-generativeai python-dotenv pillow google-cloud-vision
```

4. Set up your environment variables in `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=credentials\google-vision.json
```

5. Set up Google Cloud Vision API:
- Go to Google Cloud Console
- Create a new project or select an existing one
- Enable the Vision API
- Create a service account and download the JSON key file
- Create a 'credentials' directory in your project root
- Place the downloaded JSON key file in the credentials directory as 'google-vision.json'

## Installation

1. Clone the repository: