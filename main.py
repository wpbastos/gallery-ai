from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define structured prompt template
STRUCTURED_PROMPT = """
Analyze the provided image and provide a detailed structured description, incorporating object identification, their relationships, and visual attributes. Please use the following format, and for any category where details are not present, indicate 'None':

**1. Overall Scene & Mood:**
    - **General Description:** [A brief, overall summary of the scene in the image]
    - **Scene type:** [e.g., indoor, outdoor, landscape, cityscape]
    - **Overall mood:** [e.g., peaceful, energetic, tense, joyful]
    - **Style:** [e.g., realistic, abstract, cartoonish, photographic]

**2. Objects and Relationships:**
    - **Primary Objects:**
         - [Object 1]: [Type, color, texture, size, and position]
         - [Object 2]: [Type, color, texture, size, and position]
    - **Object Relationships:** [Describe how the objects interact]

**3. Detailed Visual Attributes:**
    - **Dominant Colors:** [List the main colors and where they appear]
    - **Lighting:** [Type and direction of light]
    - **Background Details:** [Description of the background elements]
    - **Shape Characteristics:** [General description of the shapes]
    - **Texture Characteristics:** [General description of the textures]
    - **Presence of Humans/Animals:** [Brief description if humans or animals are present]
    - **Additional Notes:** [Any other relevant details]
"""

# Initialize model with config
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def upload_to_gemini(file_path, mime_type="image/jpeg"):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_content = await file.read()
            temp_file.write(image_content)
            temp_file_path = temp_file.name

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(temp_file_path)

        # Start a chat session
        chat_session = model.start_chat()
        
        # Send the image for analysis with structured prompt
        response = chat_session.send_message([
            uploaded_file,
            STRUCTURED_PROMPT
        ])

        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "description": response.text,
            "status": "success"
        }
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/")
async def root():
    return {"message": "Welcome to Gemini Vision API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 