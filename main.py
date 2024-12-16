from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google.cloud import vision
from PIL import Image
import os
from dotenv import load_dotenv
import tempfile
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info(f"Loading credentials from: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logger.info("Gemini API configured")

# Initialize Vision API client
vision_client = vision.ImageAnnotatorClient()
logger.info("Vision API client initialized")

# Create generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def detect_faces(image_path):
    """Detects faces in an image and returns face details."""
    logger.info(f"Detecting faces in image: {image_path}")
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = vision_client.face_detection(image=image)
    faces = response.face_annotations

    if not faces:
        logger.info("No faces detected in the image")
        return None

    face_details = []
    likelihood_name = ("UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY")
    
    for i, face in enumerate(faces, 1):
        face_info = {
            "position": i,
            "emotions": {
                "joy": likelihood_name[face.joy_likelihood],
                "anger": likelihood_name[face.anger_likelihood],
                "surprise": likelihood_name[face.surprise_likelihood],
            },
            "bounds": [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        }
        face_details.append(face_info)

    logger.info(f"Detected {len(face_details)} faces in the image")
    return face_details

# Define structured prompt template
def get_analysis_prompt(face_details, names=None):
    faces_info = ""
    if face_details:
        faces_info = "\nPeople in the image:\n"
        for i, face in enumerate(face_details, 1):
            name = names[i-1].strip() if names and len(names) >= i else f"Person {i}"
            faces_info += f"- {name} (showing {', '.join(f'{emotion}: {level}' for emotion, level in face['emotions'].items())})\n"

        faces_info += "\nPlease refer to these people by their names in the analysis."

    return f"""
Analyze the provided image and provide a detailed structured description, incorporating object identification, their relationships, and visual attributes. {faces_info}
Please use the following format, and for any category where details are not present, indicate 'None':

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

@app.post("/detect-faces")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Received face detection request for file: {file.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_content = await file.read()
            temp_file.write(image_content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary path: {temp_file_path}")

        face_details = detect_faces(temp_file_path)
        os.unlink(temp_file_path)
        logger.info("Temporary file cleaned up")

        return {
            "faces": face_details,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}", exc_info=True)
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        return {
            "error": str(e),
            "status": "error"
        }

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), names: str = None):
    try:
        logger.info(f"Received image analysis request for file: {file.filename}")
        if names:
            logger.info(f"Names provided: {names}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_content = await file.read()
            temp_file.write(image_content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary path: {temp_file_path}")

        # Detect faces first
        face_details = detect_faces(temp_file_path)
        
        # Parse names if provided
        name_list = names.split(',') if names else None
        if name_list:
            logger.info(f"Parsed names: {name_list}")

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(temp_file_path)
        logger.info("File uploaded to Gemini")

        # Get the analysis prompt with face information
        prompt = get_analysis_prompt(face_details, name_list)
        logger.info("Generated analysis prompt")

        # Start a chat session
        chat_session = model.start_chat()
        
        # Send the image for analysis with structured prompt
        response = chat_session.send_message([
            uploaded_file,
            prompt
        ])
        logger.info("Received response from Gemini")

        # Clean up temporary file
        os.unlink(temp_file_path)
        logger.info("Temporary file cleaned up")
        
        return {
            "description": response.text,
            "faces": face_details,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}", exc_info=True)
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