from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional, Dict, List, Any, Union
import google.generativeai as genai
from google.cloud import vision
from dataclasses import dataclass
import tempfile
import logging
import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FaceBox:
    """Data class representing a detected face with its bounding box and label."""
    box_2d: List[int]
    label: str

class FaceDetector:
    """Class handling face detection operations using Google Cloud Vision API."""
    
    def __init__(self, vision_client: vision.ImageAnnotatorClient):
        self.client = vision_client
        
    def detect_faces(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Detects faces in an image using Google Cloud Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with their bounding boxes and labels, or None if no faces detected
        """
        logger.info(f"Detecting faces in image: {image_path}")
        
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.face_detection(image=image)
            faces = response.face_annotations
          
            if not faces:
                logger.info("No faces detected in the image")
                return None
      
            face_details = []
            for i, face in enumerate(faces, 1):
                vertices = face.bounding_poly.vertices
                face_info = {
                    "box_2d": [
                        vertices[0].x,
                        vertices[0].y,
                        vertices[2].x,
                        vertices[2].y
                    ],
                    "label": f"Face {i}"
                }
                face_details.append(face_info)
      
            # Sort faces from left to right
            face_details.sort(key=lambda x: x['box_2d'][0])
            logger.info(f"Detected {len(face_details)} faces in the image")
            return face_details
      
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

class ImageAnalyzer:
    """Class handling image analysis operations using Google Gemini API."""
    
    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        
    def format_face_locations(self, face_locations_json: Optional[str]) -> Optional[Dict]:
        """Parse face locations JSON into a structured format."""
        if not face_locations_json:
            return None
            
        try:
            data = json.loads(face_locations_json)
            faces = data.get('face_detections', [])
            if not faces:
                return None
            
            # Keep the structured format
            return {
                "face_detections": [
                    {
                        "name": face['name'],
                        "coordinates": face['coordinates']
                    }
                    for face in faces
                ]
            }
        except Exception as e:
            logger.error(f"Error formatting face locations: {str(e)}")
            return None
        
    def get_analysis_prompt(self, face_locations: Optional[str] = None) -> str:
        """Generates the analysis prompt with optional face location information."""
        prompt = """
Analyze the provided image and produce a comprehensive, structured description. Incorporate object identification, visual attributes, and relationships between elements. For any faces detected, refer to each person by the name provided in the face_detections section and do not include bounding box coordinates in your narrative.
"""
        if face_locations:
            formatted_locations = self.format_face_locations(face_locations)
            if formatted_locations:
                # Add face information in JSON format
                prompt += "Detected faces:\n"
                prompt += json.dumps(formatted_locations, indent=2)
                prompt += "\n\n"
        
        prompt += """
Format:

1. Overall Scene & Mood:

General Description: [A brief summary of the scene]
Scene Type: [e.g., indoor, outdoor, cityscape, etc.]
Overall Mood: [e.g., peaceful, energetic, tense, joyful]
Style: [e.g., realistic, abstract, cartoonish, photographic]

2. Objects and Relationships:

Primary Objects:
- face_detections[].name (use the Word "Person" if someone has no face_detections and do not add this if no person on the picture): each detected person has a box arround the face face_detections[].coordinates and their name as listed in face_detections[].name. Use the name of the person and describe their visible attributes very detailed (e.g., gender, clothing, hair, accessories, facial expression). Do not mention coordinates.
- [Object 1]: [Type, color, texture, size, and position within the scene]
- [Object 2]: [Type, color, texture, size, and position within the scene]

Object Relationships:
Describe how the identified people and any primary objects relate or interact with each other. Avoid referencing any objects or entities not listed as primary.

3. Detailed Visual Attributes:

Dominant Colors: [List the main colors and where they appear]
Lighting: [Describe the type and direction of light]
Background Details: [Describe any notable background elements]
Shape Characteristics: [General description of shapes in the scene]
Texture Characteristics: [General description of textures in the scene]
Additional Notes: [Any other relevant details not covered above]

Instructions:
- Incorporate all detected faces using the given names from face_detections.
- Do not mention bounding box coordinates in the narrative.
- Only include objects and categories if they are present in the image.
- Ensure your final description is cohesive, visually rich, and accurately reflects the image's content and atmosphere.
"""

        return prompt

class OpenAIAnalyzer:
    """Class handling image analysis operations using OpenAI API."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        
    def format_face_locations(self, face_locations_json: Optional[str]) -> Optional[Dict]:
        """Parse face locations JSON into a structured format."""
        if not face_locations_json:
            return None
            
        try:
            data = json.loads(face_locations_json)
            faces = data.get('face_detections', [])
            if not faces:
                return None
            
            # Keep the structured format
            return {
                "face_detections": [
                    {
                        "name": face['name'],
                        "coordinates": face['coordinates']
                    }
                    for face in faces
                ]
            }
        except Exception as e:
            logger.error(f"Error formatting face locations: {str(e)}")
            return None
        
    def get_analysis_prompt(self, face_locations: Optional[str] = None) -> str:
        """Generates the analysis prompt with optional face location information."""
        prompt = """
Analyze the provided image and produce a comprehensive, structured description. Incorporate object identification, visual attributes, and relationships between elements. For any faces detected, refer to each person by the name provided in the face_detections section and do not include bounding box coordinates in your narrative.
"""
        if face_locations:
            formatted_locations = self.format_face_locations(face_locations)
            if formatted_locations:
                # Add face information in JSON format
                prompt += "Detected faces:\n"
                prompt += json.dumps(formatted_locations, indent=2)
                prompt += "\n\n"
        
        prompt += """
Format:

1. Overall Scene & Mood:

General Description: [A brief summary of the scene]
Scene Type: [e.g., indoor, outdoor, cityscape, etc.]
Overall Mood: [e.g., peaceful, energetic, tense, joyful]
Style: [e.g., realistic, abstract, cartoonish, photographic]

2. Objects and Relationships:

Primary Objects:
- face_detections[].name (use the word "Person" if someone has no face_detections and do not add this if no person on the picture): each detected person has a box arround the face face_detections[].coordinates and their name as listed in face_detections[].name. Use the name of the person and describe their visible attributes very detailed (e.g., gender, clothing, hair, accessories, facial expression). Do not mention coordinates.
- [Object 1]: [Type, color, texture, size, and position within the scene]
- [Object 2]: [Type, color, texture, size, and position within the scene]

Object Relationships:
Describe how the identified people and any primary objects relate or interact with each other. Avoid referencing any objects or entities not listed as primary.

3. Detailed Visual Attributes:

Dominant Colors: [List the main colors and where they appear]
Lighting: [Describe the type and direction of light]
Background Details: [Describe any notable background elements]
Shape Characteristics: [General description of shapes in the scene]
Texture Characteristics: [General description of textures in the scene]
Additional Notes: [Any other relevant details not covered above]

Instructions:
- Incorporate all detected faces using the given names from face_detections.
- Do not mention bounding box coordinates in the narrative.
- Only include objects and categories if they are present in the image.
- Ensure your final description is cohesive, visually rich, and accurately reflects the image's content and atmosphere.
"""

        return prompt

    async def analyze_image(self, image_path: str, face_locations: Optional[str] = None, temperature: float = 1.0) -> str:
        """Analyzes an image using OpenAI's Vision model."""
        try:
            # Read image and encode to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = self.get_analysis_prompt(face_locations)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=4096
            )
            
            # Ensure proper text encoding
            content = response.choices[0].message.content
            return content.encode('utf-8', errors='ignore').decode('utf-8')
            
        except Exception as e:
            logger.error(f"OpenAI analysis error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {str(e)}")

def init_app() -> tuple[FastAPI, FaceDetector, ImageAnalyzer, OpenAIAnalyzer, Jinja2Templates]:
    """Initialize the FastAPI application and required services."""
    # Load environment variables
    load_dotenv()
    logger.info(f"Loading credentials from: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    # Configure APIs
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    vision_client = vision.ImageAnnotatorClient()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize services
    face_detector = FaceDetector(vision_client)
    openai_analyzer = OpenAIAnalyzer(openai_client)
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )
    
    image_analyzer = ImageAnalyzer(model)
    
    # Create FastAPI app
    app = FastAPI()
    
    # Configure templates
    templates = Jinja2Templates(directory="templates")
    
    # Add CORS middleware with specific origins
    origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
        "file://",
        "*"
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    return app, face_detector, image_analyzer, openai_analyzer, templates

# Initialize application and services
app, face_detector, image_analyzer, openai_analyzer, templates = init_app()

async def handle_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Handle file upload and return temporary file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        return temp_file.name, content

@app.post("/detect-faces")
async def detect_faces_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint for face detection in uploaded images."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no filename")
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file_path = None
    try:
        logger.info(f"Received face detection request for file: {file.filename}")
        temp_file_path, _ = await handle_uploaded_file(file)
        
        face_details = face_detector.detect_faces(temp_file_path)
        return {
            "faces": face_details if face_details else [],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    face_locations: Optional[str] = Form(None),
    temperature: float = Form(1.0)
) -> Dict[str, Any]:
    """Endpoint for analyzing images with optional face location information."""
    temp_file_path = None
    try:
        logger.info(f"Received image analysis request for file: {file.filename}")
        logger.info(f"Using temperature: {temperature}")
        logger.info(f"Face locations received: {face_locations}")
        temp_file_path, _ = await handle_uploaded_file(file)
        
        # Configure Gemini with temperature
        generation_config = {
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )
        
        # Upload to Gemini and analyze
        uploaded_file = genai.upload_file(temp_file_path, mime_type="image/jpeg")
        prompt = image_analyzer.get_analysis_prompt(face_locations)
        
        # Log the complete prompt for debugging
        logger.info("Complete Gemini prompt:")
        logger.info(prompt)
        
        chat_session = model.start_chat()
        response = chat_session.send_message([uploaded_file, prompt])
        
        return {
            "description": response.text,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.post("/analyze-image-openai")
async def analyze_image_openai(
    file: UploadFile = File(...),
    face_locations: Optional[str] = Form(None),
    temperature: float = Form(1.0)
) -> Dict[str, Any]:
    """Endpoint for analyzing images using OpenAI with optional face location information."""
    temp_file_path = None
    try:
        logger.info(f"Received OpenAI image analysis request for file: {file.filename}")
        logger.info(f"Using temperature: {temperature}")
        logger.info(f"Face locations received: {face_locations}")
        temp_file_path, _ = await handle_uploaded_file(file)
        
        # Get and log the prompt
        prompt = openai_analyzer.get_analysis_prompt(face_locations)
        logger.info("Complete OpenAI prompt:")
        logger.info(prompt)
        
        response_text = await openai_analyzer.analyze_image(temp_file_path, face_locations, temperature)
        
        return {
            "description": response_text,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in OpenAI image analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI image analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 