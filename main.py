from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
from typing import Optional, Dict, List, Any, Union
from google.cloud import vision
from dataclasses import dataclass
import tempfile
import logging
import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import yaml
import io
from PIL import Image
import requests

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

class FaceRecognizer:
    """Class handling face recognition using Ollama model."""
    
    def __init__(self, model_name: str = "lama3.2-vision-gallery", base_model: str = "llama3.2-vision:11b", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_model = base_model
        self.ollama_host = ollama_host
        self._ensure_model_exists()
        
    def _ensure_model_exists(self):
        """Ensure the model exists, if not create it from base model."""
        try:
            # Check if model exists using Ollama API
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                raise Exception(f"Failed to connect to Ollama at {self.ollama_host}")
                
            models = response.json().get("models", [])
            # Remove any version tags for comparison
            model_exists = any(model["name"].split(":")[0] == self.model_name for model in models)
            
            # If model doesn't exist in the list, create it
            if not model_exists:
                logger.info(f"Model {self.model_name} not found, creating from {self.base_model}")
                
                # Create initial Modelfile
                modelfile_content = f"""
FROM {self.base_model}
PARAMETER temperature 1
PARAMETER num_ctx 4096

# System prompt for face recognition
SYSTEM You are a face recognition expert. When shown an image of a person, identify them if you recognize them from your training data. Always respond in YAML format with 'name' and 'confidence' fields. If you don't recognize the person, set confidence to 0.
"""
                
                # Write Modelfile
                with open("Modelfile", "w") as f:
                    f.write(modelfile_content)
                
                # Create model using Ollama API
                with open("Modelfile", "r") as f:
                    response = requests.post(
                        f"{self.ollama_host}/api/create",
                        json={
                            "name": self.model_name,
                            "modelfile": f.read()
                        }
                    )
                if response.status_code != 200:
                    raise Exception(f"Failed to create model: {response.text}")
                    
                logger.info(f"Created initial model {self.model_name}")
                
                # Clean up Modelfile
                os.unlink("Modelfile")
            else:
                logger.info(f"Model {self.model_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring model exists: {str(e)}")
            raise
        
    def recognize_face(self, face_image: Image.Image) -> tuple[Optional[str], float]:
        """Recognize a face using the Ollama model."""
        try:
            # Convert image to base64
            buf = io.BytesIO()
            face_image.save(buf, format="PNG")
            base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Construct prompt
            prompt = f"""
Analyze this image and identify the person if you recognize them from your training data.
If you recognize the person, return their name and your confidence level (0-100).
If you don't recognize them, return null for name and 0 for confidence.

[image]
data:image/png;base64,{base64_image}
[/image]

Respond in YAML format with 'name' and 'confidence' fields.
Example:
name: John Doe
confidence: 95.5
"""
            # Log prompt without base64 image
            logger.info("=== OLLAMA PROMPT ===")
            logger.info("Analyze this image and identify the person if you recognize them from your training data.")
            logger.info("If you recognize the person, return their name and your confidence level (0-100).")
            logger.info("If you don't recognize them, return null for name and 0 for confidence.")
            logger.info("[image data omitted for brevity]")
            logger.info("Respond in YAML format with 'name' and 'confidence' fields.")
            logger.info("Example:")
            logger.info("name: John Doe")
            logger.info("confidence: 95.5")
            logger.info("===================")
            
            # Query Ollama using REST API
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            logger.info(f"Sending request to Ollama API at {self.ollama_host}/api/generate")
            logger.info("Request data: [prompt omitted for brevity]")
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return None, 0
                
            result = response.json()
            response_text = result.get("response", "")
            
            logger.info("=== OLLAMA RESPONSE ===")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Text: {response_text}")
            logger.info("======================")
            
            # Parse response
            try:
                if not response_text.strip():
                    logger.warning("Empty response from Ollama")
                    return None, 0
                    
                output = yaml.safe_load(response_text)
                if not output:
                    logger.warning("Could not parse YAML from response")
                    return None, 0
                    
                name = output.get("name", None)
                confidence = float(output.get("confidence", 0))
                logger.info(f"Recognition result: name={name}, confidence={confidence}")
                return name, confidence
            except Exception as e:
                logger.error(f"Error parsing Ollama response: {str(e)}")
                logger.error(f"Raw response: {response_text}")
                return None, 0
                
        except Exception as e:
            logging.error(f"Error in face recognition: {str(e)}", exc_info=True)
            return None, 0
            
    def update_training_data(self, face_image: Image.Image, name: str, training_file: str = 'training.yaml'):
        """Update training data with new face."""
        try:
            # Convert image to base64
            buf = io.BytesIO()
            face_image.save(buf, format="PNG")
            base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Load or create training data
            try:
                with open(training_file, 'r') as f:
                    data = yaml.safe_load(f)
            except FileNotFoundError:
                data = {
                    "from": self.base_model,
                    "prompt": "Known individuals dataset",
                    "images": []
                }
                
            # Add new face
            data["images"].append({
                "name": name,
                "image": base64_image
            })
            
            # Save updated data
            with open(training_file, 'w') as f:
                yaml.dump(data, f)
                
        except Exception as e:
            logging.error(f"Error updating training data: {str(e)}", exc_info=True)
            raise
            
    def retrain_model(self, training_file: str = 'training.yaml'):
        """Retrain the Ollama model with updated data."""
        try:
            logger.info(f"Retraining model {self.model_name} using base model {self.base_model}")
            
            # Read training file
            with open(training_file, "r") as f:
                training_data = f.read()
                
            # Create model using Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/create",
                json={
                    "name": self.model_name,
                    "modelfile": training_data
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to retrain model: {response.text}")
                
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            raise

class FaceDetector:
    """Class handling face detection operations using Google Cloud Vision API."""
    
    def __init__(self, vision_client: vision.ImageAnnotatorClient, face_recognizer: Optional[FaceRecognizer] = None):
        self.client = vision_client
        self.face_recognizer = face_recognizer
        
    def detect_faces(self, image_path: str, confidence_threshold: float = 95.0) -> Optional[List[Dict[str, Any]]]:
        """
        Detects faces in an image using Google Cloud Vision API and optionally recognizes them.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for face recognition (default: 95.0)
            
        Returns:
            List of detected faces with their bounding boxes and labels
        """
        logger.info(f"Detecting faces in image: {image_path}")
        
        try:
            # Read image for both Vision API and Ollama
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            # Send to Vision API for face detection
            logger.info("Sending request to Google Vision API")
            image = vision.Image(content=content)
            response = self.client.face_detection(image=image)
            faces = response.face_annotations
            
            if not faces:
                logger.info("No faces detected in the image")
                return None
            
            logger.info(f"Google Vision API detected {len(faces)} faces")
            
            # Convert image for cropping
            img = Image.open(image_path)
            
            # Send entire image to Ollama for recognition
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Construct prompt for all faces
            prompt = f"""
Analyze this image and identify all the people in it.
There are {len(faces)} faces detected.

For each face, from left to right, identify if you recognize them.
If you recognize a person, provide their name and confidence level (0-100).
If you don't recognize someone, label them as "Person X" where X is their position number from left to right.

[image]
data:image/png;base64,{base64_image}
[/image]

Respond in YAML format with an array of people. For each person include 'name' and 'confidence' fields.
Example:
people:
  - name: John Doe
    confidence: 95.5
  - name: Person 2
    confidence: 0
  - name: Jane Smith
    confidence: 88.2
"""
            # Query Ollama for all faces at once
            logger.info("Sending request to Ollama for face recognition")
            request_data = {
                "model": self.face_recognizer.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.face_recognizer.ollama_host}/api/generate",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                # Fall back to numbered faces
                face_details = []
                for i, face in enumerate(faces, 1):
                    vertices = face.bounding_poly.vertices
                    face_details.append({
                        "box_2d": [vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y],
                        "label": f"Person {i}"
                    })
                return face_details
            
            # Parse Ollama response
            result = response.json()
            response_text = result.get("response", "")
            logger.info(f"Ollama response: {response_text}")
            
            try:
                output = yaml.safe_load(response_text)
                recognized_people = output.get("people", [])
                
                # Match faces with recognition results
                face_details = []
                for i, (face, recognition) in enumerate(zip(faces, recognized_people), 1):
                    vertices = face.bounding_poly.vertices
                    box_2d = [vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y]
                    
                    name = recognition.get("name", f"Person {i}")
                    confidence = float(recognition.get("confidence", 0))
                    
                    face_info = {
                        "box_2d": box_2d,
                        "label": name if confidence >= confidence_threshold else f"Person {i}"
                    }
                    
                    if confidence > 0:
                        face_info["recognition"] = {
                            "name": name,
                            "confidence": confidence
                        }
                    
                    face_details.append(face_info)
                    logger.info(f"Face {i}: {name} (confidence: {confidence}%)")
                
                return face_details
                
            except Exception as e:
                logger.error(f"Error parsing recognition results: {str(e)}")
                # Fall back to numbered faces
                face_details = []
                for i, face in enumerate(faces, 1):
                    vertices = face.bounding_poly.vertices
                    face_details.append({
                        "box_2d": [vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y],
                        "label": f"Person {i}"
                    })
                return face_details
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

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
        prompt = """Analyze the provided image and produce a comprehensive, structured description. Incorporate object identification, visual attributes, and relationships between elements. For any faces detected, refer to each person by the name provided in the face_detections section and do not include bounding box coordinates in your narrative.

"""
        if face_locations:
            formatted_locations = self.format_face_locations(face_locations)
            if formatted_locations:
                # Add face information in JSON format
                prompt += "Detected faces:\n"
                prompt += json.dumps(formatted_locations, indent=2)
                prompt += "\n\n"
        
        prompt += """Format:

1. Overall Scene & Mood:

General Description: [A brief summary of the scene]
Scene Type: [e.g., indoor, outdoor, cityscape, etc.]
Overall Mood: [e.g., peaceful, energetic, tense, joyful]
Style: [e.g., realistic, abstract, cartoonish, photographic]

2. Objects and Relationships:

Primary Objects:

- face_detections[].name (use the Word "Person" if someone has no face_detections): Refer to each detected person that has a box arround the face face_detections[].coordinates and their name as listed in face_detections[].name. Use the name of the person and describe their visible attributes very detailed (e.g., gender, clothing, hair, accessories, facial expression). Do not mention coordinates.
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
- Ensure your final description is cohesive, visually rich, and accurately reflects the image's content and atmosphere."""
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

def init_app() -> tuple[FastAPI, FaceDetector, OpenAIAnalyzer, Jinja2Templates]:
    """Initialize the FastAPI application and required services."""
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    logger.info(f"Loading environment from: {env_path}")
    load_dotenv(env_path)
    
    # Get and validate Google credentials path
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    logger.info(f"Environment loaded. GOOGLE_APPLICATION_CREDENTIALS={credentials_path}")
    
    if not credentials_path:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    if not os.path.exists(credentials_path):
        raise Exception(f"Google credentials file not found at: {credentials_path}")
        
    logger.info(f"Loading Google credentials from: {credentials_path}")
    
    # Configure APIs
    try:
        vision_client = vision.ImageAnnotatorClient()
        logger.info("Successfully created Vision API client")
        
        # Initialize OpenAI client with optional organization ID
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OPENAI_API_KEY environment variable not set")
            
        openai_org_id = os.getenv("OPENAI_ORG_ID")
        openai_client = OpenAI(
            api_key=openai_api_key,
            organization=openai_org_id if openai_org_id and openai_org_id != "your_openai_org_id" else None
        )
        logger.info("Successfully created OpenAI client")
    except Exception as e:
        logger.error(f"Failed to create API clients: {str(e)}")
        raise
    
    # Initialize face recognizer with Ollama configuration
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    face_recognizer = FaceRecognizer(ollama_host=ollama_host)
    
    # Initialize services
    face_detector = FaceDetector(vision_client, face_recognizer)
    openai_analyzer = OpenAIAnalyzer(openai_client)
    
    # Create FastAPI app
    app = FastAPI()
    
    # Configure templates
    templates = Jinja2Templates(directory="templates")
    
    # Add CORS middleware
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
    
    return app, face_detector, openai_analyzer, templates

# Initialize application and services
app, face_detector, openai_analyzer, templates = init_app()

async def handle_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Handle file upload and return temporary file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        return temp_file.name, content

@app.post("/detect-faces")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(95.0)  # Default threshold is 95%
) -> Dict[str, Any]:
    """Endpoint for face detection and recognition in uploaded images."""
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
        logger.info(f"Using confidence threshold: {confidence_threshold}%")
        temp_file_path, _ = await handle_uploaded_file(file)
        logger.info(f"Saved uploaded file to: {temp_file_path}")
        
        # Pass confidence threshold to face detector
        face_details = face_detector.detect_faces(temp_file_path, confidence_threshold)
        logger.info(f"Face detection result: {json.dumps(face_details, indent=2) if face_details else 'No faces detected'}")
        
        response_data = {
            "faces": face_details if face_details else [],
            "status": "success"
        }
        logger.info(f"Sending response: {json.dumps(response_data, indent=2)}")
        return response_data
        
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

@app.post("/update-face-name")
async def update_face_name(
    file: UploadFile = File(...),
    face_index: int = Form(...),
    name: str = Form(...),
    box_2d: str = Form(...)
) -> Dict[str, Any]:
    """Endpoint to update face name and retrain the model."""
    temp_file_path = None
    try:
        logger.info(f"Received update face name request for face {face_index} with name '{name}'")
        temp_file_path, _ = await handle_uploaded_file(file)
        logger.info(f"Saved uploaded file to: {temp_file_path}")
        
        # Parse box_2d coordinates
        box_coords = json.loads(box_2d)
        logger.info(f"Face coordinates: {box_coords}")
        
        # Crop face from image
        img = Image.open(temp_file_path)
        face_img = img.crop((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
        logger.info("Successfully cropped face from image")
        
        # Update training data using the existing face_recognizer instance
        face_detector.face_recognizer.update_training_data(face_img, name)
        logger.info("Updated training data")
        
        face_detector.face_recognizer.retrain_model()
        logger.info("Retrained model")
        
        response_data = {
            "status": "success",
            "message": f"Updated face {face_index} with name '{name}' and retrained model"
        }
        logger.info(f"Sending response: {json.dumps(response_data, indent=2)}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error updating face name: {str(e)}", exc_info=True)
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