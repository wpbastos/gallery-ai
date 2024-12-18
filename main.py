from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
import tempfile
import logging
import json
import os
import base64
import numpy as np
import face_recognition
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
FACE_EMBEDDINGS_PATH = MODELS_DIR / "face_embeddings.npz"
FACE_LABELS_PATH = MODELS_DIR / "face_labels.json"

# Global variables
face_recognizer = None  # Will be initialized in init_app()

@dataclass
class FaceBox:
    """Data class representing a detected face with its bounding box and label."""
    box_2d: List[int]
    label: str
    confidence: float = 0.0

class FaceRecognizer:
    """Class handling face recognition using TensorFlow and face_recognition library."""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_labels = []
        self.confidence_threshold = 0.9
        self.load_face_data()
        
    def load_face_data(self):
        """Load saved face encodings and labels if they exist."""
        try:
            if FACE_EMBEDDINGS_PATH.exists() and FACE_LABELS_PATH.exists():
                data = np.load(str(FACE_EMBEDDINGS_PATH))
                self.known_face_encodings = data['encodings']
                
                with open(FACE_LABELS_PATH, 'r') as f:
                    self.known_face_labels = json.load(f)
                logger.info(f"Loaded {len(self.known_face_labels)} known faces")
        except Exception as e:
            logger.error(f"Error loading face data: {e}")
            self.known_face_encodings = []
            self.known_face_labels = []
    
    def save_face_data(self):
        """Save face encodings and labels."""
        try:
            np.savez(str(FACE_EMBEDDINGS_PATH), encodings=self.known_face_encodings)
            with open(FACE_LABELS_PATH, 'w') as f:
                json.dump(self.known_face_labels, f)
            logger.info("Face data saved successfully")
        except Exception as e:
            logger.error(f"Error saving face data: {e}")
    
    def recognize_face(self, face_image) -> tuple[str, float]:
        """
        Recognize a face in the given image.
        Returns tuple of (label, confidence)
        """
        # Get face encoding
        face_encoding = face_recognition.face_encodings(face_image)
        
        if not face_encoding:
            return "Unknown", 0.0
            
        face_encoding = face_encoding[0]
        
        if not self.known_face_encodings:
            return "Unknown", 0.0
            
        # Calculate face distances
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        if len(face_distances) == 0:
            return "Unknown", 0.0
            
        # Convert distance to confidence (1 - distance)
        confidence = 1 - min(face_distances)
        best_match_index = np.argmin(face_distances)
        
        if confidence >= self.confidence_threshold:
            return self.known_face_labels[best_match_index], confidence
        
        return "Unknown", confidence
    
    def learn_face(self, face_image: np.ndarray, label: str):
        """Learn a new face or update existing face."""
        face_encoding = face_recognition.face_encodings(face_image)
        
        if not face_encoding:
            raise ValueError("No face found in the image")
            
        face_encoding = face_encoding[0]
        
        # Check if this label already exists
        if label in self.known_face_labels:
            idx = self.known_face_labels.index(label)
            self.known_face_encodings[idx] = face_encoding
        else:
            if len(self.known_face_encodings) == 0:
                self.known_face_encodings = np.array([face_encoding])
            else:
                self.known_face_encodings = np.vstack([self.known_face_encodings, face_encoding])
            self.known_face_labels.append(label)
        
        self.save_face_data()

class FaceDetector:
    """Class handling face detection and recognition operations using face_recognition library."""
    
    def __init__(self, face_recognizer: FaceRecognizer):
        self.face_recognizer = face_recognizer
        
    def detect_faces(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Detects and recognizes faces in an image using face_recognition library.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with their bounding boxes, labels, and confidence scores
        """
        logger.info(f"Detecting faces in image: {image_path}")
        
        try:
            # Load the image
            logger.info("Loading image file...")
            image = face_recognition.load_image_file(image_path)
            logger.info(f"Image loaded successfully. Shape: {image.shape}")
            
            # Find all faces in the image
            logger.info("Detecting faces...")
            face_locations = face_recognition.face_locations(image, model="hog")  # Use 'cnn' for better accuracy if GPU available
            logger.info(f"Found {len(face_locations)} faces.")
            
            if not face_locations:
                logger.info("No faces detected in the image")
                return []
            
            detected_faces = []
            for i, face_location in enumerate(face_locations):
                logger.info(f"Processing face {i+1}...")
                try:
                    # Convert face location to bounding box coordinates
                    top, right, bottom, left = face_location
                    box_2d = [left, top, right, bottom]
                    logger.info(f"Face {i+1} coordinates: {box_2d}")
                    
                    # Extract face image
                    face_image = image[top:bottom, left:right]
                    if face_image.size == 0:
                        logger.warning(f"Face {i+1} has invalid dimensions, skipping")
                        continue
                    
                    logger.info(f"Face {i+1} image extracted. Shape: {face_image.shape}")
                    
                    # Recognize face
                    label, confidence = self.face_recognizer.recognize_face(face_image)
                    logger.info(f"Face {i+1} recognition result: {label} with confidence {confidence}")
                    
                    # Convert coordinates to match expected format (left, top, right, bottom)
                    box_2d = [
                        left,   # left
                        top,    # top
                        right,  # right
                        bottom  # bottom
                    ]
                    
                    detected_faces.append(FaceBox(
                        box_2d=box_2d,
                        label=label,
                        confidence=float(confidence)
                    ))
                except Exception as e:
                    logger.error(f"Error processing face {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(detected_faces)} faces")
            return [{"box_2d": face.box_2d, "label": face.label, "confidence": face.confidence} 
                    for face in detected_faces]
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

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
    load_dotenv()
    
    # Initialize OpenAI client
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID", None)
    )
    
    # Initialize services
    face_recognizer = FaceRecognizer()
    face_detector = FaceDetector(face_recognizer)
    openai_analyzer = OpenAIAnalyzer(openai_client)
    
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
        logger.info(f"Temporary file created at: {temp_file_path}")
        
        # Load the image
        logger.info("Loading image file...")
        image = face_recognition.load_image_file(temp_file_path)
        logger.info(f"Original image shape: {image.shape}")
        
        # Resize image to a much smaller size for faster detection (max dimension 500px)
        max_dimension = 500
        height, width = image.shape[:2]
        
        # Store original dimensions for scaling back
        original_height, original_width = height, width
        
        if max(height, width) > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            
            # Fast resize using numpy stride tricks
            h_stride = height // new_height
            w_stride = width // new_width
            image = image[::h_stride, ::w_stride]
            logger.info(f"Resized image shape: {image.shape}")
        
        # Use HOG model with minimal upsampling for speed
        logger.info("Detecting faces using HOG model...")
        face_locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=1)
        logger.info(f"Found {len(face_locations)} faces")
        
        if not face_locations:
            logger.warning("No faces detected in the image.")
            return {
                "status": "success",
                "faces": [],
                "message": "No faces detected in the image. Try adjusting the image or using a clearer photo."
            }
        
        # Calculate scale factors to map back to original image size
        scale_x = original_width / image.shape[1]
        scale_y = original_height / image.shape[0]
        
        detected_faces = []
        for i, face_location in enumerate(face_locations):
            try:
                # Convert face location to bounding box coordinates
                top, right, bottom, left = face_location
                
                # Scale coordinates back to original image size
                box_2d = [
                    int(left * scale_x),
                    int(top * scale_y),
                    int(right * scale_x),
                    int(bottom * scale_y)
                ]
                
                # Extract face region from original image
                face_image = image[top:bottom, left:right]
                
                # Try to recognize the face
                label, confidence = face_recognizer.recognize_face(face_image)
                
                detected_faces.append({
                    "box_2d": box_2d,
                    "label": label,
                    "confidence": float(confidence)
                })
                
            except Exception as e:
                logger.error(f"Error processing face {i+1}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "faces": detected_faces,
            "message": f"Detected {len(detected_faces)} faces"
        }
        
    except Exception as e:
        logger.error(f"Error in face detection endpoint: {str(e)}", exc_info=True)
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

@app.post("/learn-face")
async def learn_face(
    file: UploadFile = File(...),
    label: str = Form(...),
    face_box: str = Form(...)
):
    """
    Learn a new face or update existing face.
    face_box should be a JSON string containing [x1, y1, x2, y2] coordinates.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Load image and extract face
        image = face_recognition.load_image_file(temp_path)
        box = json.loads(face_box)
        face_image = image[box[1]:box[3], box[0]:box[2]]

        # Learn face
        face_recognizer.learn_face(face_image, label)
        
        os.unlink(temp_path)  # Clean up temp file
        return JSONResponse(content={"status": "success", "message": f"Learned face: {label}"})
        
    except Exception as e:
        logger.error(f"Error learning face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-faces")
async def clear_faces():
    """Clear all learned face data."""
    try:
        global face_recognizer
        # Initialize face_recognizer if it's None
        if face_recognizer is None:
            face_recognizer = FaceRecognizer()

        # Check if there's any data to clear
        has_embeddings = FACE_EMBEDDINGS_PATH.exists()
        has_labels = FACE_LABELS_PATH.exists()
        
        if not has_embeddings and not has_labels and not face_recognizer.known_face_labels:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "No face data to clear."
                }
            )

        # Clear the face recognizer data
        face_recognizer.known_face_encodings = []
        face_recognizer.known_face_labels = []
        
        # Remove the saved files if they exist
        try:
            if has_embeddings:
                FACE_EMBEDDINGS_PATH.unlink()
            if has_labels:
                FACE_LABELS_PATH.unlink()
        except PermissionError as e:
            logger.error(f"Permission error deleting face data files: {e}")
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "message": "Permission denied when trying to delete face data files."
                }
            )
        except FileNotFoundError:
            # Files might have been deleted by another process, that's okay
            pass
            
        # Save empty state
        face_recognizer.save_face_data()
            
        logger.info("All face data cleared successfully")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "All learned faces have been cleared successfully."
            }
        )
    except Exception as e:
        logger.error(f"Error clearing face data: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to clear face data: {str(e)}"
            }
        )

@app.get("/learned-faces")
async def get_learned_faces():
    """Get all learned faces and their labels."""
    try:
        # Check if face data files exist
        if not FACE_EMBEDDINGS_PATH.exists() or not FACE_LABELS_PATH.exists():
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "faces": [],
                    "message": "No face data found. Start by learning some faces."
                }
            )

        # Check if there are any faces in memory
        if not face_recognizer.known_face_labels:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "faces": [],
                    "message": "No faces have been learned yet."
                }
            )

        # Return faces with counts
        faces_data = {
            "status": "success",
            "faces": [
                {
                    "label": label,
                    "count": 1,  # Could be enhanced to track multiple encodings per label
                    "last_updated": "Recently"  # Could be enhanced to track timestamps
                }
                for label in face_recognizer.known_face_labels
            ],
            "message": f"Found {len(face_recognizer.known_face_labels)} learned faces."
        }
        return JSONResponse(status_code=200, content=faces_data)

    except FileNotFoundError as e:
        logger.error(f"Face data files not found: {e}")
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "Face data files not found. The system may need to be initialized."
            }
        )
    except PermissionError as e:
        logger.error(f"Permission error accessing face data: {e}")
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "Permission denied accessing face data files."
            }
        )
    except Exception as e:
        logger.error(f"Error getting learned faces: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 