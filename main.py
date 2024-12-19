from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional, Dict, List, Any, Union
from google.cloud import vision
from google.api_core import client_options, retry
from dataclasses import dataclass
import tempfile
import logging
import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import warnings
import onnxruntime
import time
import math

# Disable CUDA provider globally for onnxruntime
onnxruntime.set_default_logger_severity(3)  # Suppress most logging
os.environ['ONNXRUNTIME_PROVIDER_PATH'] = ''  # Prevent provider discovery
os.environ['ORT_DISABLE_PROVIDER_LOADING_VERBOSE_LOGS'] = '1'

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*LoadLibrary failed.*')
warnings.filterwarnings('ignore', message='.*CUDA.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger('insightface').setLevel(logging.WARNING)
logging.getLogger('onnxruntime').setLevel(logging.WARNING)

# Suppress grpc warning
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

@dataclass
class FaceBox:
    """Data class representing a detected face with its bounding box and label."""
    box_2d: List[int]
    label: str

class FaceDetector:
    """Class handling face detection operations using Google Cloud Vision API."""
    
    def __init__(self, vision_client: vision.ImageAnnotatorClient):
        self.client = vision_client
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def detect_faces(self, image_path: str, retry_count: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Detects faces in an image using Google Cloud Vision API with retry logic.
        
        Args:
            image_path: Path to the image file
            retry_count: Current retry attempt number
            
        Returns:
            List of detected faces with their bounding boxes and labels, or None if no faces detected
        """
        logger.info(f"Detecting faces in image: {image_path} (attempt {retry_count + 1}/{self.max_retries + 1})")
        
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
                    "label": f"Face {i}",
                    "confidence": face.detection_confidence if hasattr(face, 'detection_confidence') else 1.0
                }
                face_details.append(face_info)
      
            # Sort faces from left to right
            face_details.sort(key=lambda x: x['box_2d'][0])
            logger.info(f"Detected {len(face_details)} faces in the image")
            return face_details
      
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in face detection: {error_msg}", exc_info=True)
            
            # Check if error is related to connection or session timeout
            if (retry_count < self.max_retries and 
                ("GOAWAY" in error_msg or 
                 "session_timed_out" in error_msg or 
                 "Connection reset by peer" in error_msg or
                 "StatusCode.UNAVAILABLE" in error_msg)):
                
                logger.info(f"Retrying face detection after error: {error_msg}")
                import time
                time.sleep(self.retry_delay * (retry_count + 1))  # Exponential backoff
                return self.detect_faces(image_path, retry_count + 1)
            
            raise HTTPException(status_code=500, detail=f"Face detection failed: {error_msg}")

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

class FaceRecognizer:
    """Class handling face recognition using InsightFace's ArcFace model."""
    
    def __init__(self):
        """Initialize the face recognition model and face database."""
        # Set model directory
        self.model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        os.environ['INSIGHTFACE_HOME'] = self.model_dir
        
        # Initialize InsightFace model with explicit provider configuration
        providers = [('CPUExecutionProvider', {})]
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        
        self.app = FaceAnalysis(
            name='buffalo_l',  # Using large model for better accuracy
            root=self.model_dir,
            session_options=session_options,
            providers=providers,
            allowed_modules=['detection', 'recognition']  # Explicitly enable modules
        )
        
        # Prepare model with CPU and detection size that's compatible with the model's stride
        # Using dimensions that are multiples of 32 (common stride in detection models)
        self.app.prepare(ctx_id=-1, det_size=(1920, 1088))  # Adjusted height to be multiple of 32
        logger.info("Face recognition model loaded successfully with CPU")
        
        self.input_size = (112, 112)  # ArcFace standard size
        self.threshold = 0.5  # Base threshold
        self.min_face_size = 80  # Minimum face size in pixels
        
        # Create faces directory if it doesn't exist
        self.faces_dir = 'faces'
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize face database with support for multiple embeddings per person
        self.face_db = {}  # Dictionary to store known face embeddings
        self.load_face_database()
        
    def load_face_database(self):
        """Load known faces from database file."""
        db_path = os.path.join(self.faces_dir, 'face_database.json')
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                self.face_db = json.load(f)
                logger.info(f"Loaded {len(self.face_db)} faces from database")
        
    def save_face_database(self):
        """Save known faces to database file."""
        db_path = os.path.join(self.faces_dir, 'face_database.json')
        # Create a backup of the existing database
        if os.path.exists(db_path):
            backup_path = os.path.join(self.faces_dir, 'face_database.backup.json')
            try:
                with open(db_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
        
        # Save the new database
        with open(db_path, 'w') as f:
            json.dump(self.face_db, f, indent=2)
        logger.info(f"Saved {len(self.face_db)} faces to database")
        
    def learn_new_face(self, embedding: np.ndarray, name: str):
        """Add new face to database with support for multiple embeddings."""
        if embedding is None:
            raise ValueError("Cannot learn face: embedding is None")
            
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding_list = embedding.tolist()
        
        if not embedding_list:
            raise ValueError("Cannot learn face: empty embedding")
            
        # Add new embedding to existing ones or create new entry
        if name in self.face_db:
            # Check if this embedding is significantly different from existing ones
            existing_embeddings = np.array(self.face_db[name]['embeddings'])
            similarities = np.dot(existing_embeddings, embedding)
            if not any(sim > 0.9 for sim in similarities):  # Only add if sufficiently different
                self.face_db[name]['embeddings'].append(embedding_list)
                logger.info(f"Added new embedding variant for {name}")
        else:
            self.face_db[name] = {
                'embeddings': [embedding_list],
                'timestamp': time.time()
            }
            logger.info(f"Created new face entry for {name}")
        
        self.save_face_database()
        
        # Save face thumbnail if available
        if hasattr(self, 'current_face_image'):
            try:
                # Save both aligned and original face images
                aligned_path = os.path.join(self.faces_dir, f"{name}_aligned.jpg")
                cv2.imwrite(aligned_path, self.current_face_image)
                
                if hasattr(self, 'current_face_original'):
                    original_path = os.path.join(self.faces_dir, f"{name}_original.jpg")
                    cv2.imwrite(original_path, self.current_face_original)
                
                logger.info(f"Saved face thumbnails for {name}")
            except Exception as e:
                logger.warning(f"Failed to save face thumbnails: {str(e)}")
        
        logger.info(f"Learned new face: {name} (total embeddings: {len(self.face_db[name]['embeddings'])})")
        
    def find_matching_face(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find matching face in database using enhanced matching logic."""
        if not self.face_db:
            logger.info("No faces in database")
            return None
            
        if embedding is None:
            logger.warning("Cannot match face: embedding is None")
            return None
            
        # Normalize input embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        best_match = {
            'name': None,
            'similarity': -1,
            'confidence': 0
        }
        
        # Store all similarities for analysis
        all_similarities = {}
        
        for name, face_data in self.face_db.items():
            try:
                embeddings = np.array(face_data['embeddings'])
                # Calculate similarities with all embeddings for this person
                similarities = np.dot(embeddings, embedding)
                
                # Get best and average similarity for this person
                max_similarity = np.max(similarities)
                avg_similarity = np.mean(similarities)
                
                # Calculate a confidence score based on both max and average similarity
                confidence = (max_similarity * 0.7 + avg_similarity * 0.3)
                
                all_similarities[name] = {
                    'max': max_similarity,
                    'avg': avg_similarity,
                    'confidence': confidence
                }
                
                if confidence > best_match['similarity']:
                    best_match = {
                        'name': name,
                        'similarity': max_similarity,
                        'confidence': confidence
                    }
                    
            except Exception as e:
                logger.error(f"Error computing similarity for {name}: {str(e)}")
                continue
        
        # Log detailed similarity information
        for name, scores in all_similarities.items():
            logger.info(f"Similarity scores for {name}:")
            logger.info(f"  Max: {scores['max']:.3f}")
            logger.info(f"  Avg: {scores['avg']:.3f}")
            logger.info(f"  Confidence: {scores['confidence']:.3f}")
        
        if best_match['confidence'] > self.threshold:
            logger.info(f"Found match: {best_match['name']} with confidence: {best_match['confidence']:.3f}")
            return best_match
        else:
            logger.info(f"No match found above threshold {self.threshold:.3f} (best confidence: {best_match['confidence']:.3f})")
            return None
        
    def recognize_face(self, image_path: str, face_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            height, width = image_rgb.shape[:2]
            logger.info(f"Processing image at original size: {width}x{height}")
            
            # Pad height to be multiple of 32 if needed
            if height % 32 != 0:
                new_height = ((height // 32) + 1) * 32
                padding = new_height - height
                image_rgb = cv2.copyMakeBorder(image_rgb, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                logger.info(f"Padded height from {height} to {new_height} to match model stride")
            
            # Get all faces using InsightFace
            faces = self.app.get(image_rgb)
            if not faces:
                logger.warning("InsightFace failed to detect any faces")
                return face_boxes
            
            logger.info(f"InsightFace detected {len(faces)} faces")
            
            # Process each detected face
            face_details = []
            for i, face_box in enumerate(face_boxes):
                box_2d = face_box['box_2d']
                x1, y1, x2, y2 = [int(coord) for coord in box_2d]
                logger.info(f"Processing face {i+1} at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                # Find matching InsightFace detection
                matched_face = None
                best_iou = 0
                
                for j, face in enumerate(faces):
                    face_bbox = face.bbox.astype(int)
                    # Calculate IoU (Intersection over Union)
                    intersection = max(0, min(x2, face_bbox[2]) - max(x1, face_bbox[0])) * \
                                 max(0, min(y2, face_bbox[3]) - max(y1, face_bbox[1]))
                    union = (x2 - x1) * (y2 - y1) + \
                           (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]) - \
                           intersection
                    iou = intersection / union if union > 0 else 0
                    
                    logger.info(f"Face {i+1} - Comparing with InsightFace detection {j+1}:")
                    logger.info(f"  InsightFace bbox: {face_bbox.tolist()}")
                    logger.info(f"  IoU score: {iou:.3f}")
                    
                    if iou > best_iou and iou > 0.3:  # Lower IoU threshold from 0.4 to 0.3
                        best_iou = iou
                        matched_face = face
                
                if matched_face is not None:
                    logger.info(f"Face {i+1} - Found matching InsightFace detection with IoU: {best_iou:.3f}")
                    try:
                        # Get aligned face
                        logger.info(f"Face {i+1} - Attempting face alignment")
                        try:
                            aligned_face = face_align.norm_crop(image_rgb, matched_face.kps)
                            logger.info(f"Face {i+1} - Face alignment successful")
                        except Exception as align_error:
                            logger.warning(f"Face {i+1} - Face alignment failed, using direct crop: {str(align_error)}")
                            # Fallback to direct crop if alignment fails
                            x1, y1, x2, y2 = [int(coord) for coord in box_2d]
                            aligned_face = image_rgb[y1:y2, x1:x2]
                            # Resize to match ArcFace input size
                            aligned_face = cv2.resize(aligned_face, (112, 112))
                            
                        self.current_face_image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                        
                        # Store original face crop
                        face_crop = image[y1:y2, x1:x2]
                        self.current_face_original = face_crop
                        
                        # Create thumbnail
                        timestamp = int(time.time() * 1000)
                        thumbnail_filename = f"face_{timestamp}.jpg"
                        thumbnail_path = os.path.join('temp_faces', thumbnail_filename)
                        cv2.imwrite(os.path.join('static', thumbnail_path), self.current_face_image)
                        logger.info(f"Face {i+1} - Created thumbnail: {thumbnail_path}")
                        
                        # Get recognition result
                        if matched_face.embedding is not None:
                            logger.info(f"Face {i+1} - Successfully obtained face embedding")
                            match_result = self.find_matching_face(matched_face.embedding)
                        else:
                            logger.error(f"Face {i+1} - Failed to get face embedding")
                            raise ValueError("Face embedding is None")
                        
                        face_info = {
                            'box_2d': box_2d,
                            'label': face_box.get('label', 'Face'),
                            'confidence': match_result['confidence'] if match_result else 0.0,
                            'recognized_name': match_result['name'] if match_result else None,
                            'similarity': match_result['similarity'] if match_result else 0.0,
                            'is_known': match_result is not None,
                            'thumbnail_path': thumbnail_path,
                            'embedding': matched_face.embedding.tolist() if matched_face.embedding is not None else None
                        }
                    except Exception as e:
                        logger.error(f"Error processing face {i+1}: {str(e)}", exc_info=True)
                        face_info = {
                            'box_2d': box_2d,
                            'label': face_box.get('label', 'Face'),
                            'confidence': 0.0,
                            'recognized_name': None,
                            'is_known': False,
                            'thumbnail_path': None,
                            'embedding': None
                        }
                else:
                    logger.warning(f"Face {i+1} - No matching InsightFace detection found (best IoU: {best_iou:.3f})")
                    try:
                        # Enhanced fallback for edge faces
                        logger.info("No matching InsightFace detection found, using enhanced direct crop")
                        
                        # Check if face is at the edge
                        is_edge_face = (x1 <= 10 or y1 <= 10 or 
                                      x2 >= width - 10 or y2 >= height - 10)
                        
                        # Adjust padding based on edge position
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        # Calculate adaptive padding
                        if is_edge_face:
                            # Use asymmetric padding for edge faces
                            padding_left = min(x1, int(face_width * 0.3))
                            padding_right = min(width - x2, int(face_width * 0.3))
                            padding_top = min(y1, int(face_height * 0.3))
                            padding_bottom = min(height - y2, int(face_height * 0.3))
                            
                            # Add more padding on the non-edge sides
                            if x1 <= 10:  # Left edge
                                padding_right = int(face_width * 0.5)
                            if x2 >= width - 10:  # Right edge
                                padding_left = int(face_width * 0.5)
                            if y1 <= 10:  # Top edge
                                padding_bottom = int(face_height * 0.5)
                            if y2 >= height - 10:  # Bottom edge
                                padding_top = int(face_height * 0.5)
                        else:
                            # Standard padding for non-edge faces
                            padding_left = padding_right = int(face_width * 0.5)
                            padding_top = padding_bottom = int(face_height * 0.5)
                        
                        # Apply padding with bounds checking
                        crop_x1 = max(0, x1 - padding_left)
                        crop_y1 = max(0, y1 - padding_top)
                        crop_x2 = min(width, x2 + padding_right)
                        crop_y2 = min(height, y2 + padding_bottom)
                        
                        logger.info(f"Enhanced cropping for {'edge' if is_edge_face else 'normal'} face: "
                                   f"x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                        
                        # Crop with enhanced padding
                        face_crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                        if face_crop.size == 0:
                            raise HTTPException(status_code=400, detail="Invalid face crop dimensions")
                        
                        # Add padding to make the image square if needed
                        crop_height, crop_width = face_crop.shape[:2]
                        max_dim = max(crop_width, crop_height)
                        square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                        
                        # Center the face in the square image
                        start_x = (max_dim - crop_width) // 2
                        start_y = (max_dim - crop_height) // 2
                        square_image[start_y:start_y+crop_height, start_x:start_x+crop_width] = face_crop
                        
                        # Try different scales for edge faces
                        scales = [1.0, 0.8, 1.2] if is_edge_face else [1.0]
                        face_result = None
                        
                        for scale in scales:
                            try:
                                size = int(112 * scale)
                                scaled_face = cv2.resize(square_image, (size, size))
                                if size != 112:
                                    # Pad or crop to 112x112 if needed
                                    final_face = np.zeros((112, 112, 3), dtype=np.uint8)
                                    start = max(0, (112 - size) // 2)
                                    end = min(112, start + size)
                                    if size < 112:
                                        final_face[start:end, start:end] = scaled_face
                                    else:
                                        crop_start = (size - 112) // 2
                                        final_face = scaled_face[crop_start:crop_start+112, crop_start:crop_start+112]
                                else:
                                    final_face = scaled_face
                                
                                # Try to get embedding
                                result = self.app.get(final_face)
                                if result and len(result) > 0:
                                    face_result = result
                                    logger.info(f"Successfully got embedding with scale {scale}")
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to get embedding with scale {scale}: {str(e)}")
                                continue
                        
                        if not face_result:
                            # Try with the original crop as a last resort
                            logger.info("Trying original crop as last resort")
                            face_crop_original = image_rgb[y1:y2, x1:x2]
                            if face_crop_original.size > 0:
                                try:
                                    aligned_face_original = cv2.resize(face_crop_original, (112, 112))
                                    face_result = self.app.get(aligned_face_original)
                                except Exception as e:
                                    logger.error(f"Failed to get embedding from original crop: {str(e)}")
                        
                        if not face_result or len(face_result) == 0:
                            raise HTTPException(status_code=400, detail="Failed to get face embedding from crop")
                        
                        matched_face = face_result[0]
                        logger.info("Successfully obtained embedding from enhanced crop processing")
                        
                        face_info = {
                            'box_2d': box_2d,
                            'label': face_box.get('label', 'Face'),
                            'confidence': match_result['confidence'] if match_result else 0.0,
                            'recognized_name': match_result['name'] if match_result else None,
                            'similarity': match_result['similarity'] if match_result else 0.0,
                            'is_known': match_result is not None,
                            'thumbnail_path': thumbnail_path,
                            'embedding': matched_face.embedding.tolist() if matched_face.embedding is not None else None
                        }
                    except Exception as e:
                        logger.error(f"Error processing face {i+1}: {str(e)}", exc_info=True)
                        face_info = {
                            'box_2d': box_2d,
                            'label': face_box.get('label', 'Face'),
                            'confidence': 0.0,
                            'recognized_name': None,
                            'is_known': False,
                            'thumbnail_path': None,
                            'embedding': None
                        }
                
                face_details.append(face_info)
            
            return face_details
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}", exc_info=True)
            return face_boxes

def init_app() -> tuple[FastAPI, FaceDetector, OpenAIAnalyzer, FaceRecognizer, Jinja2Templates]:
    """Initialize the FastAPI application and required services."""
    # Load environment variables
    load_dotenv()
    logger.info(f"Loading credentials from: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    # Initialize Google Vision client
    vision_client = vision.ImageAnnotatorClient()
    
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID", None)
    )
    
    # Initialize services
    face_detector = FaceDetector(vision_client)
    openai_analyzer = OpenAIAnalyzer(openai_client)
    face_recognizer = FaceRecognizer()
    
    # Create FastAPI app
    app = FastAPI()
    
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/temp_faces', exist_ok=True)
    os.makedirs('faces', exist_ok=True)
    
    # Configure templates and static files
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/faces", StaticFiles(directory="faces"), name="faces")
    
    # Add CORS middleware with specific origins
    origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
        "null",  # Allow requests from local files
        "*"
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        allow_origin_regex="file://.*"  # Allow all file:// origins
    )
    
    return app, face_detector, openai_analyzer, face_recognizer, templates

# Initialize application and services
app, face_detector, openai_analyzer, face_recognizer, templates = init_app()

async def handle_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Handle file upload and return temporary file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        return temp_file.name, content

def create_face_thumbnail(image_path: str, box_2d: List[int]) -> Optional[str]:
    """Create a thumbnail from a face region in the image."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to read image for thumbnail creation")
            return None
            
        # Extract coordinates and ensure they are integers
        x1, y1, x2, y2 = [int(coord) for coord in box_2d]
        logger.info(f"Creating thumbnail with coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Validate coordinates
        height, width = image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            logger.error(f"Invalid coordinates: coordinates out of bounds (image size: {width}x{height})")
            return None
        
        if x2 <= x1 or y2 <= y1:
            logger.error("Invalid coordinates: end coordinates must be greater than start coordinates")
            return None
        
        # Add padding around the face (20% of face size)
        face_width = x2 - x1
        face_height = y2 - y1
        padding_x = int(face_width * 0.2)
        padding_y = int(face_height * 0.2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(width, x2 + padding_x)
        y2 = min(height, y2 + padding_y)
        
        logger.info(f"Adjusted coordinates with padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Crop the face region
        face_image = image[y1:y2, x1:x2]
        if face_image.size == 0:
            logger.error("Failed to crop face region: resulting image is empty")
            return None
        
        # Create temp_faces directory if it doesn't exist
        temp_faces_dir = os.path.join('static', 'temp_faces')
        os.makedirs(temp_faces_dir, exist_ok=True)
        
        # Create a unique filename
        timestamp = int(time.time() * 1000)
        filename = f"face_{timestamp}.jpg"
        thumbnail_path = os.path.join('temp_faces', filename)
        full_path = os.path.join('static', thumbnail_path)
        
        # Save the thumbnail
        success = cv2.imwrite(full_path, face_image)
        if not success:
            logger.error(f"Failed to save thumbnail to {full_path}")
            return None
            
        logger.info(f"Successfully created thumbnail at {thumbnail_path}")
        return thumbnail_path
        
    except Exception as e:
        logger.error(f"Error creating face thumbnail: {str(e)}", exc_info=True)
        return None

@app.post("/detect-faces")
async def detect_faces_endpoint(file: UploadFile = File(...), threshold: float = Form(0.8)) -> Dict[str, Any]:
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
        logger.info(f"Using recognition threshold: {threshold}")
        
        # Clean up old thumbnails before processing new ones
        cleanup_temp_faces()
        
        # Get file content and log size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)  # Convert to MB
        logger.info(f"Received image size: {file_size_mb:.2f} MB")
        
        # Write content to temp file
        temp_file_path = tempfile.mktemp(suffix=".jpg")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Get image dimensions
        image = cv2.imread(temp_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
            
        height, width = image.shape[:2]
        logger.info(f"Image dimensions: {width}x{height} pixels")
        
        # Temporarily set the recognition threshold
        original_threshold = face_recognizer.threshold
        face_recognizer.threshold = threshold
        
        # First, detect faces using Google Vision
        face_details = face_detector.detect_faces(temp_file_path)
        
        if not face_details:
            return {
                "faces": [],
                "status": "success"
            }
        
        logger.info(f"Google Vision detected {len(face_details)} faces")
        
        # Perform face recognition on detected faces
        recognized_faces = face_recognizer.recognize_face(temp_file_path, face_details)
        
        # If recognition was successful, use the recognition results
        if recognized_faces:
            face_details = recognized_faces
            logger.info("Face recognition completed successfully")
        else:
            # If recognition failed, just add thumbnails to the detection results
            logger.info("Face recognition failed, creating thumbnails from detection results")
            for face in face_details:
                try:
                    # Log face box coordinates
                    logger.info(f"Creating thumbnail for face at coordinates: {face['box_2d']}")
                    thumbnail_path = create_face_thumbnail(temp_file_path, face["box_2d"])
                    if thumbnail_path:
                        logger.info(f"Successfully created thumbnail at: {thumbnail_path}")
                    else:
                        logger.error("Failed to create thumbnail")
                    face["thumbnail_path"] = thumbnail_path
                    face["confidence"] = 0.0
                    face["is_known"] = False
                    face["recognized_name"] = None
                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}", exc_info=True)
                    face["thumbnail_path"] = None
                    face["confidence"] = 0.0
                    face["is_known"] = False
                    face["recognized_name"] = None
        
        return {
            "faces": face_details,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Restore original threshold
        face_recognizer.threshold = original_threshold
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

@app.get("/detect-faces", response_class=HTMLResponse)
async def detect_faces_page(request: Request):
    """Serve the face detection page."""
    return templates.TemplateResponse("detect-faces.html", {"request": request})

@app.post("/learn-face")
async def learn_face_endpoint(
    file: UploadFile = File(...),
    name: str = Form(...),
    face_coordinates: str = Form(...)  # Add face coordinates parameter
) -> Dict[str, Any]:
    """Endpoint for learning new faces."""
    temp_file_path = None
    try:
        temp_file_path, _ = await handle_uploaded_file(file)
        
        # Parse face coordinates
        try:
            coords = json.loads(face_coordinates)
            x1, y1, x2, y2 = [int(coord) for coord in coords]
            logger.info(f"Learning face at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid face coordinates format")
        
        # First detect all faces using Google Vision
        face_details = face_detector.detect_faces(temp_file_path)
        
        if not face_details:
            raise HTTPException(status_code=400, detail="No faces detected in the image")
        
        # Find the matching face based on coordinates
        matching_face = None
        for face in face_details:
            face_x1, face_y1, face_x2, face_y2 = face['box_2d']
            # Allow for some small differences in coordinates due to rounding
            if (abs(x1 - face_x1) < 10 and abs(y1 - face_y1) < 10 and 
                abs(x2 - face_x2) < 10 and abs(y2 - face_y2) < 10):
                matching_face = face
                break
        
        if not matching_face:
            raise HTTPException(status_code=400, detail="Selected face not found in detection results")
        
        # Read the image
        image = cv2.imread(temp_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Try InsightFace detection first
        faces = face_recognizer.app.get(image_rgb)
        matched_face = None
        
        if faces:
            # Find the matching InsightFace detection with a lower threshold for edge faces
            best_iou = 0
            for face in faces:
                face_bbox = face.bbox.astype(int)
                # Check if the face is at the edge
                is_edge_face = (x1 <= 10 or y1 <= 10 or 
                              x2 >= width - 10 or y2 >= height - 10)
                
                intersection = max(0, min(x2, face_bbox[2]) - max(x1, face_bbox[0])) * \
                             max(0, min(y2, face_bbox[3]) - max(y1, face_bbox[1]))
                union = (x2 - x1) * (y2 - y1) + \
                       (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]) - \
                       intersection
                iou = intersection / union if union > 0 else 0
                
                logger.info(f"Face IoU score with InsightFace detection: {iou:.3f} (edge face: {is_edge_face})")
                # Use a lower threshold for edge faces
                iou_threshold = 0.15 if is_edge_face else 0.3
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    matched_face = face
        
        if matched_face is None:
            # Enhanced fallback for edge faces
            logger.info("No matching InsightFace detection found, using enhanced direct crop")
            
            # Check if face is at the edge
            is_edge_face = (x1 <= 10 or y1 <= 10 or 
                          x2 >= width - 10 or y2 >= height - 10)
            
            # Adjust padding based on edge position
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Calculate adaptive padding
            if is_edge_face:
                # Use asymmetric padding for edge faces
                padding_left = min(x1, int(face_width * 0.3))
                padding_right = min(width - x2, int(face_width * 0.3))
                padding_top = min(y1, int(face_height * 0.3))
                padding_bottom = min(height - y2, int(face_height * 0.3))
                
                # Add more padding on the non-edge sides
                if x1 <= 10:  # Left edge
                    padding_right = int(face_width * 0.5)
                if x2 >= width - 10:  # Right edge
                    padding_left = int(face_width * 0.5)
                if y1 <= 10:  # Top edge
                    padding_bottom = int(face_height * 0.5)
                if y2 >= height - 10:  # Bottom edge
                    padding_top = int(face_height * 0.5)
            else:
                # Standard padding for non-edge faces
                padding_left = padding_right = int(face_width * 0.5)
                padding_top = padding_bottom = int(face_height * 0.5)
            
            # Apply padding with bounds checking
            crop_x1 = max(0, x1 - padding_left)
            crop_y1 = max(0, y1 - padding_top)
            crop_x2 = min(width, x2 + padding_right)
            crop_y2 = min(height, y2 + padding_bottom)
            
            logger.info(f"Enhanced cropping for {'edge' if is_edge_face else 'normal'} face: "
                       f"x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
            
            # Crop with enhanced padding
            face_crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_crop.size == 0:
                raise HTTPException(status_code=400, detail="Invalid face crop dimensions")
            
            # Add padding to make the image square if needed
            crop_height, crop_width = face_crop.shape[:2]
            max_dim = max(crop_width, crop_height)
            square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            
            # Center the face in the square image
            start_x = (max_dim - crop_width) // 2
            start_y = (max_dim - crop_height) // 2
            square_image[start_y:start_y+crop_height, start_x:start_x+crop_width] = face_crop
            
            # Try different scales for edge faces
            scales = [1.0, 0.8, 1.2] if is_edge_face else [1.0]
            face_result = None
            
            for scale in scales:
                try:
                    size = int(112 * scale)
                    scaled_face = cv2.resize(square_image, (size, size))
                    if size != 112:
                        # Pad or crop to 112x112 if needed
                        final_face = np.zeros((112, 112, 3), dtype=np.uint8)
                        start = max(0, (112 - size) // 2)
                        end = min(112, start + size)
                        if size < 112:
                            final_face[start:end, start:end] = scaled_face
                        else:
                            crop_start = (size - 112) // 2
                            final_face = scaled_face[crop_start:crop_start+112, crop_start:crop_start+112]
                    else:
                        final_face = scaled_face
                    
                    # Try to get embedding
                    result = face_recognizer.app.get(final_face)
                    if result and len(result) > 0:
                        face_result = result
                        logger.info(f"Successfully got embedding with scale {scale}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to get embedding with scale {scale}: {str(e)}")
                    continue
            
            if not face_result:
                # Try with the original crop as a last resort
                logger.info("Trying original crop as last resort")
                face_crop_original = image_rgb[y1:y2, x1:x2]
                if face_crop_original.size > 0:
                    try:
                        aligned_face_original = cv2.resize(face_crop_original, (112, 112))
                        face_result = face_recognizer.app.get(aligned_face_original)
                    except Exception as e:
                        logger.error(f"Failed to get embedding from original crop: {str(e)}")
            
            if not face_result or len(face_result) == 0:
                raise HTTPException(status_code=400, detail="Failed to get face embedding from crop")
            
            matched_face = face_result[0]
            logger.info("Successfully obtained embedding from enhanced crop processing")
            
        if matched_face.embedding is None:
            raise HTTPException(status_code=400, detail="Failed to generate face embedding")
            
        logger.info(f"Learning new face '{name}' with embedding shape: {matched_face.embedding.shape}")
        
        # Learn the new face
        face_recognizer.learn_new_face(matched_face.embedding, name)
        
        return {
            "status": "success",
            "message": f"Successfully learned face for {name}"
        }
        
    except Exception as e:
        logger.error(f"Error learning face: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

def cleanup_temp_faces():
    """Clean up temporary face thumbnails from the static/temp_faces directory."""
    try:
        temp_faces_dir = os.path.join('static', 'temp_faces')
        if os.path.exists(temp_faces_dir):
            for file in os.listdir(temp_faces_dir):
                file_path = os.path.join(temp_faces_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary face file {file_path}: {str(e)}")
            logger.info("Cleaned up temporary face thumbnails")
    except Exception as e:
        logger.error(f"Error cleaning up temporary face thumbnails: {str(e)}")

@app.post("/cleanup-temp-faces")
async def cleanup_temp_faces_endpoint() -> Dict[str, Any]:
    """Endpoint to clean up temporary face thumbnails."""
    try:
        cleanup_temp_faces()
        return {
            "status": "success",
            "message": "Temporary face thumbnails cleaned up"
        }
    except Exception as e:
        logger.error(f"Error in cleanup endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 