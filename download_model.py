import os
import logging
import warnings
from insightface.app import FaceAnalysis
import insightface

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress InsightFace info messages
logging.getLogger('insightface').setLevel(logging.WARNING)

def download_insightface_model():
    """Download InsightFace buffalo_l model."""
    try:
        logger.info("Initializing InsightFace model download...")
        # Create models directory if it doesn't exist
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Set the INSIGHTFACE_HOME environment variable
        os.environ['INSIGHTFACE_HOME'] = model_dir
        logger.info(f"Model directory set to: {model_dir}")
        
        # Initialize FaceAnalysis which will trigger the model download
        logger.info("Attempting to initialize with CUDA...")
        app = FaceAnalysis(name='buffalo_l', root=model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        try:
            app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 means using first GPU
            logger.info("Model initialized successfully with CUDA!")
        except Exception as e:
            logger.error(f"Failed to initialize with CUDA: {str(e)}")
            logger.warning("Falling back to CPU")
            app = FaceAnalysis(name='buffalo_l', root=model_dir, providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 means using CPU
            logger.info("Model initialized successfully with CPU")
        
        logger.info("Model downloaded and prepared successfully!")
        
        # List downloaded files
        buffalo_dir = os.path.join(model_dir, 'models', 'buffalo_l')
        if os.path.exists(buffalo_dir):
            model_files = os.listdir(buffalo_dir)
            logger.info("Downloaded model files:")
            for file in model_files:
                logger.info(f"- {file}")
        else:
            logger.warning(f"Model directory not found at expected path: {buffalo_dir}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_insightface_model() 