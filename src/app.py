from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from asgiref.wsgi import WsgiToAsgi

# Load environment variables
load_dotenv()

# Initialize Flask app
flask_app = Flask(__name__)

# Enable CORS
CORS(flask_app)

# Sample route
@flask_app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

# Create ASGI app
app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    flask_app.run(host='0.0.0.0', port=port, debug=True) 