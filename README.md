# Gallery AI Application

A full-stack application using Flask (Python) backend and React + Vite frontend.

## Project Structure
```
gallery-ai/
├── src/
    ├── venv/                  # Python virtual environment
    ├── app.py                 # Flask application
    ├── requirements.txt       # Python dependencies
    ├── .env                   # Environment variables
    └── frontend/              # React + Vite application
```

## Setup Instructions

### Backend Setup
1. Create and activate virtual environment:
   ```bash
   python -m venv src/venv
   source src/venv/Scripts/activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   cd src
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   # Option 1: Using Flask development server
   python app.py

   # Option 2: Using Uvicorn (Recommended for better performance)
   uvicorn app:app --reload --port 5000
   ```
   The backend will run on http://localhost:5000

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd src/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:5173

## Development

- Backend API endpoints are available at http://localhost:5000/api/
- Frontend development server includes hot module replacement (HMR)
- The backend includes CORS support for frontend development
- Uvicorn provides better performance and automatic reloading for development