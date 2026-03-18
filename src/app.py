import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask

from flask_cors import CORS

# src/ directory and project root (one level up)
project_root = Path(__file__).resolve().parent.parent

# Allow importing backend modules before importing src modules that depend on them.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from models import db
from routes import register_routes
from bootstrap import initialize_offline_data_pipeline

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=str(project_root / "frontend" / "dist"),
    static_url_path="")
CORS(app)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

# Prepare DB + vector artifacts
initialize_offline_data_pipeline(app=app, project_root=project_root)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
