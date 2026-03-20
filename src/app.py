import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from flask_cors import CORS

# src/ directory and project root (one level up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = PROJECT_ROOT / "instance"
DB_PATH = INSTANCE_DIR / "data.db"

# Allow importing backend modules before importing src modules that depend on them.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from models import db
from routes import register_routes
from bootstrap import initialize_offline_data_pipeline

INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

# Serve React build files from <project_root>/frontend/dist
app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / "frontend" / "dist"),
    static_url_path="",
    instance_path=str(INSTANCE_DIR),
)
CORS(app)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH.as_posix()}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

# Prepare DB + vector artifacts
initialize_offline_data_pipeline(app=app, project_root=PROJECT_ROOT)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
