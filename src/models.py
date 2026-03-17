from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class GuardianArticle(db.Model):
    __tablename__ = "guardian_articles"

    id = db.Column(db.String(255), primary_key=True)
    title = db.Column(db.String(512), nullable=False, index=True)
    summary = db.Column(db.Text, nullable=False, default="")
    date = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    url = db.Column(db.String(1024), nullable=False)
    author_raw = db.Column(db.String(256), nullable=False, default="")
    contributors = db.Column(db.JSON, nullable=False, default=list)
    n_contributors = db.Column(db.Integer, nullable=False, default=0)
    keywords = db.Column(db.JSON, nullable=False, default=list)
    body_text = db.Column(db.Text, nullable=False, default="")
    # Kept for backward compatibility; not used by app routes/UI.
    section_id = db.Column(db.String(128), nullable=False, default="")
    section_name = db.Column(db.String(128), nullable=False, default="")
    year = db.Column(db.Integer, nullable=False, index=True)

    def __repr__(self):
        return f"GuardianArticle<{self.id}>"
