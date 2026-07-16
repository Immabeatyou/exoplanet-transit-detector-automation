from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import uuid

db = SQLAlchemy()

class PipelineRun(db.Model):
    __tablename__ = 'pipeline_runs'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    kernel_size = db.Column(db.Integer, nullable=False, default=101)
    prominence = db.Column(db.Float, nullable=False, default=0.0002)
    review_threshold = db.Column(db.Float, nullable=False, default=70.0)
    review_now_threshold = db.Column(db.Float, nullable=False, default=75.0)
    min_peaks = db.Column(db.Integer, nullable=False, default=20)
    max_peaks = db.Column(db.Integer, nullable=False, default=400)

    processed_count = db.Column(db.Integer, default=0)
    succeeded_count = db.Column(db.Integer, default=0)
    failed_count = db.Column(db.Integer, default=0)
    total_candidate_dips = db.Column(db.Integer, default=0)

    results_csv_path = db.Column(db.String(255))
    top_candidates_csv_path = db.Column(db.String(255))
    caution_candidates_csv_path = db.Column(db.String(255))

    status = db.Column(db.String(20), default='pending')
    error_message = db.Column(db.Text)

    results = db.relationship('TransitCandidate', backref='run', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f"<PipelineRun {self.id} ({self.status})>"

class TransitCandidate(db.Model):
    __tablename__ = 'transit_candidates'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = db.Column(db.String(36), db.ForeignKey('pipeline_runs.id'), nullable=False)

    target_name = db.Column(db.String(100), nullable=False)

    num_peaks = db.Column(db.Integer)
    quality_flag = db.Column(db.String(20))
    estimated_period_days = db.Column(db.Float)
    period_stability_cv = db.Column(db.Float)
    period_stability_flag = db.Column(db.String(20))

    mean_transit_depth = db.Column(db.Float)
    median_transit_depth = db.Column(db.Float)
    max_transit_depth = db.Column(db.Float)
    first_transit_time = db.Column(db.Float)

    final_ranking_score = db.Column(db.Float, nullable=False)
    review_status = db.Column(db.String(30), nullable=False)
    review_reason = db.Column(db.Text)

    source_url = db.Column(db.String(500))
    local_path = db.Column(db.String(500))
    mean_detrended_flux = db.Column(db.Float)
    std_detrended_flux = db.Column(db.Float)

    def __repr__(self):
        return f"<TransitCandidate {self.target_name} (score={self.final_ranking_score:.1f})>"