from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_sqlalchemy import SQLAlchemy
from config import config
from models import db, PipelineRun, TransitCandidate
from datetime import datetime, timezone
import os
import sys
import threading
import pandas as pd
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import run_pipeline, fetch_kepler_llc_from_archive

def create_app(config_name='development'):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    return app

app = create_app(os.environ.get('FLASK_ENV', 'development'))

@app.route('/')
def index():
    """Home page with pipeline input form."""
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def api_run_pipeline():
    try:
        data = request.get_json()
        
        # Validate parameters
        kernel_size = int(data.get('kernel_size', 101))
        prominence = float(data.get('prominence', 0.0002))
        review_threshold = float(data.get('review_threshold', 70.0))
        review_now_threshold = float(data.get('review_now_threshold', 75.0))
        min_peaks = int(data.get('min_peaks', 20))
        max_peaks = int(data.get('max_peaks', 400))
        source = data.get('source', 'archive')  # 'archive' or 'upload'
        
        # Create database record
        run = PipelineRun(
            kernel_size=kernel_size,
            prominence=prominence,
            review_threshold=review_threshold,
            review_now_threshold=review_now_threshold,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            status='pending'
        )
        db.session.add(run)
        db.session.commit()
        
        # Start background job
        if source == 'archive':
            archive_count = int(data.get('archive_count', 10))
            thread = threading.Thread(
                target=_run_pipeline_job,
                args=(run.id, source, archive_count, kernel_size, prominence, 
                      review_threshold, review_now_threshold, min_peaks, max_peaks)
            )
        else:  # upload
            uploaded_files = data.get('uploaded_files', [])
            thread = threading.Thread(
                target=_run_pipeline_job,
                args=(run.id, source, uploaded_files, kernel_size, prominence,
                      review_threshold, review_now_threshold, min_peaks, max_peaks)
            )
        
        thread.daemon = True
        thread.start()
        
        return jsonify({'run_id': run.id, 'status': 'pending'}), 202
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/run/<run_id>', methods=['GET'])
def api_get_run_status(run_id):
    run = PipelineRun.query.get(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    return jsonify({
        'id': run.id,
        'status': run.status,
        'timestamp': run.run_timestamp.isoformat(),
        'processed': run.processed_count,
        'succeeded': run.succeeded_count,
        'failed': run.failed_count,
        'total_dips': run.total_candidate_dips,
        'results_ready': run.status == 'completed'
    }), 200

@app.route('/results/<run_id>')
def results_page(run_id):
    """Display results for a pipeline run."""
    run = PipelineRun.query.get(run_id)
    if not run:
        return "Run not found", 404
    
    return render_template('results.html', run_id=run_id, run=run)

@app.route('/api/results/<run_id>', methods=['GET'])
def api_get_results(run_id):
    """Get paginated results for a run."""
    run = PipelineRun.query.get(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    filter_status = request.args.get('review_status', None)
    
    query = TransitCandidate.query.filter_by(run_id=run_id)
    if filter_status:
        query = query.filter_by(review_status=filter_status)
    
    query = query.order_by(TransitCandidate.final_ranking_score.desc())
    
    paginated = query.paginate(page=page, per_page=per_page)
    
    results = [{
        'target': r.target_name,
        'score': round(r.final_ranking_score, 2),
        'quality_flag': r.quality_flag,
        'period': round(r.estimated_period_days, 3) if r.estimated_period_days else None,
        'stability': r.period_stability_flag,
        'review_status': r.review_status,
        'mean_depth': round(r.mean_transit_depth, 4) if r.mean_transit_depth else None,
    } for r in paginated.items]
    
    return jsonify({
        'results': results,
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page
    }), 200

@app.route('/api/results/<run_id>/download')
def api_download_results_csv(run_id):
    """Download full results CSV."""
    run = PipelineRun.query.get(run_id)
    if not run or not run.results_csv_path or not os.path.exists(run.results_csv_path):
        return jsonify({'error': 'Results not found'}), 404
    
    return send_file(run.results_csv_path, as_attachment=True, 
                     download_name=f'transit_results_{run_id}.csv')

@app.route('/history')
def history_page():
    """Display run history."""
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def api_get_history():
    """Get paginated run history."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    runs = PipelineRun.query.order_by(PipelineRun.run_timestamp.desc()).paginate(
        page=page, per_page=per_page
    )
    
    history = [{
        'id': r.id,
        'timestamp': r.run_timestamp.isoformat(),
        'status': r.status,
        'processed': r.processed_count,
        'succeeded': r.succeeded_count,
        'success_rate': round(r.succeeded_count / r.processed_count * 100, 1) if r.processed_count else 0
    } for r in runs.items]
    
    return jsonify({
        'history': history,
        'total': runs.total,
        'pages': runs.pages
    }), 200


def _run_pipeline_job(
        run_id, 
        source, 
        targets_or_count, 
        kernel_size, 
        prominence, 
        review_threshold, 
        review_now_threshold, 
        min_peaks, 
        max_peaks
):
    try:
        with app.app_context():
            run = PipelineRun.query.get(run_id)
            run.status = 'running'
            db.session.commit()
            
            # Get targets
            if source == 'archive':
                print(f"[{run_id}] Fetching {targets_or_count} targets from Kepler archive...")
                downloads_df = fetch_kepler_llc_from_archive(
                    target_count=targets_or_count,
                    download_dir=app.config['UPLOAD_FOLDER'],
                    max_buckets=20,
                    randomize=True,
                    random_seed=None
                )
                targets = downloads_df['filename'].tolist()
            else:
                targets = targets_or_count
                downloads_df = pd.DataFrame()
            
            print(f"[{run_id}] Running pipeline on {len(targets)} targets...")
            
            # Run pipeline
            results_df, failures, summary = run_pipeline(
                targets=targets,
                data_dir=app.config['UPLOAD_FOLDER'],
                export_csv=True,
                output_csv=os.path.join(app.config['RESULTS_FOLDER'], f'results_{run_id}.csv'),
                show_plot=False,
                kernel_size=kernel_size,
                prominence=prominence,
                top_candidates_csv=os.path.join(app.config['RESULTS_FOLDER'], f'top_{run_id}.csv'),
                caution_candidates_csv=os.path.join(app.config['RESULTS_FOLDER'], f'caution_{run_id}.csv'),
                review_threshold=review_threshold,
                review_now_threshold=review_now_threshold,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
                downloads_df=downloads_df
            )
            
            # Update database
            run.status = 'completed'
            run.processed_count = summary['processed']
            run.succeeded_count = summary['succeeded']
            run.failed_count = summary['failed']
            run.total_candidate_dips = summary['total_candidate_dips']
            run.results_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f'results_{run_id}.csv')
            run.top_candidates_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f'top_{run_id}.csv')
            run.caution_candidates_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f'caution_{run_id}.csv')
            
            # Store individual results
            for _, row in results_df.iterrows():
                candidate = TransitCandidate(
                    run_id=run_id,
                    target_name=row['target'],
                    num_peaks=int(row['num_peaks']),
                    quality_flag=row['quality_flag'],
                    estimated_period_days=float(row['estimated_period_days']) if pd.notna(row['estimated_period_days']) else None,
                    period_stability_cv=float(row['period_stability_cv']) if pd.notna(row['period_stability_cv']) else None,
                    period_stability_flag=row['period_stability_flag'],
                    mean_transit_depth=float(row['mean_transit_depth']) if pd.notna(row['mean_transit_depth']) else None,
                    median_transit_depth=float(row['median_transit_depth']) if pd.notna(row['median_transit_depth']) else None,
                    max_transit_depth=float(row['max_transit_depth']) if pd.notna(row['max_transit_depth']) else None,
                    first_transit_time=float(row['first_transit_time']) if pd.notna(row['first_transit_time']) else None,
                    final_ranking_score=float(row['final_ranking_score']),
                    review_status=row['review_status'],
                    review_reason=row['review_reason'],
                    source_url=row.get('source_url'),
                    local_path=row.get('local_path'),
                    mean_detrended_flux=float(row['mean_detrended_flux']) if pd.notna(row['mean_detrended_flux']) else None,
                    std_detrended_flux=float(row['std_detrended_flux']) if pd.notna(row['std_detrended_flux']) else None,
                )
                db.session.add(candidate)
            
            db.session.commit()
            print(f"[{run_id}] Pipeline completed successfully")
            
    except Exception as e:
        with app.app_context():
            run = PipelineRun.query.get(run_id)
            run.status = 'failed'
            run.error_message = str(e)
            db.session.commit()
            print(f"[{run_id}] Pipeline failed: {str(e)}")

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8080)