#!/usr/bin/env python3
"""
REST API for video personalization
Can be deployed as a microservice
"""

from flask import Flask, request, jsonify, send_file
import os
import json
import uuid
from pathlib import Path
import threading
import queue
from datetime import datetime
from personalization_pipeline import VideoPersonalizationPipeline

app = Flask(__name__)

# Job queue and status tracking
job_queue = queue.Queue()
job_status = {}
output_dir = Path("api_output")
output_dir.mkdir(exist_ok=True)


class PersonalizationJob:
    def __init__(self, job_id, video_path, replacements):
        self.job_id = job_id
        self.video_path = video_path
        self.replacements = replacements
        self.status = "queued"
        self.output_path = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None


def process_jobs():
    """Background worker to process personalization jobs"""
    while True:
        job = job_queue.get()
        if job is None:
            break
            
        try:
            job.status = "processing"
            job_status[job.job_id] = job
            
            # Create output directory for this job
            job_output_dir = output_dir / job.job_id
            job_output_dir.mkdir(exist_ok=True)
            
            # Process video
            pipeline = VideoPersonalizationPipeline(
                job.video_path, 
                str(job_output_dir)
            )
            output_path = pipeline.process(job.replacements)
            
            job.output_path = str(output_path)
            job.status = "completed"
            job.completed_at = datetime.now()
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now()
        
        job_status[job.job_id] = job


# Start background worker
worker_thread = threading.Thread(target=process_jobs, daemon=True)
worker_thread.start()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "video-personalization"})


@app.route('/personalize', methods=['POST'])
def personalize_video():
    """
    Personalize a video with custom variables
    
    Expected JSON payload:
    {
        "video_url": "https://example.com/video.mp4" (optional),
        "video_path": "/path/to/local/video.mp4" (optional),
        "replacements": {
            "customer_name": "John Smith",
            "destination": "Tokyo"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        if 'video_path' not in data and 'video_url' not in data:
            return jsonify({"error": "Either video_path or video_url required"}), 400
            
        if 'replacements' not in data:
            return jsonify({"error": "replacements field required"}), 400
        
        # Get video path
        if 'video_path' in data:
            video_path = data['video_path']
            if not os.path.exists(video_path):
                return jsonify({"error": "Video file not found"}), 404
        else:
            # Download from URL (not implemented in this example)
            return jsonify({"error": "URL download not implemented"}), 501
        
        # Create job
        job_id = str(uuid.uuid4())
        job = PersonalizationJob(job_id, video_path, data['replacements'])
        
        # Add to queue
        job_queue.put(job)
        job_status[job_id] = job
        
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Video personalization job created"
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a personalization job"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    job = job_status[job_id]
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
    }
    
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
    
    if job.status == "completed" and job.output_path:
        response["output_path"] = job.output_path
        response["download_url"] = f"/download/{job_id}"
    
    if job.status == "failed" and job.error:
        response["error"] = job.error
    
    return jsonify(response)


@app.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download personalized video"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    job = job_status[job_id]
    
    if job.status != "completed" or not job.output_path:
        return jsonify({"error": "Video not ready"}), 400
    
    if not os.path.exists(job.output_path):
        return jsonify({"error": "Output file not found"}), 404
    
    return send_file(job.output_path, as_attachment=True)


@app.route('/batch', methods=['POST'])
def batch_personalize():
    """
    Process multiple personalizations in batch
    
    Expected JSON payload:
    {
        "video_path": "/path/to/template/video.mp4",
        "personalizations": [
            {
                "customer_name": "John Smith",
                "destination": "Tokyo"
            },
            {
                "customer_name": "Jane Doe",
                "destination": "Paris"
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'video_path' not in data:
            return jsonify({"error": "video_path required"}), 400
            
        if 'personalizations' not in data:
            return jsonify({"error": "personalizations array required"}), 400
        
        video_path = data['video_path']
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404
        
        # Create jobs for each personalization
        job_ids = []
        for personalization in data['personalizations']:
            job_id = str(uuid.uuid4())
            job = PersonalizationJob(job_id, video_path, personalization)
            job_queue.put(job)
            job_status[job_id] = job
            job_ids.append(job_id)
        
        return jsonify({
            "batch_id": str(uuid.uuid4()),
            "job_ids": job_ids,
            "message": f"Created {len(job_ids)} personalization jobs"
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)