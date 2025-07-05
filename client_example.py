#!/usr/bin/env python3
"""
Example client for video personalization API
"""

import requests
import time
import json
import sys


class VideoPersonalizationClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def personalize_video(self, video_path, customer_name, destination):
        """Submit a video for personalization"""
        
        # Submit job
        response = requests.post(
            f"{self.base_url}/personalize",
            json={
                "video_path": video_path,
                "replacements": {
                    "customer_name": customer_name,
                    "destination": destination
                }
            }
        )
        
        if response.status_code != 202:
            raise Exception(f"Failed to submit job: {response.text}")
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Job submitted: {job_id}")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{self.base_url}/status/{job_id}")
            status_data = status_response.json()
            
            print(f"Status: {status_data['status']}")
            
            if status_data["status"] == "completed":
                print(f"Video ready: {status_data['download_url']}")
                return job_id, status_data
            
            elif status_data["status"] == "failed":
                raise Exception(f"Job failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(2)  # Poll every 2 seconds
    
    def download_video(self, job_id, output_path):
        """Download the personalized video"""
        response = requests.get(f"{self.base_url}/download/{job_id}", stream=True)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download video: {response.text}")
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded: {output_path}")
    
    def batch_personalize(self, video_path, personalizations):
        """Submit batch personalization job"""
        response = requests.post(
            f"{self.base_url}/batch",
            json={
                "video_path": video_path,
                "personalizations": personalizations
            }
        )
        
        if response.status_code != 202:
            raise Exception(f"Failed to submit batch: {response.text}")
        
        return response.json()


def main():
    # Example usage
    client = VideoPersonalizationClient()
    
    # Single personalization
    print("=== Testing API Health ===")
    try:
        response = requests.get(f"{client.base_url}/health")
        print(f"Health check: {response.json()}")
    except Exception as e:
        print(f"API server not running: {e}")
        print("Please start the API server with: python api_server.py")
        return
    
    print("\n=== Single Personalization ===")
    try:
        video_path = "/Users/sankalpthakur/Projects/Projects - Emtribe/personalise_video/VIDEO-2025-07-05-16-44-05.mp4"
        
        job_id, status = client.personalize_video(
            video_path=video_path,
            customer_name="Alice Johnson",
            destination="London"
        )
        
        # Download result
        output_path = f"personalized_{job_id}.mp4"
        client.download_video(job_id, output_path)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()