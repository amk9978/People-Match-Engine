import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from services.data.embedding_builder import EmbeddingBuilder
from services.graph_matcher import GraphMatcher
from services.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for handling graph analysis operations"""

    def __init__(self):
        self.jobs = {}
        self.cache = RedisEmbeddingCache(key_prefix="file_mapping")
        self.results_cache = RedisEmbeddingCache(key_prefix="job_results")

    def _serialize_numpy(self, obj):
        """Convert numpy objects and other non-JSON types to JSON serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy(item) for item in obj]
        return obj

    def _store_file_mapping(self, file_id: str, filename: str):
        """Store file_id to filename mapping in Redis"""
        try:
            cache_key = f"file_id_to_name_{file_id}"
            self.cache.set(cache_key, filename)

            # Also store reverse mapping for lookups
            reverse_key = (
                f"filename_to_id_{filename.replace('/', '_').replace(' ', '_')}"
            )
            self.cache.set(reverse_key, file_id)
        except Exception as e:
            logger.info(f"Warning: Could not store file mapping: {e}")

    def get_filename_by_file_id(self, file_id: str) -> Optional[str]:
        """Get filename by file_id from Redis"""
        try:
            cache_key = f"file_id_to_name_{file_id}"
            return self.cache.get(cache_key)
        except Exception as e:
            logger.info(
                f"Warning: Could not retrieve filename for file_id {file_id}: {e}"
            )
            return None

    def get_file_id_by_filename(self, filename: str) -> Optional[str]:
        """Get file_id by filename from Redis"""
        try:
            reverse_key = (
                f"filename_to_id_{filename.replace('/', '_').replace(' ', '_')}"
            )
            return self.cache.get(reverse_key)
        except Exception as e:
            logger.info(
                f"Warning: Could not retrieve file_id for filename {filename}: {e}"
            )
            return None

    def _store_job_result(self, job_id: str, result: Dict):
        """Store completed job result in Redis"""
        try:
            import json

            cache_key = f"result_{job_id}"
            self.results_cache.set(cache_key, json.dumps(result))
        except Exception as e:
            logger.info(f"Warning: Could not store job result for {job_id}: {e}")

    def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Get stored job result from Redis"""
        try:
            import json

            cache_key = f"result_{job_id}"
            result_str = self.results_cache.get(cache_key)
            return json.loads(result_str) if result_str else None
        except Exception as e:
            logger.info(f"Warning: Could not retrieve job result for {job_id}: {e}")
            return None

    def _store_user_job(self, user_id: str, job_id: str, job_data: Dict):
        """Store job for user in Redis"""
        try:
            import json
            # Store individual job
            job_key = f"user_job_{user_id}_{job_id}"
            job_data_serialized = {**job_data}
            job_data_serialized['timestamp'] = job_data['timestamp'].isoformat()
            self.results_cache.set(job_key, json.dumps(job_data_serialized))
            
            # Add to user's job list
            user_jobs_key = f"user_jobs_{user_id}"
            existing_jobs = self.results_cache.get(user_jobs_key)
            job_list = json.loads(existing_jobs) if existing_jobs else []
            
            if job_id not in job_list:
                job_list.append(job_id)
                self.results_cache.set(user_jobs_key, json.dumps(job_list))
        except Exception as e:
            logger.info(f"Warning: Could not store user job {job_id} for user {user_id}: {e}")

    def get_jobs_by_user(self, user_id: str) -> List[Dict]:
        """Get all jobs for a specific user"""
        try:
            import json
            user_jobs_key = f"user_jobs_{user_id}"
            job_list_str = self.results_cache.get(user_jobs_key)
            
            if not job_list_str:
                return []
            
            job_ids = json.loads(job_list_str)
            jobs = []
            
            for job_id in job_ids:
                job_key = f"user_job_{user_id}_{job_id}"
                job_data_str = self.results_cache.get(job_key)
                if job_data_str:
                    job_data = json.loads(job_data_str)
                    job_data['job_id'] = job_id
                    jobs.append(job_data)
            
            # Sort by timestamp (newest first)
            jobs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return jobs
            
        except Exception as e:
            logger.info(f"Warning: Could not retrieve jobs for user {user_id}: {e}")
            return []

    def create_job(
        self,
        filename: str,
        min_density: Optional[float] = None,
        prompt: Optional[str] = None,
        file_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Create a new analysis job"""
        job_id = str(uuid.uuid4())

        self._store_file_mapping(file_id, filename)

        job_data = {
            "status": "queued",
            "progress": "Job created",
            "result": None,
            "error": None,
            "timestamp": datetime.now(),
            "filename": filename,
            "file_id": file_id,
            "min_density": min_density,
            "prompt": prompt,
            "user_id": user_id,
        }

        self.jobs[job_id] = job_data
        
        # Store job in Redis with user_id indexing if user_id provided
        if user_id:
            self._store_user_job(user_id, job_id, job_data)

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict:
        """Get all jobs summary"""
        return {
            "jobs": [
                {
                    "job_id": job_id,
                    "status": data["status"],
                    "timestamp": data["timestamp"],
                    "filename": data.get("filename", "unknown"),
                }
                for job_id, data in self.jobs.items()
            ]
        }

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: str = None,
        result: Dict = None,
        error: str = None,
    ):
        """Update job status"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            if progress:
                self.jobs[job_id]["progress"] = progress
            if result:
                self.jobs[job_id]["result"] = result
            if error:
                self.jobs[job_id]["error"] = error
            self.jobs[job_id]["timestamp"] = datetime.now()

    async def run_analysis(
        self,
        job_id: str,
        csv_path: str,
        notification_service,
        min_density: float = None,
        prompt: Optional[str] = None,
    ):
        """Run graph analysis with progress notifications"""
        try:
            self.update_job_status(job_id, "processing", "Initializing GraphMatcher...")
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Initializing GraphMatcher..."},
            )

            matcher = GraphMatcher(csv_path, min_density)

            self.update_job_status(job_id, "processing", "Loading data...")
            await notification_service.broadcast_job_update(
                job_id, {"status": "processing", "progress": "Loading data..."}
            )
            matcher.load_data()

            # STEP 1: Preprocess tags with semantic deduplication
            self.update_job_status(
                job_id, "processing", "Preprocessing tags and embeddings..."
            )
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Preprocessing tags and embeddings...",
                },
            )

            embedding_builder = EmbeddingBuilder()
            await embedding_builder.preprocess_tags(csv_path, similarity_threshold=0.75)

            # STEP 2: Create feature embeddings
            self.update_job_status(
                job_id, "processing", "Creating feature embeddings..."
            )
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Creating feature embeddings..."},
            )
            feature_embeddings = await matcher.embed_features()

            # STEP 3: Create modern graph with weight tuning
            self.update_job_status(job_id, "processing", "Building optimized graph...")
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Building optimized graph with weight tuning...",
                },
            )

            file_id = self.jobs[job_id]["file_id"]

            # Create modern graph using GraphBuilder's create_graph with user prompt for weight tuning
            await matcher.create_graph(feature_embeddings, file_id, prompt)
            self.update_job_status(job_id, "processing", "Finding dense subgraph...")
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Finding dense subgraph..."},
            )
            largest_dense_nodes, density = matcher.find_largest_dense_subgraph()

            self.update_job_status(job_id, "processing", "Analyzing results...")
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Analyzing results and generating insights...",
                },
            )
            result = matcher.get_subgraph_info(largest_dense_nodes, feature_embeddings)

            recommendations = matcher.get_expansion_recommendations(
                largest_dense_nodes, feature_embeddings
            )
            result["expansion_recommendations"] = recommendations

            if os.path.exists(csv_path):
                os.remove(csv_path)

            serialized_result = self._serialize_numpy(result)

            # Store result in Redis for future retrieval
            self._store_job_result(job_id, serialized_result)

            self.update_job_status(
                job_id, "completed", "Analysis complete", serialized_result
            )
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "completed",
                    "progress": "Analysis complete",
                    "result": serialized_result,
                },
            )

        except Exception as e:
            self.update_job_status(job_id, "failed", f"Error: {str(e)}", error=str(e))

            await notification_service.broadcast_job_update(
                job_id,
                {"status": "failed", "error": str(e), "progress": f"Error: {str(e)}"},
            )

            if os.path.exists(csv_path):
                os.remove(csv_path)


class FileService:
    """Service for handling file operations"""

    @staticmethod
    async def save_uploaded_file(file) -> str:
        """Save uploaded file to temporary location"""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            return tmp_file.name

    @staticmethod
    def validate_csv_file(filename: str) -> bool:
        """Validate if file is CSV"""
        return filename.lower().endswith(".csv")


analysis_service = AnalysisService()
file_service = FileService()
