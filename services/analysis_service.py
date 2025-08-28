import logging
import os
from typing import Dict, List, Optional

import numpy as np

from models.job import JobStatus
from services.graph.graph_builder import GraphBuilder
from services.job_service import JobService
from services.preprocessing.embedding_builder import EmbeddingBuilder
from services.redis.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for handling graph analysis operations"""

    def __init__(self):
        self.results_cache = RedisEmbeddingCache(key_prefix="job_results")
        self.job_service = JobService()

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
        """Legacy method - file mappings now handled by FileService"""

        pass

    def get_filename_by_file_id(self, file_id: str) -> Optional[str]:
        """Get filename by file_id - now uses FileService"""
        try:
            from services.file_service import FileService

            file_service = FileService()
            file_obj = file_service.get_file(file_id)
            return file_obj.filename if file_obj else None
        except Exception as e:
            logger.info(
                f"Warning: Could not retrieve filename for file_id {file_id}: {e}"
            )
            return None

    def get_file_id_by_filename(self, user_id: str, filename: str) -> Optional[str]:
        """Get file_id by filename - now uses FileService"""
        try:
            from services.file_service import FileService

            file_service = FileService()
            file_obj = file_service.get_file_by_name(user_id, filename)
            return file_obj.file_id if file_obj else None
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

            job_key = f"user_job_{user_id}_{job_id}"
            job_data_serialized = {**job_data}
            job_data_serialized["timestamp"] = job_data["timestamp"].isoformat()
            self.results_cache.set(job_key, json.dumps(job_data_serialized))

            user_jobs_key = f"user_jobs_{user_id}"
            existing_jobs = self.results_cache.get(user_jobs_key)
            job_list = json.loads(existing_jobs) if existing_jobs else []

            if job_id not in job_list:
                job_list.append(job_id)
                self.results_cache.set(user_jobs_key, json.dumps(job_list))
        except Exception as e:
            logger.info(
                f"Warning: Could not store user job {job_id} for user {user_id}: {e}"
            )

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
                    job_data["job_id"] = job_id
                    jobs.append(job_data)

            jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return jobs

        except Exception as e:
            logger.info(f"Warning: Could not retrieve jobs for user {user_id}: {e}")
            return []

    async def _update_job_and_notify(
        self,
        job_id: str,
        status: str,
        progress: str = None,
        notification_service=None,
        result: Dict = None,
        error: str = None,
    ):
        """Update job status via JobService and broadcast notification"""
        try:

            status_enum = {
                "processing": JobStatus.RUNNING,
                "completed": JobStatus.COMPLETED,
                "failed": JobStatus.FAILED,
                "queued": JobStatus.QUEUED,
            }.get(status)

            if status_enum:
                self.job_service.update_job_status(job_id, status_enum, error)

            if progress:
                progress_value = self._extract_progress_value(progress)
                self.job_service.update_job_progress(job_id, progress_value)

            if result:
                self._store_job_result_via_service(job_id, result)

            if notification_service:
                notification_data = {"status": status}
                if progress:
                    notification_data["progress"] = progress
                if result:
                    notification_data["result"] = self._serialize_numpy(result)
                if error:
                    notification_data["error"] = error

                await notification_service.broadcast_job_update(
                    job_id, notification_data
                )

        except Exception as e:
            logger.error(f"Error updating job {job_id}: {e}")

    def _extract_progress_value(self, progress: str) -> float:
        """Extract numeric progress from string"""
        if "Step" in progress:
            try:
                parts = progress.split()
                for part in parts:
                    if "/" in part:
                        current, total = part.split("/")
                        return (float(current) / float(total)) * 100
            except:
                pass
        elif "%" in progress:
            try:
                return float(progress.replace("%", ""))
            except:
                pass
        return 0.0

    def _store_job_result_via_service(self, job_id: str, result: Dict):
        """Store job result via JobService"""
        from models.job import JobResult

        job_result = JobResult(
            job_id=job_id,
            result_data=self._serialize_numpy(result),
            metrics=result.get("metrics", {}),
            artifacts=result.get("artifacts", []),
            execution_time=result.get("execution_time", 0.0),
        )
        self.job_service.set_job_result(job_id, job_result)

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
            await self._update_job_and_notify(
                job_id,
                "processing",
                "Initializing GraphBuilder...",
                notification_service,
            )

            graph_builder = GraphBuilder(csv_path, min_density)

            await self._update_job_and_notify(
                job_id, "processing", "Loading data...", notification_service
            )
            graph_builder.load_data()

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Preprocessing tags and embeddings...",
                notification_service,
            )

            embedding_builder = EmbeddingBuilder()
            await embedding_builder.preprocess_tags(csv_path, similarity_threshold=0.75)

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Creating feature embeddings...",
                notification_service,
            )
            feature_embeddings = await graph_builder.embed_features()

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Building optimized graph with weight tuning...",
                notification_service,
            )

            job = self.job_service.get_job(job_id)
            file_id = job.file_id if job else None
            if not file_id:
                raise ValueError(f"No file_id found for job {job_id}")

            await graph_builder.create_graph(feature_embeddings, job_id, prompt)
            await self._update_job_and_notify(
                job_id, "processing", "Finding dense subgraph...", notification_service
            )
            largest_dense_nodes, density = graph_builder.find_largest_dense_subgraph()

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Analyzing results and generating insights...",
                notification_service,
            )
            result = graph_builder.get_subgraph_info(
                largest_dense_nodes, feature_embeddings
            )

            result["expansion_recommendations"] = []

            if os.path.exists(csv_path):
                os.remove(csv_path)

            serialized_result = self._serialize_numpy(result)

            await self._update_job_and_notify(
                job_id,
                "completed",
                "Analysis complete",
                notification_service,
                serialized_result,
            )

        except Exception as e:
            await self._update_job_and_notify(
                job_id, "failed", f"Error: {str(e)}", notification_service, error=str(e)
            )

            if os.path.exists(csv_path):
                os.remove(csv_path)

    async def process_file_upload_and_analysis(
        self,
        file,
        user_id: str,
        min_density: Optional[float] = None,
        prompt: Optional[str] = None,
        file_service=None,
        user_service=None,
        job_service=None,
        notification_service=None,
    ) -> Dict:
        """Complete file upload and analysis orchestration"""
        from io import StringIO

        import pandas as pd

        from models.job import JobConfiguration
        from models.job import JobStatus as JobStatusEnum
        from models.job import JobType

        try:
            # Read and validate file
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))

            if df.empty:
                raise ValueError("Dataset cannot be empty")

            # Create file record
            file_obj = file_service.create_file(
                user_id, file.filename, file.filename, df, "Initial upload"
            )

            # Update user records
            user_service.add_user_file(user_id, file_obj.file_id)
            user_service.update_user_activity(user_id)

            # Save temporary file for processing
            await file.seek(0)
            temp_path = await file_service.save_uploaded_file(file)

            # Create job configuration
            config = JobConfiguration(
                min_density=min_density,
                prompt=prompt,
                analysis_type="subgraph_analysis",
            )

            # Create analysis job
            job = job_service.create_job(
                user_id=user_id,
                file_id=file_obj.file_id,
                job_type=JobType.ANALYSIS,
                title=f"Analysis of {file.filename}",
                configuration=config,
            )

            return {
                "job_id": job.job_id,
                "temp_path": temp_path,
                "user_id": user_id,
                "filename": file.filename,
                "config": config,
            }

        except Exception as e:
            raise ValueError(f"Failed to process file upload: {str(e)}")

    async def run_analysis_with_tracking(
        self,
        job_id: str,
        temp_path: str,
        user_id: str,
        filename: str,
        min_density: Optional[float] = None,
        prompt: Optional[str] = None,
        job_service=None,
        user_service=None,
        notification_service=None,
    ):
        """Run analysis with proper status tracking and error handling"""
        from models.job import JobStatus as JobStatusEnum

        try:
            job_service.update_job_status(job_id, JobStatusEnum.RUNNING)

            await self.run_analysis(
                job_id, temp_path, notification_service, min_density, prompt
            )

            user_service.increment_user_analyses(user_id)
            job_service.update_job_status(job_id, JobStatusEnum.COMPLETED)

        except Exception as e:
            logger.error(f"Analysis failed for user {user_id}, file {filename}: {e}")
            job_service.update_job_status(job_id, JobStatusEnum.FAILED, str(e))
            raise


analysis_service = AnalysisService()
