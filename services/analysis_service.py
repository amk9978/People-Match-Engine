import json
import logging
import os
from datetime import datetime
from io import StringIO
from typing import Dict, Optional

import pandas as pd

from models.job import JobConfiguration, JobResult, JobStatus, JobType
from services.file_service import FileService
from services.graph.graph_builder import GraphBuilder
from services.job_service import JobService
from services.cache.cache import Cache
from services.cache.factory import get_cache_backend
from shared.util import sanitize_metrics, serialize_numpy

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for handling graph analysis operations"""

    def __init__(
        self,
        job_service: JobService = None,
        file_service: FileService = None,
        results_cache: Cache = None,
        graph_cache: Cache = None,
    ):
        self.job_service = job_service or JobService()
        self.file_service = file_service or FileService()
        backend = get_cache_backend()
        self.results_cache = results_cache or Cache(backend, "job_results")
        self.graph_cache = graph_cache or Cache(backend, "graph_cache")

    def get_filename_by_file_id(self, file_id: str) -> Optional[str]:
        """Get filename by file_id - now uses FileService"""
        try:
            file_obj = self.file_service.get_file(file_id)
            return file_obj.filename if file_obj else None
        except Exception as e:
            logger.info(
                f"Warning: Could not retrieve filename for file_id {file_id}: {e}"
            )
            return None

    def get_file_id_by_filename(self, user_id: str, filename: str) -> Optional[str]:
        """Get file_id by filename - now uses FileService"""
        try:
            file_obj = self.file_service.get_file_by_name(user_id, filename)
            return file_obj.file_id if file_obj else None
        except Exception as e:
            logger.info(
                f"Warning: Could not retrieve file_id for filename {filename}: {e}"
            )
            return None

    def _store_job_result(self, job_id: str, result: Dict):
        """Store completed job result in Redis"""
        try:
            cache_key = f"result_{job_id}"
            self.results_cache.set(cache_key, json.dumps(result))
        except Exception as e:
            logger.info(f"Warning: Could not store job result for {job_id}: {e}")

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
                    notification_data["result"] = serialize_numpy(result)
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
        metrics = sanitize_metrics(result.get("metrics", {}))
        job_result = JobResult(
            job_id=job_id,
            result_data=serialize_numpy(result),
            metrics=metrics,
            artifacts=result.get("artifacts", []),
            execution_time=result.get("execution_time", 0.0),
        )
        self.job_service.set_job_result(job_id, job_result)

    def _invalidate_job_caches(self, job_id: str) -> None:
        """Drop everything a rerun must recompute, keeping what stays valid.

        Complementarity scores are keyed by profile pair, so a changed profile
        yields a different key and stale rows are never read. Only the artifacts
        keyed by job need clearing."""
        self._clear_job_result_caches(job_id)
        self._clear_specific_graph_caches(job_id)

    def _clear_job_result_caches(self, job_id: str):
        self.results_cache.delete(f"result_{job_id}")
        self.job_service.cache.delete(f"job_result:{job_id}")

    def _clear_specific_graph_caches(self, job_id: str):
        keys = [
            f"networkx_graph_{job_id}",
            f"feature_embeddings_{job_id}",
        ]
        cleared = sum(1 for key in keys if self.graph_cache.delete(key) > 0)
        logger.info(f"Cleared {cleared} graph cache entries for job {job_id}")

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
            if os.path.exists(csv_path):
                temp_df = pd.read_csv(csv_path)
                sample_names = (
                    list(temp_df["Person Name"].head(3))
                    if "Person Name" in temp_df.columns
                    else []
                )

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Validating caches...",
                notification_service,
            )
            self._invalidate_job_caches(job_id)

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

            # Force fresh data load - critical for updated jobs with new files
            graph_builder.load_data()
            logger.info(f"Loaded {len(graph_builder.df)} rows from {csv_path}")

            await self._update_job_and_notify(
                job_id,
                "processing",
                "Preprocessing tags and embeddings...",
                notification_service,
            )

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

            result["debug_info"] = {
                "csv_path": csv_path,
                "dataset_rows": len(graph_builder.df),
                "job_id": job_id,
                "file_id": file_id if "file_id" in locals() else "unknown",
                "analysis_timestamp": datetime.now().isoformat(),
            }

            if hasattr(graph_builder, "df") and not graph_builder.df.empty:
                sample_names = (
                    list(graph_builder.df["Person Name"].head(3))
                    if "Person Name" in graph_builder.df.columns
                    else []
                )

            if os.path.exists(csv_path):
                os.remove(csv_path)

            serialized_result = serialize_numpy(result)

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
        job_id: Optional[str] = None,
        file_service=None,
        user_service=None,
        job_service=None,
        notification_service=None,
    ) -> Dict:
        """Complete file upload and analysis orchestration"""
        try:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode("utf-8")))

            if df.empty:
                raise ValueError("Dataset cannot be empty")

            file_obj = file_service.create_file(
                user_id, file.filename, file.filename, df, "New file upload"
            )

            user_service.add_user_file(user_id, file_obj.file_id)
            user_service.update_user_activity(user_id)

            await file.seek(0)
            temp_path = await file_service.save_uploaded_file(file)

            config = JobConfiguration(
                min_density=min_density,
                prompt=prompt,
                analysis_type="subgraph_analysis",
            )

            if job_id:
                existing_job = job_service.get_job(job_id)
                if existing_job:
                    self._clear_job_result_caches(job_id)

                    existing_job.file_id = file_obj.file_id
                    existing_job.configuration = config
                    existing_job.title = f"Analysis of {file.filename}"
                    existing_job.updated_at = datetime.now()

                    job_service._save_job(existing_job)
                    job = existing_job
                else:
                    job = job_service.create_job(
                        user_id=user_id,
                        file_id=file_obj.file_id,
                        job_type=JobType.ANALYSIS,
                        title=f"Analysis of {file.filename}",
                        configuration=config,
                        job_id=job_id,
                    )
            else:
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
