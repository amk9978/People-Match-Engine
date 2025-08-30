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
from services.redis.redis_cache import RedisEmbeddingCache
from shared.shared import BUSINESS_FEATURES, FEATURE_COLUMN_MAPPING, FEATURES
from shared.util import sanitize_metrics, serialize_numpy

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for handling graph analysis operations"""

    def __init__(
        self,
        job_service: JobService = None,
        file_service: FileService = None,
        results_cache: RedisEmbeddingCache = None,
        matrix_cache: RedisEmbeddingCache = None,
        graph_cache: RedisEmbeddingCache = None,
    ):
        self.job_service = job_service or JobService()
        self.file_service = file_service or FileService()
        self.results_cache = results_cache or RedisEmbeddingCache(
            key_prefix="job_results"
        )
        self.matrix_cache = matrix_cache or RedisEmbeddingCache()
        self.graph_cache = graph_cache or RedisEmbeddingCache(key_prefix="graph_cache")

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

    def _smart_cache_invalidation(self, job_id: str, new_csv_path: str):
        """Smart cache invalidation: preserve reusable data, clear only what's invalid"""
        try:
            # CRITICAL: Clear job result cache first - prevents stale results
            self._clear_job_result_caches(job_id)

            self._clear_specific_graph_caches(job_id)

            try:
                new_df = pd.read_csv(new_csv_path)
                self._validate_matrix_caches(job_id, new_df)
            except Exception as e:
                logger.warning(
                    f"Could not validate matrix caches, falling back to partial clear: {e}"
                )
                self._clear_specific_matrix_caches(job_id)

            logger.info(f"Smart cache invalidation completed", extra={"job_id": job_id})

        except Exception as e:
            logger.warning(
                f"Warning: Smart cache invalidation failed for job, using safe fallback",
                extra={"job_id": job_id, "e": e},
            )
            try:
                self._clear_job_result_caches(job_id)
                logger.info("✅ Applied safe fallback cache clearing")
            except Exception as fallback_error:
                logger.error(f"Even fallback cache clearing failed: {fallback_error}")

    def _clear_job_result_caches(self, job_id: str):
        """Clear all job result caches - critical for preventing stale results"""
        try:
            results_cache_key = f"result_{job_id}"
            self.results_cache.delete(results_cache_key)
            job_result_key = f"job_result:{job_id}"
            self.job_service.cache.delete(job_result_key)

        except Exception as e:
            logger.warning(f"Error clearing job result caches: {e}")

    def _clear_specific_graph_caches(self, job_id: str):
        """Clear only specific graph-related caches, preserve job metadata"""
        try:

            # Only clear specific graph cache keys, not all job_id patterns
            specific_keys = [
                f"networkx_graph_{job_id}",
                f"causal_graph_complete_{job_id}",
                f"feature_embeddings_{job_id}",
                f"graph_data_{job_id}",
                f"graph_structure_{job_id}",
            ]

            cleared_count = 0
            for key in specific_keys:
                if self.graph_cache.delete(key) > 0:
                    cleared_count += 1

            logger.info(f"🗑️ Cleared {cleared_count} specific graph cache entries")

        except Exception as e:
            logger.warning(f"Error clearing specific graph caches: {e}")

    def _clear_specific_matrix_caches(self, job_id: str):
        """Clear only matrix caches, preserve everything else"""
        try:
            matrix_keys = [
                f"causal_graph_complete_{job_id}",
                f"persona_complementarity_matrix_complete_{job_id}",
                f"experience_complementarity_matrix_complete_{job_id}",
                f"role_complementarity_matrix_complete_{job_id}",
            ]

            cleared_count = 0
            for key in matrix_keys:
                if self.matrix_cache.delete(key) > 0:
                    cleared_count += 1
        except Exception as e:
            logger.warning(f"Error clearing specific matrix caches: {e}")

    def _validate_matrix_caches(self, job_id: str, new_df: pd.DataFrame):
        """Validate complementarity matrices - keep valid rows, remove invalid ones"""
        try:
            new_profiles = self._extract_current_profiles(new_df)
            matrix_types = FEATURES

            for matrix_type in matrix_types:
                self._validate_single_matrix(
                    job_id, matrix_type, new_profiles.get(matrix_type, set())
                )

        except Exception as e:
            logger.warning(f"Matrix validation failed: {e}")
            raise e

    def _extract_current_profiles(self, df: pd.DataFrame) -> dict:
        """Extract current profile vectors from the new dataset"""
        profiles = {feature: set() for feature in FEATURES}
        column_mapping = FEATURE_COLUMN_MAPPING

        for category, column in column_mapping.items():
            if column in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[column]):
                        profile = str(row[column]).strip()
                        if profile:
                            profiles[category].add(profile)

        return profiles

    def _validate_single_matrix(
        self, job_id: str, matrix_type: str, current_profiles: set
    ):
        """Validate a single complementarity matrix, removing invalid rows"""
        try:
            if matrix_type in BUSINESS_FEATURES:
                cache_key = f"causal_graph_complete_{job_id}"
            else:
                cache_key = f"{matrix_type}_complementarity_matrix_complete_{job_id}"

            cached_data = self.matrix_cache.get(cache_key)
            if not cached_data:
                logger.info(
                    f"No cached matrix found for {matrix_type}, skipping validation"
                )
                return

            matrix_data = json.loads(cached_data)

            if matrix_type in BUSINESS_FEATURES:
                category_data = matrix_data.get(matrix_type, {})
                valid_profiles = self._filter_valid_profiles(
                    category_data, current_profiles
                )
                if len(valid_profiles) != len(category_data):
                    matrix_data[matrix_type] = valid_profiles
                    self.matrix_cache.set(cache_key, json.dumps(matrix_data))
                    removed = len(category_data) - len(valid_profiles)
                    logger.info(
                        f"🔄 Updated {matrix_type} matrix: removed {removed} invalid profiles"
                    )
            else:
                valid_profiles = self._filter_valid_profiles(
                    matrix_data, current_profiles
                )
                if len(valid_profiles) != len(matrix_data):
                    self.matrix_cache.set(cache_key, json.dumps(valid_profiles))
                    removed = len(matrix_data) - len(valid_profiles)
                    logger.info(
                        f"🔄 Updated {matrix_type} matrix: removed {removed} invalid profiles"
                    )

        except Exception as e:
            logger.warning(f"Failed to validate {matrix_type} matrix: {e}")
            self.matrix_cache.delete(cache_key)
            logger.info(
                f"🗑️ Removed entire {matrix_type} matrix due to validation failure"
            )

    def _filter_valid_profiles(self, matrix_data: dict, current_profiles: set) -> dict:
        """Filter matrix data to keep only profiles that still exist in the dataset"""
        valid_data = {}

        for profile_key, profile_relationships in matrix_data.items():
            if profile_key in current_profiles:
                valid_relationships = {
                    target: score
                    for target, score in profile_relationships.items()
                    if target in current_profiles
                }
                if valid_relationships:
                    valid_data[profile_key] = valid_relationships

        return valid_data

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
            self._smart_cache_invalidation(job_id, csv_path)

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
            logger.info(f"📊 Loaded {len(graph_builder.df)} rows from {csv_path}")

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
