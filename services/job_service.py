import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from models.job import Job, JobConfiguration, JobResult, JobStats, JobStatus, JobType
from services.redis.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class JobService:
    """Job service with Redis operations"""

    def __init__(self, cache: RedisCache = None):
        self.cache = cache or RedisCache()
        self.job_prefix = "job:"
        self.job_result_prefix = "job_result:"
        self.user_jobs_prefix = "user_jobs:"
        self.file_jobs_prefix = "file_jobs:"
        self.jobs_set_key = "jobs_set"
        self.active_jobs_key = "active_jobs"

    def create_job(
        self,
        user_id: str,
        file_id: str,
        job_type: JobType = JobType.ANALYSIS,
        title: Optional[str] = None,
        description: Optional[str] = None,
        configuration: Optional[JobConfiguration] = None,
        job_id: Optional[str] = None,
        **kwargs,
    ) -> Job:
        """Create a new job"""
        job_id = job_id or str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            user_id=user_id,
            file_id=file_id,
            job_type=job_type,
            title=title,
            description=description,
            configuration=configuration or JobConfiguration(),
            **kwargs,
        )

        self._save_job(job)

        self.cache.sadd(self.jobs_set_key, job.job_id)
        self.cache.sadd(f"{self.user_jobs_prefix}{user_id}", job.job_id)
        self.cache.sadd(f"{self.file_jobs_prefix}{file_id}", job.job_id)

        if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
            self.cache.sadd(self.active_jobs_key, job.job_id)

        logger.info(f"Created job {job.job_id} for user {user_id}, file {file_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.cache.get(job_key)

        if not job_data:
            return None

        try:
            return Job.from_dict(json.loads(job_data))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing job data for {job_id}: {e}")
            return None

    def update_job(self, job: Job) -> bool:
        """Update job data"""
        job.updated_at = datetime.now()

        if job.is_active():
            self.cache.sadd(self.active_jobs_key, job.job_id)
        else:
            self.cache.srem(self.active_jobs_key, job.job_id)

        return self._save_job(job)

    def update_job_status(
        self, job_id: str, status: JobStatus, message: Optional[str] = None
    ) -> bool:
        """Update job status"""
        job = self.get_job(job_id)
        if not job:
            return False

        job.update_status(status, message)
        return self.update_job(job)

    def update_job_progress(self, job_id: str, progress: float) -> bool:
        """Update job progress"""
        job = self.get_job(job_id)
        if not job:
            return False

        job.update_progress(progress)
        return self.update_job(job)

    def set_job_result(self, job_id: str, result: JobResult) -> bool:
        """Set job result and mark as completed"""
        job = self.get_job(job_id)
        if not job:
            return False

        job.result = result
        job.update_status(JobStatus.COMPLETED)

        result_key = f"{self.job_result_prefix}{job_id}"
        try:
            self.cache.set(result_key, json.dumps(result.to_dict()))
        except Exception as e:
            logger.error(f"Error saving job result for {job_id}: {e}")

        return self.update_job(job)

    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job result"""
        result_key = f"{self.job_result_prefix}{job_id}"
        result_data = self.cache.get(result_key)

        if not result_data:
            job = self.get_job(job_id)
            return job.result if job else None

        try:
            return JobResult.from_dict(json.loads(result_data))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing job result for {job_id}: {e}")
            return None

    def get_user_jobs(
        self,
        user_id: str,
        status_filter: Optional[JobStatus] = None,
        limit: Optional[int] = None,
    ) -> List[Job]:
        """Get jobs for a user"""
        jobs_key = f"{self.user_jobs_prefix}{user_id}"
        job_ids = self.cache.smembers(jobs_key)

        jobs = []
        for job_id in job_ids:
            job_id_str = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
            job = self.get_job(job_id_str)
            if job:

                if status_filter is None or job.status == status_filter:
                    jobs.append(job)

        jobs.sort(key=lambda j: j.created_at, reverse=True)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def get_file_jobs(
        self, file_id: str, status_filter: Optional[JobStatus] = None
    ) -> List[Job]:
        """Get jobs for a file"""
        jobs_key = f"{self.file_jobs_prefix}{file_id}"
        job_ids = self.cache.smembers(jobs_key)

        jobs = []
        for job_id in job_ids:
            job_id_str = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
            job = self.get_job(job_id_str)
            if job:

                if status_filter is None or job.status == status_filter:
                    jobs.append(job)

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def get_active_jobs(self) -> List[Job]:
        """Get all active jobs"""
        job_ids = self.cache.smembers(self.active_jobs_key)

        jobs = []
        for job_id in job_ids:
            job_id_str = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
            job = self.get_job(job_id_str)
            if job and job.is_active():
                jobs.append(job)
            elif job and not job.is_active():

                self.cache.srem(self.active_jobs_key, job_id)

        jobs.sort(key=lambda j: j.created_at)
        return jobs

    def get_jobs_by_status(
        self, status: JobStatus, limit: Optional[int] = None
    ) -> List[Job]:
        """Get jobs by status"""
        all_job_ids = self.cache.smembers(self.jobs_set_key)

        jobs = []
        for job_id in all_job_ids:
            job_id_str = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
            job = self.get_job(job_id_str)
            if job and job.status == status:
                jobs.append(job)
                if limit and len(jobs) >= limit:
                    break

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete job and its result"""
        try:
            job = self.get_job(job_id)
            if not job:
                return False

            job_key = f"{self.job_prefix}{job_id}"
            self.cache.delete(job_key)

            result_key = f"{self.job_result_prefix}{job_id}"
            self.cache.delete(result_key)

            self.cache.srem(self.jobs_set_key, job_id)
            self.cache.srem(f"{self.user_jobs_prefix}{job.user_id}", job_id)
            self.cache.srem(f"{self.file_jobs_prefix}{job.file_id}", job_id)
            self.cache.srem(self.active_jobs_key, job_id)

            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            return False

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up jobs older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        all_job_ids = self.cache.smembers(self.jobs_set_key)

        deleted_count = 0
        for job_id in all_job_ids:
            job_id_str = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
            job = self.get_job(job_id_str)

            if job and job.created_at < cutoff_date and job.is_finished():
                if self.delete_job(job_id_str):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old jobs")
        return deleted_count

    def get_job_stats(self, user_id: Optional[str] = None) -> JobStats:
        """Get job statistics"""
        if user_id:
            jobs = self.get_user_jobs(user_id)
        else:

            all_job_ids = self.cache.smembers(self.jobs_set_key)
            jobs = []
            for job_id in all_job_ids:
                job_id_str = (
                    job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
                )
                job = self.get_job(job_id_str)
                if job:
                    jobs.append(job)

        stats = JobStats(user_id=user_id or "all")
        stats.total_jobs = len(jobs)

        total_execution_time = 0.0
        execution_times = []

        for job in jobs:
            if job.status == JobStatus.COMPLETED:
                stats.completed_jobs += 1
                duration = job.get_duration()
                if duration:
                    execution_times.append(duration)
                    total_execution_time += duration
            elif job.status == JobStatus.FAILED:
                stats.failed_jobs += 1
            elif job.is_active():
                stats.running_jobs += 1

        stats.total_execution_time = total_execution_time
        if execution_times:
            stats.avg_execution_time = sum(execution_times) / len(execution_times)

        return stats

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        return self.update_job_status(
            job_id, JobStatus.CANCELLED, "Job cancelled by user"
        )

    def restart_job(self, job_id: str) -> bool:
        """Restart a failed job"""
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.FAILED:
            return False

        job.update_status(JobStatus.QUEUED)
        job.error_message = None
        job.progress = 0.0
        job.started_at = None
        job.completed_at = None

        return self.update_job(job)

    def _save_job(self, job: Job) -> bool:
        """Save job data"""
        try:
            job_key = f"{self.job_prefix}{job.job_id}"
            self.cache.set(job_key, json.dumps(job.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Error saving job {job.job_id}: {e}")
            return False

    def list_jobs_with_details(
        self, status: Optional[str] = None, limit: Optional[int] = 50
    ) -> Dict:
        """List all jobs with optional status filter and detailed formatting"""
        try:
            status_filter = None
            if status:
                status_filter = JobStatus(status)

            jobs = (
                self.get_jobs_by_status(status_filter, limit) if status_filter else []
            )
            if not status_filter:
                jobs = self.get_active_jobs()

            job_data = []
            for job in jobs:
                job_data.append(
                    {
                        "job_id": job.job_id,
                        "user_id": job.user_id,
                        "file_id": job.file_id,
                        "title": job.title,
                        "status": job.status.value,
                        "progress": job.progress,
                        "created_at": job.created_at.isoformat(),
                        "duration": job.get_duration(),
                    }
                )

            return {"jobs": job_data, "total": len(job_data), "status_code": 200}
        except Exception:
            jobs = self.get_active_jobs()
            return {
                "jobs": [job.to_dict() for job in jobs],
                "total": len(jobs),
                "status_code": 200,
            }

    def get_user_jobs_with_details(
        self, user_id: str, file_service, status: Optional[str] = None
    ) -> Dict:
        """Get all jobs for a specific user with detailed formatting"""
        try:
            status_filter = None
            if status:
                status_filter = JobStatus(status)

            jobs = self.get_user_jobs(user_id, status_filter)
            job_data = []

            for job in jobs:
                job_dict = {
                    "job_id": job.job_id,
                    "file_id": job.file_id,
                    "title": job.title,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "started_at": (
                        job.started_at.isoformat() if job.started_at else None
                    ),
                    "completed_at": (
                        job.completed_at.isoformat() if job.completed_at else None
                    ),
                    "error_message": job.error_message,
                    "configuration": (
                        job.configuration.to_dict() if job.configuration else {}
                    ),
                }

                file_obj = file_service.get_file(job.file_id)
                if file_obj:
                    job_dict["filename"] = file_obj.filename

                job_data.append(job_dict)

            return {
                "user_id": user_id,
                "jobs": job_data,
                "total_jobs": len(job_data),
                "status_code": 200,
            }
        except Exception as e:
            jobs = self.get_user_jobs(user_id)
            return {"user_id": user_id, "jobs": jobs, "status_code": 500}
