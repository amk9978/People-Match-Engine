import logging
from datetime import datetime
from typing import Optional

import uvicorn
from dotenv import load_dotenv

import settings  # noqa: F401, F403
from services.file_service import FileService
from services.job_service import JobService
from services.user_service import UserService

load_dotenv()
logger = logging.getLogger(__name__)

from io import StringIO

import pandas as pd
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from models.job import JobConfiguration
from models.job import JobStatus as JobStatusEnum
from models.job import JobType
from presentation.models import AnalysisResponse, JobStatus
from services.analysis_service import analysis_service
from services.notification_service import notification_service

app = FastAPI(
    title="Graph Matcher API",
    description="Dense subgraph analysis with multi-feature embedding matching",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_service = UserService()
file_service = FileService()
job_service = JobService()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time job updates"""
    await notification_service.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await notification_service.send_personal_message(
                {"type": "echo", "message": f"Received: {data}"}, client_id
            )
    except WebSocketDisconnect:
        notification_service.disconnect(client_id)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Graph Matcher API is running", "status": "healthy"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_density: Optional[float] = Form(None),
    prompt: Optional[str] = Form(None),
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Upload CSV file and start graph analysis with optional hyperparameter tuning

    Parameters:
    - min_density: Minimum density threshold for subgraph extraction
    - prompt: Optional user intent to tune hyperparameters (e.g., "I want to hire for my startup", "I need peer networking", "I want business partnerships")
    - file_id: Optional file identifier for cache isolation (auto-generated if not provided)
    - X-User-ID: User identifier in header
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    if not file_service.validate_csv_file(file.filename):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:

        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset cannot be empty")

        file_obj = file_service.create_file(
            user_id, file.filename, file.filename, df, "Initial upload"
        )

        user_service.add_user_file(user_id, file_obj.file_id)
        user_service.update_user_activity(user_id)

        await file.seek(0)
        temp_path = await file_service.save_uploaded_file(file)

        config = JobConfiguration(
            min_density=min_density, prompt=prompt, analysis_type="subgraph_analysis"
        )

        job = job_service.create_job(
            user_id=user_id,
            file_id=file_obj.file_id,
            job_type=JobType.ANALYSIS,
            title=f"Analysis of {file.filename}",
            configuration=config,
        )

        async def run_analysis_with_tracking():
            try:
                job_service.update_job_status(job.job_id, JobStatusEnum.RUNNING)

                await analysis_service.run_analysis(
                    job.job_id, temp_path, notification_service, min_density, prompt
                )

                user_service.increment_user_analyses(user_id)
                job_service.update_job_status(job.job_id, JobStatusEnum.COMPLETED)

            except Exception as e:
                logger.info(
                    f"Analysis failed for user {user_id}, file {file.filename}: {e}"
                )
                job_service.update_job_status(job.job_id, JobStatusEnum.FAILED, str(e))
                raise

        background_tasks.add_task(run_analysis_with_tracking)

        return AnalysisResponse(
            job_id=job.job_id,
            status="queued",
            message="Analysis started successfully",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = job_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job.status.value,
        progress=job.progress,
        result=job.result.to_dict() if job.result else None,
        error=job.error_message,
        timestamp=job.created_at,
    )


@app.get("/jobs")
async def list_jobs():
    """List all analysis jobs"""
    jobs = job_service.get_active_jobs()
    return {"jobs": [job.to_dict() for job in jobs], "total": len(jobs)}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if not job_service.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get stored job result from Redis"""
    result = job_service.get_job_result(job_id)

    if not result:
        raise HTTPException(
            status_code=404, detail="Job result not found or job not completed"
        )

    return {"job_id": job_id, "result": result}


@app.get("/users/{user_id}/jobs")
async def get_user_jobs(user_id: str, status: Optional[str] = Query(None)):
    """Get all jobs for a specific user"""
    try:
        status_filter = None
        if status:
            status_filter = JobStatusEnum(status)

        jobs = job_service.get_user_jobs(user_id, status_filter)
        job_data = []

        for job in jobs:
            job_dict = {
                "job_id": job.job_id,
                "file_id": job.file_id,
                "title": job.title,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
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

        return {"user_id": user_id, "jobs": job_data, "total_jobs": len(job_data)}
    except Exception as e:

        jobs = job_service.get_user_jobs(user_id)
        return {"user_id": user_id, "jobs": jobs}


@app.get("/ws/connections")
async def get_websocket_connections():
    """Get list of active WebSocket connections"""
    return notification_service.get_connections_info()


@app.get("/users/me")
async def get_current_user(user_id: str = Header(..., alias="X-User-ID")):
    """Get current user profile and stats"""
    user_service.update_user_activity(user_id)
    stats = user_service.get_user_stats(user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    return stats.to_dict()


@app.get("/users/me/files")
async def get_user_files(user_id: str = Header(..., alias="X-User-ID")):
    """Get user's uploaded files"""
    user_service.update_user_activity(user_id)
    file_ids = user_service.get_user_files(user_id)

    files = []
    for file_id in file_ids:
        file_obj = file_service.get_file(file_id)
        if file_obj:
            stats = file_service.get_file_stats(file_id)
            files.append(
                {
                    "file_id": file_obj.file_id,
                    "filename": file_obj.filename,
                    "created_at": file_obj.created_at.isoformat(),
                    "updated_at": file_obj.updated_at.isoformat(),
                    "total_versions": file_obj.total_versions,
                    "total_jobs": file_obj.total_jobs,
                    "file_size": stats.current_size if stats else 0,
                }
            )

    return {"user_id": user_id, "files": files, "total_files": len(files)}


@app.delete("/users/me/files/{file_id}")
async def delete_user_file(file_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Delete a user's uploaded file"""
    user_service.update_user_activity(user_id)

    user_files = user_service.get_user_files(user_id)
    if file_id not in user_files:
        raise HTTPException(status_code=404, detail="File not found")

    if file_service.delete_file(file_id):
        user_service.remove_user_file(user_id, file_id)
        return {"message": "File deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete file")


@app.get("/admin/users")
async def list_all_users():
    """Admin endpoint to list all users"""
    users = user_service.list_users(limit=100)
    return {"total_users": len(users), "users": [u.to_dict() for u in users]}


@app.get("/files")
async def list_user_files(user_id: str = Header(..., alias="X-User-ID")):
    """List all files for the current user"""
    user_service.update_user_activity(user_id)
    files = file_service.get_user_files(user_id)

    datasets = []
    for file_obj in files:
        stats = file_service.get_file_stats(file_obj.file_id)
        datasets.append(
            {
                "file_id": file_obj.file_id,
                "filename": file_obj.filename,
                "created_at": file_obj.created_at.isoformat(),
                "updated_at": file_obj.updated_at.isoformat(),
                "total_versions": file_obj.total_versions,
                "total_jobs": file_obj.total_jobs,
                "current_size": stats.current_size if stats else 0,
            }
        )

    return {"user_id": user_id, "files": datasets, "total_files": len(datasets)}


@app.get("/files/{file_id}")
async def get_file_info(file_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Get detailed information about a specific file"""
    user_service.update_user_activity(user_id)

    user_files = user_service.get_user_files(user_id)
    if file_id not in user_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_obj = file_service.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    stats = file_service.get_file_stats(file_id)
    versions = file_service.get_file_versions(file_id)

    return {
        "file_id": file_obj.file_id,
        "filename": file_obj.filename,
        "created_at": file_obj.created_at.isoformat(),
        "updated_at": file_obj.updated_at.isoformat(),
        "total_versions": len(versions),
        "total_jobs": file_obj.total_jobs,
        "current_size": stats.current_size if stats else 0,
        "versions": [
            {
                "version_id": v.version_id,
                "version_number": v.version_number,
                "created_at": v.created_at.isoformat(),
                "description": v.description,
                "row_count": v.row_count,
                "column_count": v.column_count,
            }
            for v in versions
        ],
    }


@app.get("/files/{file_id}/preview")
async def get_file_preview(
    file_id: str,
    user_id: str = Header(..., alias="X-User-ID"),
    version_id: Optional[str] = Query(
        None, description="Version ID to preview (default: current)"
    ),
    limit: int = Query(10, description="Number of rows to preview"),
):
    """Get preview of file data with sample rows"""
    user_service.update_user_activity(user_id)

    user_files = user_service.get_user_files(user_id)
    if file_id not in user_files:
        raise HTTPException(status_code=404, detail="File not found")

    df = file_service.load_file_data(file_id, version_id)
    if df is None:
        raise HTTPException(status_code=404, detail="File data not found")

    preview_df = df.head(limit)

    return {
        "file_id": file_id,
        "version_id": version_id,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "preview_rows": preview_df.to_dict("records"),
        "preview_count": len(preview_df),
    }


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None), limit: Optional[int] = Query(50)
):
    """List all jobs with optional status filter"""
    try:
        status_filter = None
        if status:
            status_filter = JobStatusEnum(status)

        jobs = (
            job_service.get_jobs_by_status(status_filter, limit)
            if status_filter
            else []
        )
        if not status_filter:
            jobs = job_service.get_active_jobs()

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

        return {"jobs": job_data, "total": len(job_data)}
    except Exception:

        jobs = job_service.get_active_jobs()
        return {"jobs": [job.to_dict() for job in jobs], "total": len(jobs)}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Cancel a running job"""

    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this job")

    if job_service.cancel_job(job_id):
        return {"message": "Job cancelled successfully"}
    else:
        raise HTTPException(
            status_code=400, detail="Cannot cancel job in current state"
        )


@app.post("/jobs/{job_id}/restart")
async def restart_job(job_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Restart a failed job"""

    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="Not authorized to restart this job"
        )

    if job_service.restart_job(job_id):
        return {"message": "Job restarted successfully"}
    else:
        raise HTTPException(
            status_code=400, detail="Cannot restart job in current state"
        )


@app.get("/jobs/stats")
async def get_job_stats(user_id: Optional[str] = Header(None, alias="X-User-ID")):
    """Get job statistics"""
    stats = job_service.get_job_stats(user_id)
    return stats.to_dict()


@app.delete("/jobs/cleanup")
async def cleanup_old_jobs(days: int = Query(30, description="Days to keep jobs")):
    """Admin endpoint to cleanup old completed jobs"""
    deleted_count = job_service.cleanup_old_jobs(days)
    return {"message": f"Cleaned up {deleted_count} old jobs"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
