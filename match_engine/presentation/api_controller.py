import logging
import sys
from datetime import datetime
from typing import Optional

import uvicorn
from dotenv import load_dotenv
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
from fastapi.responses import FileResponse

from match_engine import settings  # noqa: F401, F403
from match_engine.presentation.models import AnalysisResponse, JobStatus
from match_engine.services.analysis_service import AnalysisService
from match_engine.services.cache.cache import Cache
from match_engine.services.cache.factory import get_cache_backend
from match_engine.services.file_service import FileService
from match_engine.services.job_service import JobService
from match_engine.services.notification_service import NotificationService
from match_engine.services.scoring.weights import ExplicitWeights, WeightsError
from match_engine.services.user_service import UserService
from match_engine.shared.util import serialize_numpy

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_services():
    """Create all services with proper dependency injection"""

    backend = get_cache_backend()
    shared_cache = Cache(backend)
    results_cache = Cache(backend, "job_results")
    graph_cache = Cache(backend, "graph_cache")

    job_service = JobService(cache=shared_cache)
    file_service = FileService(cache=shared_cache)
    user_service = UserService(cache=shared_cache)

    analysis_service = AnalysisService(
        job_service=job_service,
        file_service=file_service,
        results_cache=results_cache,
        graph_cache=graph_cache,
    )

    notification_service = NotificationService()

    return {
        "user_service": user_service,
        "file_service": file_service,
        "job_service": job_service,
        "analysis_service": analysis_service,
        "notification_service": notification_service,
    }


services = create_services()
user_service = services["user_service"]
file_service = services["file_service"]
job_service = services["job_service"]
analysis_service = services["analysis_service"]
notification_service = services["notification_service"]

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


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time job updates with heartbeat support"""
    import asyncio
    import json

    await notification_service.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "resume_job":
                    job_id = message.get("job_id")
                    if job_id:
                        job = job_service.get_job(job_id)
                        if job:
                            await notification_service.send_personal_message(
                                {
                                    "type": "job_update",
                                    "job_id": job_id,
                                    "status": job.status.value,
                                    "progress": (
                                        str(job.progress)
                                        if job.progress is not None
                                        else None
                                    ),
                                    "result": (
                                        job.result.result_data
                                        if job.result
                                        and job.status.value == "completed"
                                        else None
                                    ),
                                },
                                client_id,
                            )
                            logger.info(
                                f"Client {client_id} resumed monitoring job {job_id}"
                            )
                        else:
                            await notification_service.send_personal_message(
                                {"type": "error", "message": f"Job {job_id} not found"},
                                client_id,
                            )
                else:
                    await notification_service.send_personal_message(
                        {"type": "echo", "message": f"Received: {data}"}, client_id
                    )
            except json.JSONDecodeError:
                await notification_service.send_personal_message(
                    {"type": "echo", "message": f"Received: {data}"}, client_id
                )
    except WebSocketDisconnect:
        notification_service.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        notification_service.disconnect(client_id)


@app.get("/")
async def root():
    """Serve the demo front end"""
    return FileResponse("frontend.html")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_density: Optional[float] = Form(None),
    prompt: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
    weights: Optional[str] = Form(None),
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Upload CSV file and start graph analysis with optional hyperparameter tuning

    Parameters:
    - min_density: Minimum density threshold for subgraph extraction
    - prompt: Optional user intent to tune hyperparameters (e.g., "I want to hire for my startup", "I need peer
        networking", "I want business partnerships")
    - job_id: Optional job identifier to reuse existing job (for reruns)
    - weights: Optional JSON object with a similarity and a complementarity map,
        each keyed by feature name, replacing the measured weights
    - file_id: Optional file identifier for cache isolation (auto-generated if not provided)
    - X-User-ID: User identifier in header
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    if not file_service.validate_csv_file(file.filename):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    chosen_weights = None
    if weights:
        try:
            chosen_weights = ExplicitWeights.parse(weights)
        except WeightsError as invalid:
            raise HTTPException(status_code=400, detail=str(invalid))

    try:
        analysis_result = await analysis_service.process_file_upload_and_analysis(
            file=file,
            user_id=user_id,
            min_density=min_density,
            prompt=prompt,
            job_id=job_id,
            file_service=file_service,
            user_service=user_service,
            job_service=job_service,
            notification_service=notification_service,
        )

        background_tasks.add_task(
            analysis_service.run_analysis_with_tracking,
            job_id=analysis_result["job_id"],
            temp_path=analysis_result["temp_path"],
            user_id=analysis_result["user_id"],
            filename=analysis_result["filename"],
            min_density=min_density,
            prompt=prompt,
            weights=chosen_weights,
            job_service=job_service,
            user_service=user_service,
            notification_service=notification_service,
        )

        return AnalysisResponse(
            job_id=analysis_result["job_id"],
            status="queued",
            message="Analysis started successfully",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


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


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = job_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job.status.value,
        progress=str(job.progress) if job.progress is not None else None,
        result=job.result.result_data if job.result else None,
        error=job.error_message,
        timestamp=job.created_at,
    )


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

    return {"job_id": job_id, "result": serialize_numpy(result.result_data)}


@app.get("/jobs/{job_id}/people/{position}/matches")
async def get_person_matches(job_id: str, position: int, top: int = Query(5, ge=1)):
    """Rank one person's best matches out of a completed run."""
    result = job_service.get_job_result(job_id)

    if not result:
        raise HTTPException(
            status_code=404, detail="Job result not found or job not completed"
        )

    matches = (result.result_data or {}).get("matches", {})
    person = matches.get(str(position))
    if person is None:
        raise HTTPException(
            status_code=404, detail=f"No person at position {position} in job {job_id}"
        )

    people = (result.result_data or {}).get("people", [])
    name = people[position] if position < len(people) else str(position)

    return {
        "job_id": job_id,
        "position": position,
        "name": name,
        "matches": serialize_numpy(person[:top]),
    }


@app.get("/users/{user_id}/jobs")
async def get_user_jobs(user_id: str, status: Optional[str] = Query(None)):
    """Get all jobs for a specific user"""
    result = job_service.get_user_jobs_with_details(user_id, file_service, status)
    if result.get("status_code", 200) != 200:
        raise HTTPException(status_code=500, detail="Failed to get user jobs")
    return {k: v for k, v in result.items() if k != "status_code"}


@app.get("/ws/connections")
async def get_websocket_connections():
    """Get list of active WebSocket connections"""
    return notification_service.get_connections_info()


@app.get("/users/me")
async def get_current_user(user_id: str = Header(..., alias="X-User-ID")):
    """Get current user profile and stats"""
    result = user_service.get_current_user_profile(user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@app.get("/users/me/files")
async def get_user_files(user_id: str = Header(..., alias="X-User-ID")):
    """Get user's uploaded files"""
    return user_service.get_user_files_with_details(user_id, file_service)


@app.delete("/users/me/files/{file_id}")
async def delete_user_file(file_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Delete a user's uploaded file"""
    result = user_service.delete_user_file_with_validation(
        user_id, file_id, file_service
    )
    if result.get("status_code") != 200:
        raise HTTPException(status_code=result["status_code"], detail=result["error"])
    return {"message": result["message"]}


@app.get("/admin/users")
async def list_all_users():
    """Admin endpoint to list all users"""
    users = user_service.list_users(limit=100)
    return {"total_users": len(users), "users": [u.to_dict() for u in users]}


@app.get("/files")
async def list_user_files(user_id: str = Header(..., alias="X-User-ID")):
    """List all files for the current user"""
    return file_service.list_user_files_with_details(user_id, user_service)


@app.get("/files/{file_id}")
async def get_file_info(file_id: str, user_id: str = Header(..., alias="X-User-ID")):
    """Get detailed information about a specific file"""
    result = file_service.get_file_info_with_validation(file_id, user_id, user_service)
    if result.get("status_code") != 200:
        raise HTTPException(status_code=result["status_code"], detail=result["error"])
    return {k: v for k, v in result.items() if k != "status_code"}


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
    result = file_service.get_file_preview_with_validation(
        file_id, user_id, user_service, version_id, limit
    )
    if result.get("status_code") != 200:
        raise HTTPException(status_code=result["status_code"], detail=result["error"])
    return {k: v for k, v in result.items() if k != "status_code"}


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None), limit: Optional[int] = Query(50)
):
    """List all jobs with optional status filter"""
    result = job_service.list_jobs_with_details(status, limit)
    return {k: v for k, v in result.items() if k != "status_code"}


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
