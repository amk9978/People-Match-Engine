#!/usr/bin/env python3
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

import uvicorn
from dotenv import load_dotenv

import settings  # noqa: F401, F403
from services.user_service_clean import UserService

load_dotenv()
logger = logging.getLogger(__name__)

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
from pydantic import BaseModel

from controllers.dataset_controller_clean import DatasetController
from controllers.user_controller_clean import UserController
from presentation.models import AnalysisResponse, JobStatus
from services.analysis_service import analysis_service, file_service
from services.cache_service import cache_service
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

# Initialize controllers
user_controller = UserController()
dataset_controller = DatasetController()


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


@app.get("/health")
async def health_check():
    """Detailed health check"""
    cache_health = cache_service.health_check()
    if cache_health["status"] != "healthy":
        raise HTTPException(
            status_code=503, detail=f"Service unhealthy: {cache_health}"
        )

    return {
        "status": "healthy",
        "redis": cache_health["redis"],
        "timestamp": datetime.now(),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        min_density: Optional[float] = Form(None),
        prompt: Optional[str] = Form(None),
        file_id: Optional[str] = Form(None),
        user_id: str = Header(..., alias="X-User-ID"),
):
    """Upload CSV file and start graph analysis with optional hyperparameter tuning

    Parameters:
    - file: CSV file with professional data
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
        dataset_result = await dataset_controller.upload_dataset(user_id, file)

        await file.seek(0)
        temp_path = await file_service.save_uploaded_file(file)

        job_id = analysis_service.create_job(
            file.filename, min_density, prompt, file_id, user_id
        )

        async def run_analysis_with_tracking():
            try:
                await analysis_service.run_analysis(
                    job_id, temp_path, notification_service, min_density, prompt
                )
                user_service_clean = UserService()
                user_service_clean.update_file_analysis(user_id, file.filename)
            except Exception as e:
                logger.info(
                    f"Analysis failed for user {user_id}, file {file.filename}: {e}"
                )
                raise

        background_tasks.add_task(run_analysis_with_tracking)

        return AnalysisResponse(
            job_id=job_id,
            status="queued",
            message="Analysis started successfully",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = analysis_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        timestamp=job["timestamp"],
    )


@app.get("/jobs")
async def list_jobs():
    """List all analysis jobs"""
    return analysis_service.get_all_jobs()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if not analysis_service.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get stored job result from Redis"""
    result = analysis_service.get_job_result(job_id)

    if not result:
        raise HTTPException(
            status_code=404, detail="Job result not found or job not completed"
        )

    return {"job_id": job_id, "result": result}


@app.get("/users/{user_id}/jobs")
async def get_user_jobs(user_id: str):
    """Get all jobs for a specific user"""
    jobs = analysis_service.get_jobs_by_user(user_id)
    return {"user_id": user_id, "jobs": jobs}


# Cache endpoints
@app.get("/cache/info")
async def get_cache_info():
    """Get Redis cache information"""
    cache_info = cache_service.get_cache_info()
    if "error" in cache_info:
        raise HTTPException(
            status_code=500, detail=f"Cache error: {cache_info['error']}"
        )
    return cache_info


@app.delete("/cache")
async def clear_cache():
    """Clear all embeddings from Redis cache"""
    result = cache_service.clear_cache()
    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Cache error: {result['error']}")
    return result


# WebSocket management endpoints
@app.get("/ws/connections")
async def get_websocket_connections():
    """Get list of active WebSocket connections"""
    return notification_service.get_connections_info()


# User management endpoints
@app.get("/users/me")
async def get_current_user(user_id: str = Header(..., alias="X-User-ID")):
    """Get current user profile and stats"""
    return await user_controller.get_user_stats(user_id)


@app.get("/users/me/files")
async def get_user_files(user_id: str = Header(..., alias="X-User-ID")):
    """Get user's uploaded files"""
    files = await user_controller.get_user_files(user_id)
    return {"user_id": user_id, "files": files, "total_files": len(files)}


@app.delete("/users/me/files/{filename}")
async def delete_user_file(
        filename: str, user_id: str = Header(..., alias="X-User-ID")
):
    """Delete a user's uploaded file"""
    return await user_controller.remove_user_file(user_id, filename)


@app.get("/admin/users")
async def list_all_users():
    """Admin endpoint to list all users"""
    users = await user_controller.list_users()
    return {"total_users": len(users), "users": users}


# Dataset modification endpoints
@app.get("/datasets")
async def list_user_datasets(user_id: str = Header(..., alias="X-User-ID")):
    """List all datasets for the current user"""
    datasets = await dataset_controller.list_user_datasets(user_id)
    return {"user_id": user_id, "datasets": datasets, "total_datasets": len(datasets)}


@app.get("/datasets/{filename}")
async def get_dataset_info(
        filename: str, user_id: str = Header(..., alias="X-User-ID")
):
    """Get detailed information about a specific dataset"""
    return await dataset_controller.get_dataset_info(user_id, filename)


@app.get("/datasets/{filename}/preview")
async def get_dataset_preview(
        filename: str,
        user_id: str = Header(..., alias="X-User-ID"),
        version_id: Optional[str] = Query(
            None, description="Version ID to preview (default: current)"
        ),
        limit: int = Query(10, description="Number of rows to preview"),
):
    """Get preview of dataset with sample rows"""
    return await dataset_controller.get_dataset_preview(
        user_id, filename, version_id, limit
    )


@app.post("/datasets/{filename}/add-rows")
async def add_rows_to_dataset(
        filename: str, request: BaseModel, user_id: str = Header(..., alias="X-User-ID")
):
    """Add new rows to an existing dataset"""
    from models.requests import AddRowsRequest as AddRowsRequestModel

    request_obj = AddRowsRequestModel(
        rows=request.rows, description=request.description
    )
    result = await dataset_controller.add_rows(user_id, filename, request_obj)
    return {"message": "Rows added successfully", "result": result}


@app.post("/datasets/{filename}/delete-rows")
async def delete_rows_from_dataset(
        filename: str, request: BaseModel, user_id: str = Header(..., alias="X-User-ID")
):
    """Delete specific rows by index from dataset"""
    from models.requests import DeleteRowsRequest as DeleteRowsRequestModel

    request_obj = DeleteRowsRequestModel(
        row_indices=request.row_indices, description=request.description
    )
    result = await dataset_controller.delete_rows(user_id, filename, request_obj)
    return {
        "message": f"Deleted {len(request.row_indices)} rows successfully",
        "result": result,
    }


@app.post("/datasets/{filename}/delete-rows-by-criteria")
async def delete_rows_by_criteria(
        filename: str, request: BaseModel, user_id: str = Header(..., alias="X-User-ID")
):
    """Delete rows matching specific criteria"""
    from models.requests import (
        DeleteRowsByCriteriaRequest as DeleteRowsByCriteriaRequestModel,
    )

    request_obj = DeleteRowsByCriteriaRequestModel(
        criteria=request.criteria, description=request.description
    )
    result = await dataset_controller.delete_rows_by_criteria(
        user_id, filename, request_obj
    )
    return {
        "message": f"Deleted {result['changes']['deleted_rows']} rows matching criteria",
        "result": result,
    }


@app.post("/datasets/{filename}/revert/{version_id}")
async def revert_dataset_to_version(
        filename: str,
        version_id: str,
        user_id: str = Header(..., alias="X-User-ID"),
        description: Optional[str] = Query(
            None, description="Description of the revert operation"
        ),
):
    """Revert dataset to a specific version"""
    result = await dataset_controller.revert_dataset(
        user_id, filename, version_id, description
    )
    return {
        "message": f"Successfully reverted to version {version_id}",
        "result": result,
    }


@app.get("/datasets/{filename}/diff")
async def get_version_diff(
        filename: str,
        version1: str = Query(..., description="First version ID"),
        version2: str = Query(..., description="Second version ID"),
        user_id: str = Header(..., alias="X-User-ID"),
):
    """Get differences between two versions of a dataset"""
    return await dataset_controller.compare_versions(
        user_id, filename, version1, version2
    )


@app.post("/analyze-modified/{filename}")
async def analyze_modified_dataset(
        filename: str,
        background_tasks: BackgroundTasks,
        user_id: str = Header(..., alias="X-User-ID"),
        version_id: Optional[str] = Query(
            None, description="Version to analyze (default: current)"
        ),
        min_density: Optional[float] = None,
        prompt: Optional[str] = None,
):
    """Run analysis on a modified version of dataset"""
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    try:
        from services.dataset_service_clean import DatasetService

        dataset_service_clean = DatasetService()
        df = dataset_service_clean.load_dataset(user_id, filename, version_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset version not found")

        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv")
        os.close(temp_fd)
        df.to_csv(temp_path, index=False)

        version_suffix = f"_v{version_id}" if version_id else "_current"
        job_filename = f"{filename}{version_suffix}"
        job_id = analysis_service.create_job(job_filename, min_density, prompt)

        async def run_analysis_with_tracking():
            try:
                await analysis_service.run_analysis(
                    job_id, temp_path, notification_service, min_density, prompt
                )
                user_service_clean = UserService()
                user_service_clean.update_file_analysis(user_id, filename)
                os.unlink(temp_path)
            except Exception as e:
                logger.info(f"Analysis failed for user {user_id}, file {filename}: {e}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        background_tasks.add_task(run_analysis_with_tracking)

        await user_controller.get_user_stats(user_id)

        return AnalysisResponse(
            job_id=job_id,
            status="queued",
            message=f"Analysis started for {job_filename}",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start analysis: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
