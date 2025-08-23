#!/usr/bin/env python3

from datetime import datetime
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

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
    min_density: Optional[float] = None,
):
    """Upload CSV file and start graph analysis"""
    if not file_service.validate_csv_file(file.filename):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        temp_path = await file_service.save_uploaded_file(file)

        job_id = analysis_service.create_job(file.filename, min_density)

        background_tasks.add_task(
            analysis_service.run_analysis,
            job_id,
            temp_path,
            notification_service,
            min_density,
        )

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
